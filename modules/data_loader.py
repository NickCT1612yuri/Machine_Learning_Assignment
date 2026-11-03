"""
Dataset loading and boolean text preprocessing.

Every dataset is returned in the same dictionary shape, so the feature,
classifier and deep-learning steps never need to know which corpus they run on:

    {
        "name", "title", "label_names", "n_classes", "max_len_cap",
        "train_raw", "test_raw",          # untouched strings (for error analysis)
        "train_text", "test_text",        # strings after clean_text(prep_cfg)
        "train_label", "test_label",      # np.int64 arrays
        "p95_len",                        # 95th percentile words per train document
        "preprocess_tag",                 # e.g. "sw0_tok1_pad1_neg0"
    }

All corpora are downloaded from public Hugging Face links; nothing is read from
Google Drive.
"""

from __future__ import annotations

import html
import os
import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from modules.config import DATASET_REGISTRY, SEED, PreprocessConfig

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY_LEFTOVER = re.compile(r"#\d+;|&[a-z]+;")
_BACKSLASH = re.compile(r"\\+")
_SPACE = re.compile(r"\s+")
# Words (with an optional contraction tail such as "n't" or "'s") or single punctuation marks.
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
# Cue tokens that attach to the following alphanumeric word when negation_marking=True.
NEGATION_CUES = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither",
    "nor", "cannot", "without", "n't",
})


def quick_mode() -> bool:
    """`TC_QUICK=1` shrinks every split so the whole suite runs in minutes (smoke tests)."""
    return os.environ.get("TC_QUICK", "0") == "1"


def normalize_text(text: str) -> str:
    """Lowercase, unescape HTML, drop URLs/tags/backslash artefacts, collapse whitespace."""
    text = html.unescape(str(text)).lower()
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY_LEFTOVER.sub(" ", text)
    text = _BACKSLASH.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def apply_negation_marking(tokens: list[str]) -> list[str]:
    """Join negation cues with the next alphanumeric token: not good -> not_good."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        cue = tok.lstrip("'") if tok.startswith("n'") else tok
        if cue in NEGATION_CUES or tok.endswith("n't"):
            # Prefer attaching to the next alphanumeric content word.
            j = i + 1
            while j < len(tokens) and not tokens[j][0].isalnum():
                out.append(tokens[j])
                j += 1
            if j < len(tokens) and tokens[j][0].isalnum():
                prefix = "not" if tok.endswith("n't") else cue
                out.append(f"{prefix}_{tokens[j]}")
                i = j + 1
                continue
        out.append(tok)
        i += 1
    return out


def clean_text(
    text: str,
    stop_word_removal: bool = False,
    keep_punctuation: bool = True,
    negation_marking: bool = False,
) -> str:
    """
    Clean one document.

    Negation marking (if enabled) runs after tokenization and *before* stop-word
    removal so cues like "not" are not dropped before they can attach. Punctuation
    is kept by default because Transformer encoders use it to scope negation and
    clause boundaries. A document is never returned empty.
    """
    base = normalize_text(text)
    needs_tokenize = stop_word_removal or negation_marking or not keep_punctuation
    if not needs_tokenize:
        return base

    tokens = _TOKEN.findall(base)
    if negation_marking:
        tokens = apply_negation_marking(tokens)
    if not keep_punctuation:
        tokens = [tok for tok in tokens if tok[0].isalnum() or "_" in tok]
    if stop_word_removal:
        # Keep marked phrases (not_good); drop plain stop words only.
        tokens = [tok for tok in tokens if "_" in tok or tok not in STOP_WORDS]
    cleaned = " ".join(tokens)
    if cleaned.strip():
        return cleaned
    fallback = " ".join(tok for tok in _TOKEN.findall(base) if tok[0].isalnum())
    return fallback or base or "empty"


def clean_corpus(
    texts,
    stop_word_removal: bool = False,
    keep_punctuation: bool = True,
    negation_marking: bool = False,
) -> list[str]:
    return [
        clean_text(t, stop_word_removal, keep_punctuation, negation_marking)
        for t in texts
    ]


def word_counts(texts) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts], dtype=np.int32)


def _stratified_cap(texts, labels, n, seed):
    texts = [str(t) for t in texts]
    labels = np.asarray(labels, dtype=np.int64)
    if n is None or len(texts) <= n:
        return texts, labels
    index, _ = train_test_split(
        np.arange(len(texts)), train_size=n, stratify=labels, random_state=seed
    )
    index = np.sort(index)
    return [texts[i] for i in index], labels[index]


def _names_from_label_text(split) -> list[str]:
    pairs = sorted({(int(lab), name) for lab, name in zip(split["label"], split["label_text"])})
    return [name for _, name in pairs]


@lru_cache(maxsize=None)
def _download(name: str):
    """Download one dataset from the Hugging Face Hub (cached per process)."""
    from datasets import load_dataset

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Choose one of: {', '.join(DATASET_REGISTRY)}")
    raw = load_dataset(DATASET_REGISTRY[name]["hf_id"])
    train, test = raw["train"], raw["test"]
    if "label_text" in train.column_names:
        names = _names_from_label_text(train)
    else:
        names = list(train.features["label"].names)
    return (
        tuple(names),
        tuple(train["text"]),
        tuple(int(x) for x in train["label"]),
        tuple(test["text"]),
        tuple(int(x) for x in test["label"]),
    )


def load_raw_dataset(name: str, seed: int = SEED) -> dict:
    """Official train/test split after stratified sub-sampling, before any cleaning."""
    cfg = DATASET_REGISTRY[name]
    names, tr_text, tr_lab, te_text, te_lab = _download(name)
    max_train, max_test = cfg["max_train"], cfg["max_test"]
    if quick_mode():
        max_train = min(max_train or 10**9, 1500)
        max_test = min(max_test or 10**9, 600)
    tr_text, tr_lab = _stratified_cap(tr_text, tr_lab, max_train, seed)
    te_text, te_lab = _stratified_cap(te_text, te_lab, max_test, seed + 1)

    def _drop_blank(texts, labels):
        keep = [i for i, t in enumerate(texts) if normalize_text(t)]
        return [texts[i] for i in keep], labels[keep]

    tr_text, tr_lab = _drop_blank(tr_text, tr_lab)
    te_text, te_lab = _drop_blank(te_text, te_lab)
    return {
        "name": name,
        "title": cfg["title"],
        "hf_id": cfg["hf_id"],
        "label_names": list(names),
        "n_classes": len(names),
        "max_len_cap": int(cfg["max_len_cap"]),
        "train_raw": tr_text,
        "test_raw": te_text,
        "train_label": tr_lab,
        "test_label": te_lab,
    }


def load_and_preprocess_dataset(
    dataset_name: str, prep_cfg: PreprocessConfig | None = None, seed: int = SEED
) -> dict:
    """Load one dataset and apply the boolean preprocessing configuration."""
    prep_cfg = prep_cfg or PreprocessConfig()
    data = load_raw_dataset(dataset_name, seed)
    data["train_text"] = clean_corpus(
        data["train_raw"], prep_cfg.stop_word_removal, True, prep_cfg.negation_marking
    )
    data["test_text"] = clean_corpus(
        data["test_raw"], prep_cfg.stop_word_removal, True, prep_cfg.negation_marking
    )
    data["p95_len"] = int(np.ceil(np.percentile(word_counts(data["train_text"]), 95)))
    data["preprocess_tag"] = prep_cfg.tag()
    data["preprocess"] = prep_cfg
    print(
        f"[data] {data['title']} ({prep_cfg.tag()}): {len(data['train_text'])} train / "
        f"{len(data['test_text'])} test, {data['n_classes']} classes, p95 words = {data['p95_len']}"
    )
    return data
