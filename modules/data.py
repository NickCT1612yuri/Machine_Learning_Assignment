"""Loading, cleaning, splitting and exploratory analysis of the text corpora."""
import html
import re
from collections import Counter
from dataclasses import dataclass, field

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from .config import DATASETS, QUICK_CAPS, SEED, VAL_FRACTION

SPLITS = ("fit", "val", "test")
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEGATION = re.compile(r"\b(?:not|no|never|nothing|nobody|none|cannot|nor|without)\b|n't\b")
BROKEN_ENTITY = re.compile(r"(?<!&)(#\d+|quot|amp|lt|gt|nbsp|hellip);")  # AG News drops the '&'
SPLIT_CONTRACTION = re.compile(r"\s+#39;(?=(?:s|t|re|ve|ll|m|d)\b)", re.I)  # "Google #39;s" -> "Google's"


def clean_text(text: str) -> str:
    """Repair HTML entities, drop markup and line-break backslashes, collapse whitespace."""
    text = SPLIT_CONTRACTION.sub("'", str(text))
    text = html.unescape(html.unescape(BROKEN_ENTITY.sub(r"&\1;", text)))
    text = re.sub(r"<[^>]+>", " ", text).replace("\\", " ")
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps contractions such as "don't"."""
    return TOKEN.findall(text.lower())


def has_negation(text: str) -> bool:
    return bool(NEGATION.search(text.lower()))


@dataclass
class Corpus:
    name: str
    text: dict          # split -> list[str]
    y: dict             # split -> np.ndarray of label ids
    labels: list        # label id -> name
    max_len: int
    quality: dict = field(default_factory=dict)


def _stratified_cap(texts, y, n):
    if n is None or len(y) <= n:
        return texts, y
    keep, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=SEED)
    keep = np.sort(keep)
    return [texts[i] for i in keep], y[keep]


def load_corpus(name: str, quick: bool = False) -> Corpus:
    """Download a corpus, clean it, cap it, and build leakage-free fit/val/test splits."""
    from datasets import load_dataset

    spec = DATASETS[name]
    raw = load_dataset(spec["hf_id"])
    label_feature = raw["train"].features["label"]
    labels = list(getattr(label_feature, "names", None) or [])
    if not labels:  # e.g. mteb/banking77 stores names in a separate column
        labels = [s for _, s in sorted({(int(i), s) for i, s in zip(raw["train"]["label"], raw["train"]["label_text"])})]

    quality, parts = {}, {}
    for split in ("train", "test"):
        rows = [(clean_text(t), int(y)) for t, y in zip(raw[split]["text"], raw[split]["label"]) if t is not None]
        kept = [(t, y) for t, y in rows if t]
        quality[f"{split}_missing_or_blank"] = len(raw[split]) - len(kept)
        quality[f"{split}_duplicates"] = len(kept) - len({t for t, _ in kept})
        cap = spec[f"{split}_cap"]
        if quick:
            cap = min(cap or np.inf, QUICK_CAPS[split == "test"])
        parts[split] = _stratified_cap([t for t, _ in kept], np.array([y for _, y in kept]), cap)

    texts, y = parts["train"]
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=VAL_FRACTION, stratify=y, random_state=SEED)

    # Exact-duplicate leakage: a text seen in test may not be used for fitting or selection.
    test_set = set(parts["test"][0])
    val_idx_clean = [i for i in val_idx if texts[i] not in test_set]
    val_set = {texts[i] for i in val_idx_clean}
    fit_idx_clean = [i for i in fit_idx if texts[i] not in test_set and texts[i] not in val_set]
    quality["removed_from_fit"] = len(fit_idx) - len(fit_idx_clean)
    quality["removed_from_val"] = len(val_idx) - len(val_idx_clean)

    text = {"fit": [texts[i] for i in fit_idx_clean], "val": [texts[i] for i in val_idx_clean], "test": parts["test"][0]}
    labels_y = {"fit": y[fit_idx_clean], "val": y[val_idx_clean], "test": parts["test"][1]}
    return Corpus(name, text, labels_y, labels, spec["max_len"], quality)


def run_eda(corpus: Corpus, out_dir):
    """Fit-split statistics and a three-panel figure: class balance, length, frequent words."""
    texts, y = corpus.text["fit"], corpus.y["fit"]
    tokens = [tokenize(t) for t in texts]
    lengths = np.array([len(t) for t in tokens])
    counts = np.bincount(y, minlength=len(corpus.labels))
    freq = Counter(w for doc in tokens for w in doc)
    content = Counter({w: n for w, n in freq.items() if w not in ENGLISH_STOP_WORDS})
    stats = {
        "n_fit": len(texts), "n_val": len(corpus.text["val"]), "n_test": len(corpus.text["test"]),
        "classes": len(corpus.labels), "min_class": int(counts.min()), "max_class": int(counts.max()),
        "length_mean": float(lengths.mean()), "length_median": float(np.median(lengths)),
        "length_p95": float(np.percentile(lengths, 95)), "length_max": int(lengths.max()),
        "vocabulary": len(freq), "hapax_share": sum(n == 1 for n in freq.values()) / len(freq),
        "negation_share": float(np.mean([has_negation(t) for t in texts])),
        "top_words": freq.most_common(15), "top_content_words": content.most_common(15),
        **corpus.quality,
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].bar(range(len(counts)), np.sort(counts)[::-1], color="#4C72B0")
    ax[0].set(title="Fit-split class sizes (sorted)", xlabel="class rank", ylabel="examples")
    ax[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[1].hist(lengths, bins=40, color="#4C72B0")
    ax[1].axvline(stats["length_p95"], color="#C44E52", ls="--", label=f"P95 = {stats['length_p95']:.0f}")
    ax[1].set(title="Words per text", xlabel="words", ylabel="texts")
    ax[1].legend()
    words, n = zip(*content.most_common(15))
    ax[2].barh(words[::-1], n[::-1], color="#4C72B0")
    ax[2].set(title="Top content words (stop words removed)", xlabel="count")
    fig.tight_layout()
    fig.savefig(out_dir / "eda.png", dpi=150)
    plt.close(fig)
    return stats
