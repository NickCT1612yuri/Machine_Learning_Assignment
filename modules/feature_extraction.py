"""
Feature extraction with on-disk persistence in both .npy and .h5 (HDF5).

Embedding types
---------------
bow        CountVectorizer (sparse) + 30-d TruncatedSVD projection
tfidf      sublinear TfidfVectorizer (sparse) + 30-d TruncatedSVD projection
word2vec   skip-gram Word2Vec trained on train texts (300-d mean pool) + PCA-30;
           also saves word2vec.kv for BiLSTM / TextCNN / SetFit reuse
distilbert frozen distilbert-base-uncased, attention-masked mean pooling (768-d) + PCA-30
bge_small  frozen BAAI/bge-small-en-v1.5, CLS pooling + L2 norm (384-d) + PCA-30

Files written to features/<dataset>/<embedding>_<preprocess tag>/
-----------------------------------------------------------------
X_train.npy, X_test.npy              full dense matrix (dense embeddings only)
X_train_sparse.npz, X_test_sparse.npz full sparse matrix (bow / tfidf only)
X_train_pca30.npy, X_test_pca30.npy  30-d projection (SVD for sparse, PCA for dense)
y_train.npy, y_test.npy              integer labels
features.h5                          every array above in one HDF5 file
meta.json                            settings used to build the features
word2vec.kv                          gensim KeyedVectors (word2vec only)

How the boolean preprocessing switches reach the extractors
-----------------------------------------------------------
stop_word_removal / negation_marking  already applied by data_loader.clean_text
tokenization       sparse: True -> regex words + bigrams, False -> str.split unigrams
                   neural: True -> WordPiece subwords, False -> whole-word vocabulary
                   lookup (out-of-vocabulary words become [UNK], no subword split)
                   word2vec: always whitespace tokens (static word table)
padding            neural: True -> truncate at the 95th-percentile token length
                   (capped per dataset) and mask padding in attention and pooling;
                   False -> truncate at 10 tokens and pool over padding unmasked
                   sparse / word2vec mean-pool: not applicable
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from modules.config import (
    BERT_BATCH_SIZE,
    BERT_MODEL,
    BGE_MODEL,
    FEATURE_DIR,
    MAX_FEATURES,
    PCA_DIM,
    SEED,
    ExperimentConfig,
)
from modules.utils import get_device

SPARSE_TYPES = ("bow", "tfidf")
DENSE_MODELS = {
    "distilbert": {"model_id": BERT_MODEL, "pooling": "mean", "normalize": False},
    "bge_small": {"model_id": BGE_MODEL, "pooling": "cls", "normalize": True},
}
UNPADDED_MAX_LEN = 10


def cache_dir_for(dataset_name: str, embedding_type: str, tag: str, feature_root: Path = FEATURE_DIR) -> Path:
    return Path(feature_root) / dataset_name / f"{embedding_type}_{tag}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save(folder: Path, arrays: dict, sparse_arrays: dict, meta: dict, save_format: str) -> None:
    import h5py

    folder.mkdir(parents=True, exist_ok=True)
    for key, mat in sparse_arrays.items():
        sparse.save_npz(folder / f"{key}.npz", mat.tocsr())
    if save_format in ("npy", "both"):
        for key, arr in arrays.items():
            np.save(folder / f"{key}.npy", arr)
    if save_format in ("h5", "both"):
        with h5py.File(folder / "features.h5", "w") as h5:
            for key, arr in arrays.items():
                h5.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            for key, mat in sparse_arrays.items():
                grp = h5.create_group(key)
                csr = mat.tocsr()
                grp.create_dataset("data", data=csr.data, compression="gzip", compression_opts=4)
                grp.create_dataset("indices", data=csr.indices, compression="gzip", compression_opts=4)
                grp.create_dataset("indptr", data=csr.indptr)
                grp.attrs["shape"] = csr.shape
            h5.attrs["meta"] = json.dumps(meta)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def load_h5_features(path: Path) -> dict:
    """Read a features.h5 file back into numpy / scipy objects (used to verify the .h5 copy)."""
    import h5py

    out = {}
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            node = h5[key]
            if isinstance(node, h5py.Group):
                out[key] = sparse.csr_matrix(
                    (node["data"][:], node["indices"][:], node["indptr"][:]), shape=tuple(node.attrs["shape"])
                )
            else:
                out[key] = node[:]
        out["meta"] = json.loads(h5.attrs["meta"])
    return out


def _load_cached(folder: Path, is_sparse: bool, y_train, y_test):
    needed = ["X_train_pca30.npy", "X_test_pca30.npy", "y_train.npy", "y_test.npy", "meta.json", "features.h5"]
    needed += ["X_train_sparse.npz", "X_test_sparse.npz"] if is_sparse else ["X_train.npy", "X_test.npy"]
    if not all((folder / f).exists() for f in needed):
        return None
    if not (np.array_equal(np.load(folder / "y_train.npy"), y_train)
            and np.array_equal(np.load(folder / "y_test.npy"), y_test)):
        return None
    if is_sparse:
        x_train, x_test = sparse.load_npz(folder / "X_train_sparse.npz"), sparse.load_npz(folder / "X_test_sparse.npz")
    else:
        x_train, x_test = np.load(folder / "X_train.npy"), np.load(folder / "X_test.npy")
    return {
        "X_train": x_train,
        "X_test": x_test,
        "X_train_pca30": np.load(folder / "X_train_pca30.npy"),
        "X_test_pca30": np.load(folder / "X_test_pca30.npy"),
        "meta": json.loads((folder / "meta.json").read_text()),
    }


# ---------------------------------------------------------------------------
# Sparse lexical features
# ---------------------------------------------------------------------------
def build_vectorizer(embedding_type: str, tokenization: bool):
    common = dict(max_features=MAX_FEATURES, min_df=2, lowercase=True, dtype=np.float32)
    if tokenization:
        common.update(ngram_range=(1, 2))
    else:
        common.update(ngram_range=(1, 1), tokenizer=str.split, token_pattern=None)
    if embedding_type == "bow":
        return CountVectorizer(**common)
    return TfidfVectorizer(sublinear_tf=True, **common)


def _sparse_features(data: dict, cfg: ExperimentConfig):
    vec = build_vectorizer(cfg.embedding_type, cfg.preprocess.tokenization)
    x_train = vec.fit_transform(data["train_text"]).astype(np.float32)
    x_test = vec.transform(data["test_text"]).astype(np.float32)
    svd = TruncatedSVD(n_components=PCA_DIM, random_state=SEED)
    z_train = svd.fit_transform(x_train).astype(np.float32)
    z_test = svd.transform(x_test).astype(np.float32)
    meta = {
        "vocabulary_size": len(vec.vocabulary_),
        "ngram_range": list(vec.ngram_range),
        "projection": f"TruncatedSVD({PCA_DIM})",
        "explained_variance_30d": round(float(svd.explained_variance_ratio_.sum()), 4),
    }
    return x_train, x_test, z_train, z_test, meta


# ---------------------------------------------------------------------------
# Dense Transformer features
# ---------------------------------------------------------------------------
def _word_level_batch(texts, tokenizer, max_length):
    """Whole-word vocabulary lookup without subword decomposition (tokenization=False)."""
    import torch

    vocab = tokenizer.get_vocab()
    unk, cls, sep, pad = (tokenizer.unk_token_id, tokenizer.cls_token_id,
                          tokenizer.sep_token_id, tokenizer.pad_token_id)
    rows = [[cls] + [vocab.get(w, unk) for w in t.split()][: max_length - 2] + [sep] for t in texts]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def token_lengths(texts, tokenizer, tokenization: bool) -> np.ndarray:
    if tokenization:
        enc = tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
        return np.array([len(e) for e in enc])
    return np.array([len(t.split()) + 2 for t in texts])


def resolve_max_length(train_text, tokenizer, prep, cap: int) -> int:
    if not prep.padding:
        return UNPADDED_MAX_LEN
    p95 = int(np.ceil(np.percentile(token_lengths(train_text, tokenizer, prep.tokenization), 95)))
    return int(min(cap, max(24, p95)))


def _encode(texts, tokenizer, model, device, prep, max_length, pooling, normalize):
    import torch

    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), BERT_BATCH_SIZE):
            idx = order[start : start + BERT_BATCH_SIZE]
            batch = [texts[i] for i in idx]
            if prep.tokenization:
                enc = tokenizer(batch, padding="longest", truncation=True, max_length=max_length,
                                return_tensors="pt")
                enc = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            else:
                enc = _word_level_batch(batch, tokenizer, max_length)
            mask = enc["attention_mask"].to(device)
            ids = enc["input_ids"].to(device)
            if prep.padding:
                hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            else:
                hidden = model(input_ids=ids, attention_mask=torch.ones_like(mask)).last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            elif prep.padding:
                m = mask.unsqueeze(-1).type_as(hidden)
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(1)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[idx] = pooled.float().cpu().numpy()
    return out


def _dense_features(data: dict, cfg: ExperimentConfig):
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    spec = DENSE_MODELS[cfg.embedding_type]
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    model = AutoModel.from_pretrained(spec["model_id"]).to(device)
    max_length = resolve_max_length(data["train_text"], tokenizer, cfg.preprocess, data["max_len_cap"])
    print(f"[features] {spec['model_id']} on {device}, max_length={max_length}, "
          f"pooling={spec['pooling']}, tokenization={cfg.preprocess.tokenization}, "
          f"padding/masking={cfg.preprocess.padding}")
    args = (tokenizer, model, device, cfg.preprocess, max_length, spec["pooling"], spec["normalize"])
    x_train = _encode(data["train_text"], *args)
    x_test = _encode(data["test_text"], *args)
    scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(scaler.transform(x_train))
    z_train = pca.transform(scaler.transform(x_train)).astype(np.float32)
    z_test = pca.transform(scaler.transform(x_test)).astype(np.float32)
    unk_rate = None
    if not cfg.preprocess.tokenization:
        vocab = tokenizer.get_vocab()
        words = [w for t in data["train_text"] for w in t.split()]
        unk_rate = round(float(np.mean([w not in vocab for w in words])), 4)
    meta = {
        "model_id": spec["model_id"],
        "pooling": spec["pooling"],
        "l2_normalized": spec["normalize"],
        "max_length": max_length,
        "unk_word_rate": unk_rate,
        "projection": f"StandardScaler + PCA({PCA_DIM})",
        "explained_variance_30d": round(float(pca.explained_variance_ratio_.sum()), 4),
    }
    del model
    if device.type == "mps":
        import torch

        torch.mps.empty_cache()
    return x_train, x_test, z_train, z_test, meta


# ---------------------------------------------------------------------------
# Word2Vec (static word vectors + mean-pooled document features)
# ---------------------------------------------------------------------------
def _mean_pool_word2vec(texts, kv) -> np.ndarray:
    dim = kv.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs = [kv[w] for w in str(text).split() if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def train_or_load_word2vec(train_texts, folder: Path, seed: int = SEED):
    """Train skip-gram Word2Vec on train texts, or load WORD2VEC_KV_PATH / cached kv."""
    from modules.word2vec_utils import load_or_train_kv

    return load_or_train_kv(train_texts, folder, seed)


def _word2vec_features(data: dict, cfg: ExperimentConfig, folder: Path):
    kv = train_or_load_word2vec(data["train_text"], folder, SEED)
    x_train = _mean_pool_word2vec(data["train_text"], kv)
    x_test = _mean_pool_word2vec(data["test_text"], kv)
    scaler = StandardScaler().fit(x_train)
    z_train_full = np.nan_to_num(scaler.transform(x_train), nan=0.0, posinf=0.0, neginf=0.0)
    z_test_full = np.nan_to_num(scaler.transform(x_test), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(z_train_full)
    z_train = pca.transform(z_train_full).astype(np.float32)
    z_test = pca.transform(z_test_full).astype(np.float32)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    covered = sum(1 for t in data["train_text"] for w in t.split() if w in kv)
    total = sum(len(t.split()) for t in data["train_text"]) or 1
    meta = {
        "model_id": "word2vec-skipgram-train",
        "pooling": "mean",
        "vector_size": int(kv.vector_size),
        "vocab_size": int(len(kv)),
        "token_coverage": round(covered / total, 4),
        "kv_path": str(folder / "word2vec.kv.npz"),
        "projection": f"StandardScaler + PCA({PCA_DIM})",
        "explained_variance_30d": round(float(pca.explained_variance_ratio_.sum()), 4),
    }
    return x_train, x_test, z_train, z_test, meta


def load_word2vec_kv(dataset_name: str, prep_tag: str, feature_root: Path = FEATURE_DIR):
    """Load the cached KeyedVectors for a dataset/preprocess tag (trains if missing)."""
    from modules.word2vec_utils import SimpleKeyedVectors

    folder = cache_dir_for(dataset_name, "word2vec", prep_tag, feature_root)
    kv_path = folder / "word2vec.kv.npz"
    if kv_path.exists():
        return SimpleKeyedVectors.load(kv_path)
    raise FileNotFoundError(
        f"No Word2Vec cache at {kv_path}. Call extract_and_save_features(..., embedding_type='word2vec') first."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_and_save_features(dataset_dict: dict, exp_cfg: ExperimentConfig,
                              feature_root: Path = FEATURE_DIR) -> dict:
    """
    Build (or load from cache) one embedding for one dataset and preprocessing tag.

    Returns a dict with X_train / X_test (full representation, sparse or dense),
    X_train_pca30 / X_test_pca30, is_sparse, dim, cache_dir, extract_time_sec, meta.
    """
    emb = exp_cfg.embedding_type
    known = set(SPARSE_TYPES) | set(DENSE_MODELS) | {"word2vec"}
    if emb not in known:
        raise ValueError(f"Unknown embedding_type '{emb}'")
    tag = exp_cfg.preprocess.tag()
    if dataset_dict.get("preprocess_tag") not in (None, tag):
        raise ValueError(f"dataset was cleaned with {dataset_dict['preprocess_tag']} but config asks for {tag}")
    folder = cache_dir_for(exp_cfg.dataset_name, emb, tag, feature_root)
    is_sparse = emb in SPARSE_TYPES
    y_train, y_test = dataset_dict["train_label"], dataset_dict["test_label"]

    cached = _load_cached(folder, is_sparse, y_train, y_test)
    if cached is not None and (emb != "word2vec" or (folder / "word2vec.kv.npz").exists()):
        out = {**cached, "is_sparse": is_sparse, "cache_dir": folder,
               "extract_time_sec": cached["meta"].get("extract_time_sec", 0.0), "from_cache": True}
        print(f"[features] loaded {emb}_{tag} from {folder}: train {out['X_train'].shape}, "
              f"test {out['X_test'].shape}, 30-d {out['X_train_pca30'].shape}")
    else:
        start = time.perf_counter()
        if is_sparse:
            x_train, x_test, z_train, z_test, meta = _sparse_features(dataset_dict, exp_cfg)
        elif emb == "word2vec":
            x_train, x_test, z_train, z_test, meta = _word2vec_features(dataset_dict, exp_cfg, folder)
        else:
            x_train, x_test, z_train, z_test, meta = _dense_features(dataset_dict, exp_cfg)
        elapsed = time.perf_counter() - start
        meta.update({
            "dataset": exp_cfg.dataset_name,
            "embedding_type": emb,
            "preprocess": tag,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "dim": int(x_train.shape[1]),
            "extract_time_sec": round(elapsed, 3),
        })
        arrays = {"X_train_pca30": z_train, "X_test_pca30": z_test, "y_train": y_train, "y_test": y_test}
        sparse_arrays = {}
        if is_sparse:
            sparse_arrays = {"X_train_sparse": x_train, "X_test_sparse": x_test}
        else:
            arrays.update({"X_train": x_train, "X_test": x_test})
        _save(folder, arrays, sparse_arrays, meta, exp_cfg.save_format)
        out = {"X_train": x_train, "X_test": x_test, "X_train_pca30": z_train, "X_test_pca30": z_test,
               "meta": meta, "is_sparse": is_sparse, "cache_dir": folder,
               "extract_time_sec": elapsed, "from_cache": False}
        print(f"[features] {emb}_{tag}: train {x_train.shape}, test {x_test.shape}, "
              f"30-d {z_train.shape} in {elapsed:.1f}s -> {folder}")
    out["dim"] = int(out["X_train"].shape[1])
    out["embedding"] = emb
    return out


def list_feature_files(feature_root: Path = FEATURE_DIR):
    """Rows (dataset, folder, file, size_mb) for every saved .npy / .npz / .h5 file."""
    rows = []
    for path in sorted(Path(feature_root).rglob("*")):
        if path.suffix in (".npy", ".npz", ".h5"):
            rel = path.relative_to(feature_root)
            rows.append({
                "dataset": rel.parts[0],
                "folder": rel.parts[1] if len(rel.parts) > 2 else "",
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1e6, 3),
            })
    return rows
