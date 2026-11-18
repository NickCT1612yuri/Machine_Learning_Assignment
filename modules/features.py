"""Feature extraction and persistence: TF-IDF (.h5) and frozen BGE-small embeddings (.npy)."""
import warnings

import h5py
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BGE_MODEL, TFIDF_PARAMS
from .data import SPLITS, tokenize


def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- TF-IDF
def make_vectorizer(tokenizer: str, stop_words: bool, max_ngram: int) -> TfidfVectorizer:
    """tokenizer: 'regex' (word pattern, punctuation dropped) or 'whitespace' (str.split)."""
    return TfidfVectorizer(
        lowercase=True,
        tokenizer=tokenize if tokenizer == "regex" else str.split,
        token_pattern=None,
        stop_words="english" if stop_words else None,
        ngram_range=(1, max_ngram),
        dtype=np.float32,
        **TFIDF_PARAMS,
    )


def fit_tfidf(vectorizer, text: dict) -> dict:
    """Fit the vocabulary and IDF on the fit split only, then transform every split."""
    with warnings.catch_warnings():  # stop-word list vs custom tokenizer notice
        warnings.simplefilter("ignore", UserWarning)
        fit = vectorizer.fit_transform(text["fit"])
    return {"fit": fit, **{s: vectorizer.transform(text[s]) for s in ("val", "test")}}


def save_sparse(path, X: dict, y: dict):
    with h5py.File(path, "w") as f:
        for s in SPLITS:
            m = X[s].tocsr()
            g = f.create_group(s)
            g.attrs["shape"] = m.shape
            for key in ("data", "indices", "indptr"):
                g.create_dataset(key, data=getattr(m, key), compression="gzip")
            g.create_dataset("y", data=y[s])


def load_sparse(path) -> dict:
    with h5py.File(path, "r") as f:
        return {s: sparse.csr_matrix((f[s]["data"][:], f[s]["indices"][:], f[s]["indptr"][:]),
                                     shape=tuple(f[s].attrs["shape"])) for s in SPLITS}


# ---------------------------------------------------------------- BGE-small
def encode_bge(text: dict, max_len: int, padding: str = "dynamic", batch_size: int = 128) -> dict:
    """Frozen encoder: [CLS] vector of the last layer, L2-normalised (BGE's recommended pooling).

    padding='dynamic' pads each batch to its longest text; 'fixed' pads to max_len.
    The attention mask hides padding either way, so both give the same embeddings.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(dev).eval()
    out = {}
    with torch.inference_mode():
        for s in SPLITS:
            chunks = []
            for i in range(0, len(text[s]), batch_size):
                batch = tok(text[s][i:i + batch_size], truncation=True, max_length=max_len, return_tensors="pt",
                            padding="longest" if padding == "dynamic" else "max_length").to(dev)
                cls = model(**batch).last_hidden_state[:, 0]
                chunks.append(torch.nn.functional.normalize(cls, dim=1).cpu().numpy())
            out[s] = np.concatenate(chunks).astype(np.float32)
    return out


def save_dense(feature_dir, X: dict):
    for s in SPLITS:
        np.save(feature_dir / f"bge_{s}.npy", X[s])


def load_dense(feature_dir) -> dict:
    return {s: np.load(feature_dir / f"bge_{s}.npy") for s in SPLITS}
