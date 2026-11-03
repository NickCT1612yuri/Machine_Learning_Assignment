"""
Lightweight skip-gram Word2Vec (no gensim) for environments where gensim wheels
are unavailable. Provides a KeyedVectors-like object used by feature extraction
and the DL / SetFit adapters.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from modules.config import SEED, WORD2VEC
from modules.data_loader import quick_mode

_WEIGHT_CLIP = 1.0


class SimpleKeyedVectors:
    """Minimal gensim.KeyedVectors-compatible surface."""

    def __init__(self, vectors: np.ndarray, index_to_key: list[str]):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.index_to_key = list(index_to_key)
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        self.vector_size = int(self.vectors.shape[1])

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_index

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.key_to_index[key]]

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.vectors).all())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, words=np.array(self.index_to_key, dtype=object))
        # also write word2vec text for debugging / optional ST loaders
        txt = path.with_suffix(".txt") if path.suffix == ".npz" else Path(str(path) + ".txt")
        # Prefer word2vec.kv.txt next to word2vec.kv.npz
        if path.name.endswith(".kv.npz"):
            txt = path.with_name(path.name.replace(".kv.npz", ".kv.txt"))
        with txt.open("w", encoding="utf-8") as f:
            f.write(f"{len(self)} {self.vector_size}\n")
            for w, i in self.key_to_index.items():
                vals = " ".join(f"{x:.6f}" for x in self.vectors[i])
                f.write(f"{w} {vals}\n")

    def save_word2vec_format(self, path: str | Path) -> None:
        self.save(Path(path).with_suffix(".kv.npz") if not str(path).endswith(".npz") else path)

    @classmethod
    def load(cls, path: str | Path, mmap: str | None = None) -> "SimpleKeyedVectors":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        words = [str(w) for w in data["words"].tolist()]
        return cls(data["vectors"], words)


def _build_vocab(sentences: list[list[str]], min_count: int) -> list[str]:
    counts = Counter(w for s in sentences for w in s)
    return [w for w, c in counts.most_common() if c >= min_count]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _sg_update(w_in: np.ndarray, w_out: np.ndarray, center: int, targets: np.ndarray,
               labels: np.ndarray, lr: float) -> None:
    """Vectorized skip-gram NEG update for one center vs many target ids."""
    if targets.size == 0:
        return
    # Copy so in/out updates do not alias.
    v_c = w_in[center].copy()
    w_t = w_out[targets]
    scores = w_t @ v_c
    pred = _sigmoid(scores)
    g = (lr * (labels.astype(np.float32) - pred)).astype(np.float32)[:, None]
    w_in[center] += (g * w_t).sum(axis=0)
    w_out[targets] += g * v_c[None, :]
    np.clip(w_in[center], -_WEIGHT_CLIP, _WEIGHT_CLIP, out=w_in[center])
    np.clip(w_out[targets], -_WEIGHT_CLIP, _WEIGHT_CLIP, out=w_out[targets])


def train_skipgram(train_texts, seed: int = SEED) -> SimpleKeyedVectors:
    """Train a small skip-gram model with negative sampling (vectorized numpy)."""
    rng = np.random.default_rng(seed)
    sentences = [str(t).split() for t in train_texts]
    if quick_mode():
        sentences = sentences[:800]
    dim = 100 if quick_mode() else WORD2VEC["vector_size"]
    window = WORD2VEC["window"]
    min_count = WORD2VEC["min_count"]
    epochs = 1 if quick_mode() else WORD2VEC["epochs"]
    neg = 5
    vocab = _build_vocab(sentences, min_count)
    if not vocab:
        vocab = ["unk"]
    word2id = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    w_in = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    w_out = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    lr = 0.025

    freqs = np.ones(n, dtype=np.float64)
    counts = Counter(w for s in sentences for w in s if w in word2id)
    for w, c in counts.items():
        freqs[word2id[w]] = c
    noise = freqs ** 0.75
    noise /= noise.sum()

    print(f"[features] Word2Vec skip-gram: vocab={n}, dim={dim}, epochs={epochs}, docs={len(sentences)}")
    for epoch in range(epochs):
        order = rng.permutation(len(sentences))
        for si in order:
            ids = [word2id[w] for w in sentences[si] if w in word2id]
            if len(ids) < 2:
                continue
            for i, center in enumerate(ids):
                left = max(0, i - window)
                right = min(len(ids), i + window + 1)
                ctx = np.array([ids[j] for j in range(left, right) if j != i], dtype=np.int64)
                if ctx.size == 0:
                    continue
                # positives
                _sg_update(w_in, w_out, center, ctx, np.ones(ctx.size, dtype=np.float32), lr)
                # negatives (shared draw per center position)
                neg_ids = rng.choice(n, size=neg * ctx.size, p=noise).astype(np.int64)
                # avoid accidental identity with center
                neg_ids[neg_ids == center] = (center + 1) % n
                _sg_update(w_in, w_out, center, neg_ids, np.zeros(neg_ids.size, dtype=np.float32), lr)
        lr = max(lr * 0.95, 0.001)
        np.clip(w_in, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=w_in)
        np.clip(w_out, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=w_out)
        print(f"[features] Word2Vec epoch {epoch + 1}/{epochs} done")

    np.nan_to_num(w_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(w_in, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=w_in)
    return SimpleKeyedVectors(w_in, vocab)


def load_or_train_kv(train_texts, folder: Path, seed: int = SEED) -> SimpleKeyedVectors:
    import os

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    kv_path = folder / "word2vec.kv.npz"
    pretrained = os.environ.get(WORD2VEC["pretrained_env"], "").strip()
    if pretrained and Path(pretrained).exists():
        print(f"[features] loading Word2Vec from {pretrained}")
        kv = SimpleKeyedVectors.load(pretrained)
        if not kv.is_finite():
            raise ValueError(f"Pretrained Word2Vec at {pretrained} contains NaN/Inf")
        kv.save(kv_path)
        return kv
    if kv_path.exists():
        kv = SimpleKeyedVectors.load(kv_path)
        if kv.is_finite():
            return kv
        print(f"[features] cached Word2Vec at {kv_path} has NaN/Inf; deleting and retraining")
        kv_path.unlink(missing_ok=True)
        txt = folder / "word2vec.kv.txt"
        txt.unlink(missing_ok=True)
    print("[features] training lightweight skip-gram Word2Vec (no gensim)...")
    kv = train_skipgram(train_texts, seed)
    if not kv.is_finite():
        raise RuntimeError("Word2Vec training produced non-finite vectors after sanitization")
    kv.save(kv_path)
    print(f"[features] Word2Vec vocab={len(kv)}, dim={kv.vector_size}")
    return kv
