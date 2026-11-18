"""Metrics: accuracy, macro P/R/F1, calibration (Brier, ECE), bootstrap intervals, error analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import N_BOOTSTRAP, SEED


def brier(y, proba):
    """Multiclass Brier score: mean over texts of sum_k (p_k - 1[y = k])^2. Lower is better."""
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ece(y, proba, n_bins=15):
    """Expected calibration error of the top-label confidence over equal-width bins."""
    conf, correct = proba.max(1), proba.argmax(1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * np.mean(bins == b)
                     for b in np.unique(bins)))


def scores(y, pred, proba=None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
           "brier": np.nan, "ece": np.nan}
    if proba is not None:
        out.update(brier=brier(y, proba), ece=ece(y, proba))
    return out


def bootstrap_f1(y, preds: dict, n=N_BOOTSTRAP) -> dict:
    """Resample the test set; return the macro-F1 sample for every model (same resamples for all)."""
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in preds}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for k, p in preds.items():
            draws[k].append(f1_score(y[idx], p[idx], average="macro", zero_division=0))
    return {k: np.array(v) for k, v in draws.items()}


def per_class_f1(y, preds: dict, labels) -> pd.DataFrame:
    ids = np.arange(len(labels))
    table = {k: f1_score(y, p, labels=ids, average=None, zero_division=0) for k, p in preds.items()}
    return pd.DataFrame(table, index=pd.Index(labels, name="class")).assign(support=np.bincount(y, minlength=len(labels)))


def top_confusions(y, pred, labels, k=10) -> pd.DataFrame:
    cm = confusion_matrix(y, pred, labels=np.arange(len(labels)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    rows = [(labels[i], labels[j], cm[i, j]) for i, j in zip(*np.unravel_index(flat, cm.shape)) if cm[i, j]]
    return pd.DataFrame(rows, columns=["true", "predicted", "count"])
