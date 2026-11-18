"""
Downstream classifiers on saved features, with one shared metric dictionary.

classifier_type   sparse input (bow / tfidf)        dense input (embeddings)
---------------   -------------------------------   --------------------------------
naive_bayes       MultinomialNB(alpha=0.5)          GaussianNB()
logreg            LogisticRegression(C=2.0)          StandardScaler + LogisticRegression(C=2.0)
svm               Calibrated LinearSVC(C=1.0)        L2 Normalizer + Calibrated LinearSVC(C=1.0)
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from modules.config import SEED

CLASSIFIER_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logreg": "LogReg",
    "svm": "LinearSVC (calibrated)",
}


def multiclass_brier(y_true, proba) -> float:
    """Mean over samples of sum_k (p_k - 1[y = k])^2; 0 is perfect, 2 is worst."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over equal-width confidence bins."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def compute_metrics(y_true, proba, method: str, train_time: float = 0.0, infer_time: float = 0.0) -> dict:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 0.0, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True).clip(min=1e-12)
    pred = proba.argmax(axis=1)
    metrics = {
        "method": method,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier_score": multiclass_brier(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "train_time_sec": float(train_time),
        "infer_time_sec": float(infer_time),
        "predictions": pred,
        "probabilities": proba.astype(np.float32),
    }
    print(
        f"  {method:42s} acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
        f"Brier={metrics['brier_score']:.4f} ECE={metrics['ece']:.4f} "
        f"train={train_time:.1f}s infer={infer_time:.2f}s"
    )
    return metrics


def align_proba(model_classes, proba, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    out[:, np.asarray(model_classes, dtype=int)] = proba
    return out


def build_classifier(classifier_type: str, is_sparse: bool, seed: int = SEED, n_classes: int = 2):
    if classifier_type == "naive_bayes":
        return MultinomialNB(alpha=0.5) if is_sparse else GaussianNB()
    if classifier_type == "logreg":
        clf = LogisticRegression(
            C=2.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=seed
        )
        return clf if is_sparse else make_pipeline(StandardScaler(), clf)
    if classifier_type == "svm":
        svc = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)
        clf = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
        # Dense SVM: L2-normalize rows for stable, fast convergence.
        return clf if is_sparse else make_pipeline(Normalizer(), clf)
    raise ValueError(f"Unknown classifier_type '{classifier_type}'")


def train_and_evaluate_classifier(
    features_dict: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str,
    method_label: str,
    seed: int = SEED,
    use_projection: bool = False,
    n_classes: int | None = None,
) -> dict:
    """
    Fit one classifier on saved features and return the shared metric dictionary.

    Set `use_projection=True` to run on the 30-d PCA/SVD projection (optional control).
    """
    n_classes = n_classes or int(max(y_train.max(), y_test.max()) + 1)
    reduced = use_projection
    x_train = features_dict["X_train_pca30"] if reduced else features_dict["X_train"]
    x_test = features_dict["X_test_pca30"] if reduced else features_dict["X_test"]
    is_sparse = sparse.issparse(x_train)

    model = build_classifier(classifier_type, is_sparse, seed, n_classes=n_classes)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    proba = align_proba(model.classes_, model.predict_proba(x_test), n_classes)
    infer_time = time.perf_counter() - start

    metrics = compute_metrics(y_test, proba, method_label, train_time, infer_time)
    metrics["classifier"] = classifier_type
    metrics["input"] = "30-d projection" if reduced else ("sparse" if is_sparse else "dense")
    metrics["input_dim"] = int(np.asarray(x_train.shape[1]))
    return metrics
