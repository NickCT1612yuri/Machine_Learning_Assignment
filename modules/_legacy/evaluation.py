"""
Experiment orchestration, result tables and figures for Banking77.

Results are cached in-process by (dataset, preprocessing tag, method) so ablation
and main comparison always report the same number for a shared configuration.

Outputs
-------
reports/comparison_banking77.csv
reports/ablation_banking77.csv / ablation_delta_banking77.csv
reports/errors_banking77.csv / negation_banking77.csv
reports/eda_summary.csv, master_comparison.csv, master_f1_pivot.csv, master_ablation.csv
reports/figures/banking77/*.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.classifiers import train_and_evaluate_classifier
from modules.config import (
    DATASET_ORDER,
    DATASET_REGISTRY,
    DL_EMBEDDINGS,
    DL_MODELS,
    FIG_DIR,
    REPORT_DIR,
    ExperimentConfig,
    PreprocessConfig,
)
from modules.data_loader import load_and_preprocess_dataset, normalize_text
from modules.deep_learning import run_dl_pipeline
from modules.eda import run_eda
from modules.feature_extraction import extract_and_save_features

MAIN_PREP = PreprocessConfig(
    stop_word_removal=False, tokenization=True, padding=True, negation_marking=False
)

FLAG_SWITCHES = {
    "stop_word_removal": (
        MAIN_PREP,
        PreprocessConfig(True, True, True, False),
    ),
    "tokenization": (
        PreprocessConfig(False, False, True, False),
        MAIN_PREP,
    ),
    "padding": (
        PreprocessConfig(False, True, False, False),
        MAIN_PREP,
    ),
    "negation_marking": (
        MAIN_PREP,
        PreprocessConfig(False, True, True, True),
    ),
}
ABLATION_MODELS = {
    "TF-IDF + LogReg": ("tfidf", "logreg"),
    "DistilBERT + LogReg": ("distilbert", "logreg"),
    "Word2Vec + BiLSTM": ("bilstm", "word2vec"),
}

EMBEDDINGS = ["tfidf", "word2vec", "distilbert"]
EMBEDDING_NAMES = {
    "bow": "BoW",
    "tfidf": "TF-IDF",
    "word2vec": "Word2Vec",
    "distilbert": "DistilBERT",
}
CLASSICAL = [("naive_bayes", False), ("logreg", False), ("svm", False)]
MAIN_GRID = {
    "tfidf": CLASSICAL,
    "word2vec": CLASSICAL,
    "distilbert": CLASSICAL,
}
FAMILY = {
    "bow": "Traditional (sparse lexical)",
    "tfidf": "Traditional (sparse lexical)",
    "word2vec": "Static word embeddings",
    "distilbert": "Modern (frozen Transformer)",
    "bilstm": "Deep learning (end-to-end)",
}
FAMILY_COLORS = {
    "Traditional (sparse lexical)": "#3d5a80",
    "Static word embeddings": "#8ab17d",
    "Modern (frozen Transformer)": "#ee6c4d",
    "Deep learning (end-to-end)": "#2a9d8f",
}
CLASSIFIER_COLORS = {
    "naive_bayes": "#e76f51",
    "logreg": "#264653",
    "svm": "#e9c46a",
    "bilstm": "#2a9d8f",
}
EMBEDDING_MARKERS = {
    "tfidf": "D",
    "word2vec": "v",
    "distilbert": "^",
    "bilstm": "P",
}
CSV_COLUMNS = [
    "dataset", "method", "family", "embedding", "classifier", "input", "input_dim", "preprocess",
    "accuracy", "precision", "recall", "f1", "brier_score", "ece",
    "extract_time_sec", "train_time_sec", "infer_time_sec", "notes",
]
NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|cannot|nor|without)\b|n't\b")

_DATA: dict = {}
_FEATURES: dict = {}
_RESULTS: dict = {}


def get_dataset(name: str, prep: PreprocessConfig = MAIN_PREP) -> dict:
    key = (name, prep.tag())
    if key not in _DATA:
        _DATA[key] = load_and_preprocess_dataset(name, prep)
    return _DATA[key]


def get_features(name: str, embedding: str, prep: PreprocessConfig = MAIN_PREP) -> dict:
    key = (name, embedding, prep.tag())
    if key not in _FEATURES:
        _FEATURES[key] = extract_and_save_features(
            get_dataset(name, prep), ExperimentConfig(name, prep, embedding)
        )
    return _FEATURES[key]


def method_label(embedding: str, classifier: str, projection: bool) -> str:
    sparse_input = embedding in ("bow", "tfidf")
    emb = EMBEDDING_NAMES[embedding]
    if projection:
        emb += " (SVD-30)" if sparse_input else " (PCA-30)"
    clf = {
        "naive_bayes": "MultinomialNB" if sparse_input and not projection else "GaussianNB",
        "logreg": "LogReg",
        "svm": "LinearSVC",
    }[classifier]
    return f"{emb} + {clf}"


def _result_key(name, prep, method_key):
    return (name, prep.tag(), method_key)


def evaluate(
    name: str,
    embedding: str,
    classifier: str,
    projection: bool = False,
    prep: PreprocessConfig = MAIN_PREP,
) -> dict:
    key = _result_key(name, prep, f"{embedding}|{classifier}|{int(projection)}")
    if key not in _RESULTS:
        data = get_dataset(name, prep)
        feats = get_features(name, embedding, prep)
        m = train_and_evaluate_classifier(
            feats,
            data["train_label"],
            data["test_label"],
            classifier,
            method_label(embedding, classifier, projection),
            use_projection=projection,
            n_classes=data["n_classes"],
        )
        m.update({
            "embedding": EMBEDDING_NAMES[embedding],
            "embedding_key": embedding,
            "family": FAMILY[embedding],
            "preprocess": prep.tag(),
            "extract_time_sec": float(feats["meta"].get("extract_time_sec", 0.0)),
            "notes": "",
        })
        _RESULTS[key] = m
    return _RESULTS[key]


def run_dl_cell(
    name: str,
    dl_model: str,
    dl_embedding: str,
    prep: PreprocessConfig = MAIN_PREP,
    show: bool = False,
) -> dict | None:
    method_key = f"dl|{dl_model}|{dl_embedding}"
    key = _result_key(name, prep, method_key)
    if key not in _RESULTS:
        data = get_dataset(name, prep)
        plot = None
        if dl_model == "bilstm" and dl_embedding == "word2vec" and prep.tag() == MAIN_PREP.tag():
            plot = FIG_DIR / name / "bilstm_loss.png"
        m = run_dl_pipeline(data, dl_model, dl_embedding, prep, loss_plot_path=plot, show=show)
        if m is None:
            return None
        emb_name = EMBEDDING_NAMES.get(dl_embedding, dl_embedding)
        m.update({
            "embedding": emb_name,
            "embedding_key": dl_model,
            "family": FAMILY[dl_model],
            "preprocess": prep.tag(),
            "extract_time_sec": 0.0,
            "notes": (
                f"{dl_model} x {dl_embedding}"
                + (f", {m['epochs_run']} epochs" if "epochs_run" in m else "")
            ),
        })
        _RESULTS[key] = m
    return _RESULTS[key]


def _row(name: str, m: dict) -> dict:
    row = {col: m.get(col) for col in CSV_COLUMNS}
    row["dataset"] = DATASET_REGISTRY[name]["title"]
    row["notes"] = m.get("notes", "")
    return row


def _main_keys():
    for emb in EMBEDDINGS:
        for clf, proj in MAIN_GRID[emb]:
            yield emb, clf, proj, f"{emb}|{clf}|{int(proj)}"
    for dl_emb in DL_EMBEDDINGS:
        for dl_model in DL_MODELS:
            yield dl_emb, dl_model, False, f"dl|{dl_model}|{dl_emb}"


def collected_results(name: str) -> list[dict]:
    out = []
    for *_, method_key in _main_keys():
        m = _RESULTS.get(_result_key(name, MAIN_PREP, method_key))
        if m is not None:
            out.append(m)
    return out


def run_all_eda(datasets=DATASET_ORDER, show: bool = False) -> pd.DataFrame:
    frames = [run_eda(get_dataset(name, MAIN_PREP), name, show=show) for name in datasets]
    table = pd.concat(frames, ignore_index=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "eda_summary.csv", index=False)
    return table


def run_feature_extraction(name: str, embeddings=EMBEDDINGS) -> pd.DataFrame:
    rows = []
    for emb in embeddings:
        f = get_features(name, emb, MAIN_PREP)
        rows.append({
            "dataset": DATASET_REGISTRY[name]["title"],
            "embedding": EMBEDDING_NAMES[emb],
            "full_dim": f["dim"],
            "sparse": f["is_sparse"],
            "projection": f["meta"]["projection"],
            "variance_kept_30d": f["meta"]["explained_variance_30d"],
            "max_length": f["meta"].get("max_length"),
            "extract_time_sec": round(float(f["meta"]["extract_time_sec"]), 2),
            "folder": str(Path(f["cache_dir"]).relative_to(REPORT_DIR.parent)),
        })
    return pd.DataFrame(rows)


def run_main_comparison(name: str, embeddings=EMBEDDINGS) -> pd.DataFrame:
    print(f"\n== {DATASET_REGISTRY[name]['title']}: embedding x classifier matrix ({MAIN_PREP.tag()}) ==")
    for emb in embeddings:
        for clf, proj in MAIN_GRID[emb]:
            evaluate(name, emb, clf, proj)
    return save_comparison(name)


def run_deep_learning(name: str, run_bilstm_model: bool = True, show: bool = False) -> pd.DataFrame:
    print(f"\n== {DATASET_REGISTRY[name]['title']}: bonus deep learning (Word2Vec + BiLSTM) ==")
    rows = []
    if run_bilstm_model:
        for dl_emb in DL_EMBEDDINGS:
            for dl_model in DL_MODELS:
                m = run_dl_cell(name, dl_model, dl_emb, MAIN_PREP, show=show)
                if m is not None:
                    rows.append(_row(name, m))
    save_comparison(name)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def save_comparison(name: str) -> pd.DataFrame:
    table = pd.DataFrame([_row(name, m) for m in collected_results(name)], columns=CSV_COLUMNS)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(REPORT_DIR / f"comparison_{name}.csv", index=False)
    return table


def run_preprocessing_ablation(name: str, include_bilstm: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-flag-at-a-time ablation around MAIN_PREP for the three ablation models."""
    print(f"\n== {DATASET_REGISTRY[name]['title']}: boolean preprocessing ablation ==")
    runs, deltas = {}, []
    for model_name, (emb, clf) in ABLATION_MODELS.items():
        if emb == "bilstm" and not include_bilstm:
            continue
        for flag, (cfg_off, cfg_on) in FLAG_SWITCHES.items():
            if flag == "padding" and emb in ("bow", "tfidf"):
                deltas.append({
                    "dataset": DATASET_REGISTRY[name]["title"],
                    "model": model_name,
                    "flag": flag,
                    "config_false": cfg_off.tag(),
                    "config_true": cfg_on.tag(),
                    "f1_false": np.nan,
                    "f1_true": np.nan,
                    "delta_f1": np.nan,
                    "acc_false": np.nan,
                    "acc_true": np.nan,
                    "note": "not applicable: bag-of-words has no sequence length",
                })
                continue
            results = []
            for cfg in (cfg_off, cfg_on):
                if emb == "bilstm":
                    m = run_dl_cell(name, "bilstm", clf or "word2vec", cfg)
                else:
                    m = evaluate(name, emb, clf, False, cfg)
                if m is None:
                    continue
                runs[(model_name, cfg.tag())] = {
                    "dataset": DATASET_REGISTRY[name]["title"],
                    "model": model_name,
                    "preprocess": cfg.tag(),
                    "stop_word_removal": cfg.stop_word_removal,
                    "tokenization": cfg.tokenization,
                    "padding": cfg.padding,
                    "negation_marking": cfg.negation_marking,
                    "accuracy": m["accuracy"],
                    "f1": m["f1"],
                    "brier_score": m["brier_score"],
                    "notes": m.get("notes", ""),
                }
                results.append(m)
            if len(results) == 2:
                deltas.append({
                    "dataset": DATASET_REGISTRY[name]["title"],
                    "model": model_name,
                    "flag": flag,
                    "config_false": cfg_off.tag(),
                    "config_true": cfg_on.tag(),
                    "f1_false": results[0]["f1"],
                    "f1_true": results[1]["f1"],
                    "delta_f1": results[1]["f1"] - results[0]["f1"],
                    "acc_false": results[0]["accuracy"],
                    "acc_true": results[1]["accuracy"],
                    "note": "",
                })
    runs_df = pd.DataFrame(list(runs.values()))
    delta_df = pd.DataFrame(deltas)
    runs_df.round(6).to_csv(REPORT_DIR / f"ablation_{name}.csv", index=False)
    delta_df.round(6).to_csv(REPORT_DIR / f"ablation_delta_{name}.csv", index=False)
    return runs_df, delta_df


def best_result(name: str) -> dict:
    return max(collected_results(name), key=lambda m: m["f1"])


def error_analysis(name: str, top: int = 10) -> pd.DataFrame:
    data = get_dataset(name, MAIN_PREP)
    best = best_result(name)
    y, pred, names = data["test_label"], best["predictions"], data["label_names"]
    wrong = np.where(y != pred)[0]
    pairs = pd.Series(list(zip(y[wrong], pred[wrong]))).value_counts().head(top)
    rows = []
    for (t, p), count in pairs.items():
        examples = [data["test_raw"][i] for i in wrong if y[i] == t and pred[i] == p][:2]
        rows.append({
            "dataset": data["title"],
            "model": best["method"],
            "true": names[t],
            "predicted": names[p],
            "count": int(count),
            "share_of_true_class": round(count / max((y == t).sum(), 1), 3),
            "example_1": normalize_text(examples[0])[:200] if examples else "",
            "example_2": normalize_text(examples[1])[:200] if len(examples) > 1 else "",
        })
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / f"errors_{name}.csv", index=False)
    return table


def negation_analysis(name: str) -> pd.DataFrame:
    data = get_dataset(name, MAIN_PREP)
    has_neg = np.array([bool(NEGATION.search(normalize_text(t))) for t in data["test_raw"]])
    rows = []
    for m in collected_results(name):
        correct = m["predictions"] == data["test_label"]
        rows.append({
            "dataset": data["title"],
            "method": m["method"],
            "n_negation": int(has_neg.sum()),
            "n_other": int((~has_neg).sum()),
            "acc_negation": float(correct[has_neg].mean()) if has_neg.any() else np.nan,
            "acc_other": float(correct[~has_neg].mean()) if (~has_neg).any() else np.nan,
        })
    table = pd.DataFrame(rows)
    table["gap"] = table["acc_negation"] - table["acc_other"]
    table.round(6).to_csv(REPORT_DIR / f"negation_{name}.csv", index=False)
    return table


def _finish(fig, path: Path, show: bool):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def plot_accuracy_f1(name: str, results_df: pd.DataFrame, show: bool = False):
    df = results_df.sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(df) + 1.4))
    ypos = np.arange(len(df))
    colors = [FAMILY_COLORS.get(f, "#888888") for f in df["family"]]
    ax.barh(ypos + 0.2, df["accuracy"] * 100, height=0.4, color=colors, alpha=0.45, label="Accuracy")
    ax.barh(ypos - 0.2, df["f1"] * 100, height=0.4, color=colors, label="Macro-F1")
    for y, f1 in zip(ypos, df["f1"]):
        ax.text(f1 * 100 + 0.3, y - 0.2, f"{f1 * 100:.1f}", va="center", fontsize=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["method"], fontsize=8)
    lo = max(0, np.floor(min(df["f1"].min(), df["accuracy"].min()) * 100 / 5) * 5 - 5)
    ax.set_xlim(lo, 100)
    ax.set_xlabel("score (%)  [solid = Macro-F1, light = Accuracy]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), loc="lower right", fontsize=8)
    ax.set_title(f"{DATASET_REGISTRY[name]['title']}: test Accuracy and Macro-F1 ({MAIN_PREP.tag()})")
    _finish(fig, FIG_DIR / name / "accuracy_f1_bars.png", show)


def plot_calibration_scatter(name: str, show: bool = False):
    ms = collected_results(name)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in ms:
        ax.scatter(
            m["brier_score"],
            m["f1"] * 100,
            s=90,
            color=CLASSIFIER_COLORS.get(m["classifier"], "#888"),
            marker=EMBEDDING_MARKERS.get(m["embedding_key"], "o"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            m["method"],
            (m["brier_score"], m["f1"] * 100),
            fontsize=6,
            xytext=(4, 2),
            textcoords="offset points",
        )
    clf_handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in CLASSIFIER_COLORS.items()
    ]
    emb_handles = [
        plt.Line2D([], [], marker=mk, ls="", color="grey", label=k)
        for k, mk in EMBEDDING_MARKERS.items()
    ]
    first = ax.legend(handles=clf_handles, title="classifier", fontsize=7, title_fontsize=8, loc="lower left")
    ax.add_artist(first)
    ax.legend(handles=emb_handles, title="representation", fontsize=7, title_fontsize=8, loc="upper right")
    ax.set_xlabel("multiclass Brier score (lower = better calibrated)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title(f"{DATASET_REGISTRY[name]['title']}: accuracy vs probability calibration")
    _finish(fig, FIG_DIR / name / "brier_calibration_vs_f1.png", show)


def plot_reliability(name: str, show: bool = False, n_bins: int = 10):
    data = get_dataset(name, MAIN_PREP)
    wanted = [
        "tfidf|naive_bayes|0",
        "tfidf|logreg|0",
        "tfidf|svm|0",
        "distilbert|logreg|0",
        "distilbert|svm|0",
        "dl|bilstm|word2vec",
    ]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    edges = np.linspace(0, 1, n_bins + 1)
    for key in wanted:
        m = _RESULTS.get(_result_key(name, MAIN_PREP, key))
        if m is None:
            continue
        conf = m["probabilities"].max(1)
        correct = m["predictions"] == data["test_label"]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (conf > lo) & (conf <= hi)
            if sel.sum() >= 5:
                xs.append(conf[sel].mean())
                ys.append(correct[sel].mean())
        ax.plot(
            xs,
            ys,
            "o-",
            color=CLASSIFIER_COLORS.get(m["classifier"], "#888"),
            ms=4,
            label=f"{m['method']} (ECE {m['ece']:.3f})",
        )
    ax.set_xlabel("predicted confidence (top class)")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"{DATASET_REGISTRY[name]['title']}: reliability diagram")
    _finish(fig, FIG_DIR / name / "reliability_diagram.png", show)


def plot_ablation_heatmap(name: str, delta_df: pd.DataFrame, show: bool = False):
    pivot = delta_df.pivot(index="model", columns="flag", values="delta_f1")
    pivot = pivot.reindex(
        index=[m for m in ABLATION_MODELS if m in pivot.index],
        columns=list(FLAG_SWITCHES),
    ) * 100
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.values))))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    image = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c}\n(True - False)" for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(
                j,
                i,
                "n/a" if np.isnan(v) else f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if np.isnan(v) or abs(v) < 0.6 * vmax else "white",
            )
    fig.colorbar(image, ax=ax, label="delta Macro-F1 (pp)")
    ax.set_title(f"{DATASET_REGISTRY[name]['title']}: effect of switching each flag on")
    _finish(fig, FIG_DIR / name / "preprocessing_ablation_heatmap.png", show)


def plot_confusion_best(name: str, show: bool = False):
    from sklearn.metrics import confusion_matrix

    data = get_dataset(name, MAIN_PREP)
    best = best_result(name)
    names = data["label_names"]
    mat = confusion_matrix(data["test_label"], best["predictions"], labels=np.arange(len(names)))
    shown = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    many = len(names) > 10
    fig, ax = plt.subplots(figsize=(9, 8) if many else (5.5, 4.8))
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
    if many:
        ax.set_xlabel(f"predicted intent (0-{len(names) - 1})")
        ax.set_ylabel(f"true intent (0-{len(names) - 1})")
    else:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    f"{shown[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if shown[i, j] > 0.5 else "black",
                )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.colorbar(image, ax=ax, fraction=0.046, label="row-normalized rate")
    ax.set_title(f"{data['title']}: {best['method']} (Macro-F1 {best['f1'] * 100:.2f}%)")
    _finish(fig, FIG_DIR / name / "confusion_best.png", show)


def plot_suite_results(name: str, results_df: pd.DataFrame, delta_df: pd.DataFrame | None, show: bool = False):
    plot_accuracy_f1(name, results_df, show)
    plot_calibration_scatter(name, show)
    plot_reliability(name, show)
    if delta_df is not None and len(delta_df):
        plot_ablation_heatmap(name, delta_df, show)
    plot_confusion_best(name, show)


def run_dataset_benchmark_suite(
    name: str,
    run_ablation: bool = True,
    run_bilstm_model: bool = True,
    show: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Ablation + main matrix + Word2Vec BiLSTM + figures + error analysis."""
    delta_df = None
    if run_ablation:
        _, delta_df = run_preprocessing_ablation(name, include_bilstm=run_bilstm_model)
    run_main_comparison(name)
    run_deep_learning(name, run_bilstm_model=run_bilstm_model, show=show)
    results_df = save_comparison(name)
    plot_suite_results(name, results_df, delta_df, show)
    error_analysis(name)
    negation_analysis(name)
    return results_df, delta_df


def build_master_table(datasets=DATASET_ORDER, show: bool = False) -> pd.DataFrame:
    frames = [
        pd.read_csv(REPORT_DIR / f"comparison_{n}.csv")
        for n in datasets
        if (REPORT_DIR / f"comparison_{n}.csv").exists()
    ]
    master = pd.concat(frames, ignore_index=True)
    master.to_csv(REPORT_DIR / "master_comparison.csv", index=False)
    titles = [DATASET_REGISTRY[n]["title"] for n in datasets]
    pivot = master.pivot_table(index="method", columns="dataset", values="f1", sort=False)
    pivot = pivot[[t for t in titles if t in pivot.columns]]
    pivot.round(6).to_csv(REPORT_DIR / "master_f1_pivot.csv")

    ablations = [
        pd.read_csv(REPORT_DIR / f"ablation_delta_{n}.csv")
        for n in datasets
        if (REPORT_DIR / f"ablation_delta_{n}.csv").exists()
    ]
    if ablations:
        pd.concat(ablations, ignore_index=True).to_csv(REPORT_DIR / "master_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(pivot) + 1.5))
    image = ax.imshow(pivot.values * 100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j] * 100
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < np.nanmean(pivot.values) * 100 else "black",
                )
    fig.colorbar(image, ax=ax, label="Macro-F1 (%)")
    ax.set_title(f"Macro-F1 (default preprocessing {MAIN_PREP.tag()})")
    _finish(fig, FIG_DIR / "master_f1_heatmap.png", show)
    return master
