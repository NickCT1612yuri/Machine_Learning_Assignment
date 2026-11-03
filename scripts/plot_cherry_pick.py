#!/usr/bin/env python3
"""Build cherry-picked CSVs and figures from master report tables (no retraining)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# Locked headline methods (CSV method names -> display labels for figures).
CHERRY_METHODS = [
    ("TF-IDF + MultinomialNB", "TF-IDF + MultinomialNB"),
    ("TF-IDF + LinearSVC", "TF-IDF + LinearSVC"),
    ("Word2Vec + LinearSVC", "Word2Vec + LinearSVC"),
    ("DistilBERT + LinearSVC", "DistilBERT + LinearSVC"),
    ("BGE-small + LinearSVC", "BGE-small + LinearSVC"),
    ("BGE-small + XGBoost", "BGE-small + XGBoost"),
    ("BiLSTM (word2vec)", "Word2Vec + BiLSTM"),
    ("TextCNN (word2vec)", "Word2Vec + TextCNN"),
    ("TextCNN (bge_small)", "TextCNN (BGE-small)"),
]

DATASET_ORDER = ["AG News", "Rotten Tomatoes", "Banking77"]
DATASET_SLUG = {
    "AG News": "ag_news",
    "Rotten Tomatoes": "rotten_tomatoes",
    "Banking77": "banking77",
}

ABLATION_MODELS = ["TF-IDF + LogReg", "BGE-small + LogReg", "Word2Vec + BiLSTM"]
ABLATION_FLAGS = ["stop_word_removal", "tokenization", "padding", "negation_marking"]

FAMILY_COLORS = {
    "Traditional (sparse lexical)": "#3d5a80",
    "Static word embeddings": "#8ab17d",
    "Modern (frozen Transformer)": "#ee6c4d",
    "Deep learning (end-to-end)": "#2a9d8f",
}


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _is_setfit_bge(method: str) -> bool:
    return method.startswith("SetFit") and "bge_small" in method


def select_cherry(master: pd.DataFrame) -> pd.DataFrame:
    """Keep locked classical/DL rows plus native BGE SetFit (16- or 8-shot)."""
    wanted = {src for src, _ in CHERRY_METHODS}
    rows = []
    for dataset in DATASET_ORDER:
        sub = master[master["dataset"] == dataset]
        for src, display in CHERRY_METHODS:
            hit = sub[sub["method"] == src]
            if len(hit) == 0:
                continue
            row = hit.iloc[0].copy()
            row["display_method"] = display
            rows.append(row)
        setfit = sub[sub["method"].map(_is_setfit_bge)]
        if len(setfit):
            row = setfit.iloc[0].copy()
            # One display row across datasets (16-shot AG/RT, 8-shot Banking77).
            row["display_method"] = "SetFit (BGE)"
            rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def write_cherry_csvs(cherry: pd.DataFrame) -> pd.DataFrame:
    cherry_path = REPORT_DIR / "master_comparison_cherry.csv"
    cherry.to_csv(cherry_path, index=False)
    print(f"wrote {cherry_path.relative_to(ROOT)}")

    pivot = cherry.pivot_table(
        index="display_method", columns="dataset", values="f1", sort=False
    )
    # Preserve cherry order
    order = []
    for _, display in CHERRY_METHODS:
        if display not in order and display in pivot.index:
            order.append(display)
    for idx in pivot.index:
        if idx not in order:
            order.append(idx)
    pivot = pivot.reindex(index=order, columns=[c for c in DATASET_ORDER if c in pivot.columns])
    pivot_path = REPORT_DIR / "master_f1_pivot_cherry.csv"
    pivot.round(6).to_csv(pivot_path)
    print(f"wrote {pivot_path.relative_to(ROOT)}")
    return pivot


def plot_f1_heatmap(pivot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(pivot) + 1.6))
    vals = pivot.values * 100
    image = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    mean = np.nanmean(vals)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v < mean else "black",
                )
    fig.colorbar(image, ax=ax, label="Macro-F1 (%)")
    ax.set_title("Macro-F1 cherry-pick (default preprocessing sw0_tok1_pad1_neg0)")
    _finish(fig, FIG_DIR / "master_f1_heatmap.png")


def plot_accuracy_f1_bars(cherry: pd.DataFrame) -> None:
    for dataset in DATASET_ORDER:
        df = cherry[cherry["dataset"] == dataset].copy()
        if df.empty:
            continue
        df = df.sort_values("f1")
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.4))
        ypos = np.arange(len(df))
        colors = [FAMILY_COLORS.get(f, "#888888") for f in df["family"]]
        ax.barh(ypos + 0.2, df["accuracy"] * 100, height=0.4, color=colors, alpha=0.45, label="Accuracy")
        ax.barh(ypos - 0.2, df["f1"] * 100, height=0.4, color=colors, label="Macro-F1")
        for y, f1 in zip(ypos, df["f1"]):
            ax.text(f1 * 100 + 0.3, y - 0.2, f"{f1 * 100:.1f}", va="center", fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["display_method"], fontsize=8)
        lo = max(0, np.floor(min(df["f1"].min(), df["accuracy"].min()) * 100 / 5) * 5 - 5)
        ax.set_xlim(lo, 100)
        ax.set_xlabel("score (%)  [solid = Macro-F1, light = Accuracy]")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
        ax.legend(handles, FAMILY_COLORS.keys(), loc="lower right", fontsize=7)
        ax.set_title(f"{dataset}: cherry-pick Accuracy and Macro-F1")
        slug = DATASET_SLUG[dataset]
        _finish(fig, FIG_DIR / slug / "accuracy_f1_bars.png")


def plot_ablation_heatmaps(ablation: pd.DataFrame) -> None:
    for dataset in DATASET_ORDER:
        sub = ablation[ablation["dataset"] == dataset]
        if sub.empty:
            continue
        pivot = sub.pivot(index="model", columns="flag", values="delta_f1")
        pivot = pivot.reindex(
            index=[m for m in ABLATION_MODELS if m in pivot.index],
            columns=[f for f in ABLATION_FLAGS if f in pivot.columns],
        ) * 100
        vmax = max(1.0, float(np.nanmax(np.abs(pivot.values))))
        fig, ax = plt.subplots(figsize=(9.0, 3.4))
        image = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{c}\n(True - False)" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(
                    j, i, "n/a" if np.isnan(v) else f"{v:+.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if np.isnan(v) or abs(v) < 0.6 * vmax else "white",
                )
        fig.colorbar(image, ax=ax, label="delta Macro-F1 (pp)")
        ax.set_title(f"{dataset}: effect of switching each flag on")
        slug = DATASET_SLUG[dataset]
        _finish(fig, FIG_DIR / slug / "preprocessing_ablation_heatmap.png")


def main() -> None:
    master = pd.read_csv(REPORT_DIR / "master_comparison.csv")
    ablation = pd.read_csv(REPORT_DIR / "master_ablation.csv")
    cherry = select_cherry(master)
    pivot = write_cherry_csvs(cherry)
    plot_f1_heatmap(pivot)
    plot_accuracy_f1_bars(cherry)
    plot_ablation_heatmaps(ablation)
    print(f"cherry rows: {len(cherry)} ({cherry.dataset.nunique()} datasets)")


if __name__ == "__main__":
    main()
