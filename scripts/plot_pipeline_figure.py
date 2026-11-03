#!/usr/bin/env python3
"""Draw a professional pipeline overview figure for the academic report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "pipeline_overview.png"

# Soft academic palette (distinct stages, not flashy)
C_DATA = "#E8F1F8"
C_EDA = "#F3EDE3"
C_FEAT = "#E6F0EA"
C_CLF = "#F7E9E4"
C_DL = "#EEE8F5"
C_EVAL = "#E8ECEF"
EDGE = "#2F3A44"
TEXT = "#1F2933"


def _box(ax, xy, w, h, text, facecolor, fontsize=8.5, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1, edgecolor=EDGE, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=TEXT, weight=weight, wrap=True,
        linespacing=1.25,
    )
    return (x + w / 2, y + h, x + w / 2, y, x, x + w)  # top_mid_x, top_y, bot_mid_x, bot_y, left, right


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=EDGE, shrinkA=1, shrinkB=1,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Row 1: data source
    src = _box(ax, (3.6, 7.35), 4.0, 0.55,
               "Public Hugging Face datasets", C_DATA, fontsize=9.5, weight="bold")

    # Row 2: loader
    loader = _box(ax, (3.1, 6.35), 5.0, 0.65,
                  "data_loader + PreprocessConfig\n(stop / tok / pad / negation switches)",
                  C_DATA, fontsize=8.5)

    # Row 3: EDA (side) + feature extraction (main)
    eda = _box(ax, (0.35, 5.15), 2.7, 0.7,
               "EDA\nstatistics & figures", C_EDA, fontsize=8.5)
    feat = _box(ax, (3.6, 5.05), 4.0, 0.85,
                "feature_extraction\nBoW · TF-IDF · Word2Vec\nDistilBERT · BGE-small",
                C_FEAT, fontsize=8.5, weight="bold")

    # Features on disk
    disk = _box(ax, (3.35, 3.95), 4.5, 0.7,
                "features/<dataset>/<embedding>_<tag>/\n.npy  ·  .npz  ·  .h5",
                C_FEAT, fontsize=8)

    # Classical vs DL branches
    clf = _box(ax, (0.55, 2.45), 4.3, 1.05,
               "Classical classifiers\nNaive Bayes · LogReg\nLinearSVC · XGBoost",
               C_CLF, fontsize=8.5, weight="bold")
    dl = _box(ax, (6.35, 2.45), 4.3, 1.05,
              "Deep learning (bonus)\nBiLSTM · TextCNN · SetFit\n× Word2Vec / DistilBERT / BGE",
              C_DL, fontsize=8.5, weight="bold")

    # Evaluation
    ev = _box(ax, (2.85, 0.55), 5.5, 0.85,
              "evaluation\nmetrics · tables · figures · error / negation analysis",
              C_EVAL, fontsize=9, weight="bold")

    # Arrows
    _arrow(ax, (src[0], src[3]), (loader[0], loader[1]))
    _arrow(ax, (loader[0], loader[3]), (feat[0], feat[1]))
    # loader -> EDA (side)
    _arrow(ax, (loader[4] + 0.35, loader[3]), (eda[0], eda[1]))
    _arrow(ax, (feat[0], feat[3]), (disk[0], disk[1]))
    # disk -> classical / DL
    _arrow(ax, (disk[4] + 0.4, disk[3]), (clf[0], clf[1]))
    _arrow(ax, (disk[5] - 0.4, disk[3]), (dl[0], dl[1]))
    # also note DL can use preprocessed text / token states: light side note
    ax.text(8.5, 4.15, "bonus path also uses\npreprocessed sequences\n/ frozen token states",
            ha="center", va="center", fontsize=7, color="#5B6B7A", style="italic")
    # both -> evaluation
    _arrow(ax, (clf[0], clf[3]), (ev[4] + 1.2, ev[1]))
    _arrow(ax, (dl[0], dl[3]), (ev[5] - 1.2, ev[1]))

    ax.set_title(
        "Traditional and bonus text-classification pipelines",
        fontsize=12, color=TEXT, pad=8, weight="bold",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
