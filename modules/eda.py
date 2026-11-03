"""
Exploratory data analysis required by the assignment: word-count statistics,
text-length distribution, class balance and word frequencies (with and without
stop words). Figures go to reports/figures/<dataset>/.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.config import FIG_DIR
from modules.data_loader import STOP_WORDS, normalize_text, word_counts

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
# Function words that carry polarity or direction; highlighted in the word-frequency plot.
SYNTACTIC_OPERATORS = {
    "not", "no", "never", "nor", "nothing", "none", "cannot", "without",
    "still", "into", "from", "but", "too", "very", "can", "how", "why", "again",
}


def _words(texts):
    for text in texts:
        yield from _WORD.findall(normalize_text(text))


def _finish(fig, path: Path, show: bool):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def run_eda(dataset_dict: dict, dataset_name: str | None = None, fig_dir: Path | None = None,
            show: bool = False, top_k: int = 15) -> pd.DataFrame:
    """Print EDA statistics, save three figures, and return a one-row summary DataFrame."""
    name = dataset_name or dataset_dict["name"]
    out = Path(fig_dir) if fig_dir is not None else FIG_DIR / name
    title = dataset_dict["title"]
    train_raw, test_raw = dataset_dict["train_raw"], dataset_dict["test_raw"]
    labels, names = dataset_dict["train_label"], dataset_dict["label_names"]

    train_len = word_counts([normalize_text(t) for t in train_raw])
    test_len = word_counts([normalize_text(t) for t in test_raw])
    counts = np.bincount(labels, minlength=len(names))
    all_counts = Counter(_words(train_raw))
    content_counts = Counter({w: c for w, c in all_counts.items() if w not in STOP_WORDS})
    total_tokens = sum(all_counts.values())
    stop_tokens = total_tokens - sum(content_counts.values())
    operator_tokens = sum(all_counts[w] for w in SYNTACTIC_OPERATORS & STOP_WORDS)
    docs_with_negation = np.mean([
        bool({"not", "no", "never", "nothing", "cannot"} & set(_WORD.findall(normalize_text(t))))
        or "n't" in normalize_text(t)
        for t in train_raw
    ])

    summary = {
        "dataset": title,
        "n_train": len(train_raw),
        "n_test": len(test_raw),
        "n_classes": len(names),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "balance_ratio": round(float(counts.min() / counts.max()), 3),
        "words_mean": round(float(train_len.mean()), 2),
        "words_std": round(float(train_len.std()), 2),
        "words_min": int(train_len.min()),
        "words_median": round(float(np.median(train_len)), 1),
        "words_p95": round(float(np.percentile(train_len, 95)), 1),
        "words_p99": round(float(np.percentile(train_len, 99)), 1),
        "words_max": int(train_len.max()),
        "test_words_mean": round(float(test_len.mean()), 2),
        "vocab_with_stopwords": len(all_counts),
        "vocab_without_stopwords": len(content_counts),
        "stopword_token_share": round(stop_tokens / max(total_tokens, 1), 3),
        "operator_token_share": round(operator_tokens / max(total_tokens, 1), 4),
        "docs_with_negation": round(float(docs_with_negation), 3),
    }

    print(f"--- EDA: {title} ---")
    print(f"train/test: {summary['n_train']}/{summary['n_test']}, classes: {summary['n_classes']}, "
          f"class counts min/max: {summary['min_class_count']}/{summary['max_class_count']} "
          f"(ratio {summary['balance_ratio']})")
    print(f"words/doc: mean={summary['words_mean']}, std={summary['words_std']}, "
          f"median={summary['words_median']:.0f}, p95={summary['words_p95']:.0f}, "
          f"p99={summary['words_p99']:.0f}, max={summary['words_max']}")
    print(f"vocabulary: {summary['vocab_with_stopwords']} with stop words, "
          f"{summary['vocab_without_stopwords']} without; stop words are "
          f"{summary['stopword_token_share']:.1%} of tokens; "
          f"{summary['docs_with_negation']:.1%} of documents contain a negation")

    # 1. Class balance
    many = len(names) > 20
    order = np.argsort(-counts) if many else np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12 if many else 7, 4))
    ax.bar(range(len(names)), counts[order], color="#3d5a80")
    ax.axhline(counts.mean(), color="#ee6c4d", ls="--", lw=1, label=f"mean = {counts.mean():.0f}")
    ax.set_title(f"{title}: training documents per class")
    ax.set_ylabel("documents")
    if many:
        ax.set_xlabel(f"{len(names)} intents, sorted by frequency")
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([names[i] for i in order], rotation=0)
    ax.legend()
    _finish(fig, out / "class_balance.png", show)

    # 2. Length distribution with median and p95 padding cut-off
    cap = int(np.percentile(train_len, 99))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_len.clip(max=cap), bins=min(40, max(cap, 5)), color="#98c1d9", edgecolor="white")
    med, p95 = np.median(train_len), np.percentile(train_len, 95)
    ax.axvline(med, color="#293241", ls="--", label=f"median = {med:.0f}")
    ax.axvline(p95, color="#ee6c4d", ls="--", label=f"95th pct (padding cut-off) = {p95:.0f}")
    ax.set_title(f"{title}: words per training document")
    ax.set_xlabel("word count (clipped at the 99th percentile)")
    ax.set_ylabel("documents")
    ax.legend()
    _finish(fig, out / "length_hist.png", show)

    # 3. Top words with vs without stop words
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, counter, label in [
        (axes[0], all_counts, "with stop words"),
        (axes[1], content_counts, "stop words removed"),
    ]:
        top = counter.most_common(top_k)
        words = [w for w, _ in top][::-1]
        freqs = [c for _, c in top][::-1]
        colors = ["#ee6c4d" if w in SYNTACTIC_OPERATORS else "#3d5a80" for w in words]
        ax.barh(words, freqs, color=colors)
        ax.set_title(f"Top {top_k} words ({label})")
        ax.set_xlabel("frequency in training set")
    fig.suptitle(f"{title}: word frequencies (orange = negation / direction / question words)")
    _finish(fig, out / "top_words_comparison.png", show)

    print(f"[eda] figures saved to {out}")
    return pd.DataFrame([summary])
