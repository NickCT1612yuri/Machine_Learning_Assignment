"""
Generate notebooks/text_classification_colab.ipynb from the current modules/ sources.

The notebook embeds every modules/*.py file, so when it is opened on Google Colab
without the repository it can recreate the package itself (no Drive mount).

    python notebooks/build_colab_notebook.py
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "text_classification_colab.ipynb"

GROUP_TABLE = """| Full name | Student ID | Email | Main responsibilities | Contribution |
| --- | --- | --- | --- | --- |
| [Member 1 full name] | [Student ID] | [member1@hcmut.edu.vn] | Data loading, boolean preprocessing, EDA, TF-IDF, Naive Bayes / LogReg / SVM, ablation, README | 50% |
| [Member 2 full name] | [Student ID] | [member2@hcmut.edu.vn] | Word2Vec / DistilBERT, .npy / .h5 persistence, bonus Word2Vec + BiLSTM, evaluation figures, report | 50% |"""


CONCLUSIONS = """
## 7. Conclusions

Numbers are Macro-F1 on the Banking77 test set from this notebook run (single seed 42).

1. **Fine-grained intent classification stresses lexical baselines**: TF-IDF remains strong, but frozen DistilBERT and sequence modelling test whether representation geometry separates overlapping intents.
2. **Preprocessing is representation-dependent**: stop-word removal, tokenization, padding, and negation marking are ablated one at a time on TF-IDF + LogReg, DistilBERT + LogReg, and Word2Vec + BiLSTM.
3. **Mean-pooled Word2Vec vs BiLSTM** isolates the value of order for short banking queries under a shared static embedding table.
4. **Macro-F1 and calibration** (Brier, ECE) must be read together on a 77-way problem with class imbalance.
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def module_sources() -> dict:
    return {p.name: p.read_text() for p in sorted((ROOT / "modules").glob("*.py"))}


def bootstrap_source(sources: dict) -> str:
    return f"""
import os
from pathlib import Path

REPO_URL = ""  # e.g. "https://github.com/your-username/your-repo.git" once the repository is public
MODULE_SOURCES = {sources!r}

here = Path.cwd()
if (here / "modules" / "__init__.py").exists():
    ROOT = here
elif (here.parent / "modules" / "__init__.py").exists():
    ROOT = here.parent
else:
    ROOT = here
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "repo"], check=False)
        if (here / "repo" / "modules" / "__init__.py").exists():
            ROOT = here / "repo"
    if not (ROOT / "modules" / "__init__.py").exists():
        (ROOT / "modules").mkdir(exist_ok=True)
        for filename, text in MODULE_SOURCES.items():
            (ROOT / "modules" / filename).write_text(text)
        print(f"wrote {{len(MODULE_SOURCES)}} module files to {{ROOT / 'modules'}}")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("project root:", ROOT)
"""


def refresh_sources():
    """Update the embedded module sources and the conclusions, keeping outputs."""
    nb = nbf.read(OUT, as_version=4)
    found = set()
    for cell in nb.cells:
        if cell.cell_type == "code" and "MODULE_SOURCES = " in cell.source:
            cell.source = bootstrap_source(module_sources()).strip()
            found.add("sources")
        elif cell.cell_type == "markdown" and cell.source.startswith("## 7. Conclusions"):
            cell.source = CONCLUSIONS.strip()
            found.add("conclusions")
    if found != {"sources", "conclusions"}:
        raise SystemExit(f"expected cells not found (found: {sorted(found)})")
    nbf.write(nb, OUT)
    print(f"refreshed embedded module sources and conclusions in {OUT}")


def build():
    sources = module_sources()
    cells = []

    cells.append(md(f"""
# Configurable Text Classification Pipeline — Banking77
### TF-IDF, Word2Vec, DistilBERT, classical classifiers, and bonus Word2Vec + BiLSTM

**Course:** Machine Learning ([Course code, e.g. CO3117]) - Semester I, Academic Year 2026-2027
**Faculty:** Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
**Instructor:** [Instructor full name and title] ([instructor email])
**Track:** Task 2 - Machine Learning with Text Data (+ bonus deep-learning pipeline)
**GitHub repository:** [https://github.com/your-username/your-repo](https://github.com/your-username/your-repo)

{GROUP_TABLE}

**What this notebook does** (Runtime -> Run all, no Google Drive; Banking77 downloaded from a public Hugging Face link):

| Section | Content |
| --- | --- |
| 0 | install missing libraries, recreate `modules/` if the notebook is opened alone |
| 1 | EDA on Banking77 (word counts, length distribution, word frequencies, class balance) |
| 2 | boolean preprocessing switches and one-flag-at-a-time ablation |
| 3 | feature extraction: TF-IDF, Word2Vec, DistilBERT, saved as `.npy` and `.h5` |
| 4 | classifiers: Naive Bayes, Logistic Regression, calibrated LinearSVC |
| 5 | bonus deep learning: Word2Vec + BiLSTM |
| 6 | comparison tables, calibration, confusion, error and negation analysis, conclusions |

All logic lives in `modules/`; the notebook calls the same code path as `python main.py`.
"""))

    cells.append(md("""
## 0. Setup

The first cell installs only missing libraries. The second finds `modules/`: repository checkout if present, otherwise clones `REPO_URL` or writes the embedded module files. A GPU runtime (T4) is recommended; CPU / MPS also work.
"""))

    cells.append(code("""
import importlib.util, subprocess, sys

REQUIRED = {
    "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "sklearn": "scikit-learn", "scipy": "scipy", "h5py": "h5py", "torch": "torch",
    "transformers": "transformers", "datasets": "datasets",
}
missing = [pip for module, pip in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    print("installing:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("all required libraries are already installed")
"""))

    cells.append(code(bootstrap_source(sources)))

    cells.append(code("""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

from modules import (DATASET_ORDER, DATASET_REGISTRY, FEATURE_DIR, REPORT_DIR, FIG_DIR, PreprocessConfig,
                     clean_text, seed_everything, get_device, run_all_eda, run_preprocessing_ablation,
                     run_feature_extraction, list_feature_files, load_h5_features, run_main_comparison,
                     run_deep_learning, plot_suite_results, error_analysis, negation_analysis, build_master_table)
from modules import evaluation as ev

%matplotlib inline
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
seed_everything(42)
print("device:", get_device())
for name in DATASET_ORDER:
    cfg = DATASET_REGISTRY[name]
    print(f"{cfg['title']:16s} https://huggingface.co/datasets/{cfg['hf_id']}")
"""))

    cells.append(md("""
## 1. Exploratory data analysis

| Dataset (Hugging Face) | Task | Classes | Split used | Role |
| --- | --- | --- | --- | --- |
| `mteb/banking77` | online-banking customer intent | 77 | 9,993 / 3,076 (official) | fine-grained intents that share nouns and differ by function words / negation |

We report class balance, word-count statistics, vocabulary size with and without stop words, and top-15 words. The 95th percentile length is the padding cut-off for DistilBERT and BiLSTM.
"""))

    cells.append(code("""
eda = run_all_eda(DATASET_ORDER, show=True)
display(eda.set_index("dataset").T)
"""))

    cells.append(md("""
## 2. Boolean preprocessing configuration and ablation

`PreprocessConfig(stop_word_removal, tokenization, padding, negation_marking)` is passed to every step; its tag (e.g. `sw0_tok1_pad1_neg0`) names the feature cache folder.

| Switch | `False` | `True` |
| --- | --- | --- |
| `stop_word_removal` | keep every word | drop scikit-learn `ENGLISH_STOP_WORDS` |
| `tokenization` | whitespace unigrams / whole-word lookup | bigrams for TF-IDF; WordPiece for DistilBERT |
| `padding` | truncate to 10 tokens; no masking | pad/truncate to 95th-percentile length with masks |
| `negation_marking` | unchanged | join negation cues with the next word (`not arrived` -> `not_arrived`) |

Default main configuration: `sw0_tok1_pad1_neg0`. Ablation flips **one switch at a time** for TF-IDF + LogReg, DistilBERT + LogReg, and Word2Vec + BiLSTM.
"""))

    cells.append(code("""
example = "I'm still waiting for my card, it has NOT arrived from the bank yet!"
for cfg in [PreprocessConfig(False, True, True, False), PreprocessConfig(True, True, True, False),
            PreprocessConfig(False, True, True, True)]:
    print(f"{cfg.tag()}: {clean_text(example, cfg.stop_word_removal, True, cfg.negation_marking)!r}")
"""))

    cells.append(code("""
ablation = {}
for name in DATASET_ORDER:
    runs, delta = run_preprocessing_ablation(name)
    ablation[name] = delta
    display(delta[["dataset", "model", "flag", "f1_false", "f1_true", "delta_f1", "note"]].round(4))
    ev.plot_ablation_heatmap(name, delta, show=True)
"""))

    cells.append(md("""
## 3. Feature extraction (saved as `.npy` and `.h5`)

| Key | Family | Representation |
| --- | --- | --- |
| `tfidf` | traditional | sublinear TF-IDF, uni+bigrams, `min_df=2` (sparse `.npz`) |
| `word2vec` | static | skip-gram on train texts, mean-pooled 300-d (+ `word2vec.kv.npz` for BiLSTM) |
| `distilbert` | modern | frozen `distilbert-base-uncased`, attention-masked mean pooling, 768-d |

Fit on the training split only. Each folder `features/banking77/<embedding>_<tag>/` holds dense `.npy` or sparse `.npz`, optional 30-d projections, labels, `features.h5`, and `meta.json`.
"""))

    cells.append(code("""
feature_table = pd.concat([run_feature_extraction(name) for name in DATASET_ORDER], ignore_index=True)
display(feature_table)
"""))

    cells.append(code("""
files = pd.DataFrame(list_feature_files(FEATURE_DIR))
summary = files.groupby(["dataset", "folder"]).agg(files=("file", "count"), size_mb=("size_mb", "sum")).round(2)
display(summary)
print(f"{len(files)} feature files, {files['size_mb'].sum():.1f} MB in {FEATURE_DIR}")

folder = FEATURE_DIR / "banking77" / "distilbert_sw0_tok1_pad1_neg0"
h5 = load_h5_features(folder / "features.h5")
print("h5 keys:", sorted(k for k in h5 if k != "meta"))
print("X_train.npy == features.h5['X_train']:", np.array_equal(np.load(folder / "X_train.npy"), h5["X_train"]))
"""))

    cells.append(md("""
## 4. Classifiers on the saved features

| Classifier | Sparse input (TF-IDF) | Dense input (embeddings) |
| --- | --- | --- |
| Naive Bayes | `MultinomialNB(alpha=0.5)` | `GaussianNB()` |
| Logistic Regression | `LogisticRegression(C=2.0, balanced)` | StandardScaler + same |
| Linear SVM | calibrated `LinearSVC(C=1.0)` (Platt) | L2 Normalizer + same |

Metrics: accuracy, macro precision / recall / F1, multiclass Brier score, ECE.
"""))

    cells.append(code("""
show_cols = ["method", "input_dim", "accuracy", "f1", "brier_score", "ece", "train_time_sec", "infer_time_sec"]
main_tables = {}
for name in DATASET_ORDER:
    main_tables[name] = run_main_comparison(name)
    display(main_tables[name][show_cols].round(4))
"""))

    cells.append(md("""
## 5. Bonus deep-learning pipeline (Word2Vec + BiLSTM)

A bidirectional LSTM over the same skip-gram Word2Vec table used for mean-pooled document vectors. Packing / masking follows the `padding` preprocess switch. Early stopping uses a stratified 10% validation split of the training set.
"""))

    cells.append(code("""
dl_tables = {}
for name in DATASET_ORDER:
    dl_tables[name] = run_deep_learning(name)
    display(dl_tables[name][["dataset", "method", "accuracy", "f1", "brier_score", "ece",
                             "train_time_sec", "infer_time_sec", "notes"]].round(4))
    loss_png = FIG_DIR / name / "bilstm_loss.png"
    if loss_png.exists():
        display(Image(filename=str(loss_png)))
"""))

    cells.append(md("""
## 6. Results, calibration and error analysis

Accuracy / Macro-F1 bars, Brier vs Macro-F1, reliability diagram, ablation heat map, confusion matrix of the best model, frequent confusions, and accuracy on negated vs non-negated test queries.
"""))

    cells.append(code("""
results = {}
for name in DATASET_ORDER:
    print("=" * 100)
    print(DATASET_REGISTRY[name]["title"])
    results[name] = ev.save_comparison(name)
    display(results[name][["method", "family", "accuracy", "precision", "recall", "f1", "brier_score", "ece"]].round(4))
    plot_suite_results(name, results[name], ablation.get(name), show=True)
    display(error_analysis(name)[["true", "predicted", "count", "share_of_true_class", "example_1"]])
    display(negation_analysis(name).round(4))
"""))

    cells.append(code("""
master = build_master_table(DATASET_ORDER, show=True)
pivot = pd.read_csv(REPORT_DIR / "master_f1_pivot.csv", index_col=0)
display((pivot * 100).round(2))
best = master.loc[master.groupby("dataset")["f1"].idxmax(), ["dataset", "method", "accuracy", "f1", "brier_score"]]
display(best.round(4))
"""))

    cells.append(code("""
print("Deliverables written by this notebook:")
for path in sorted(REPORT_DIR.glob("*.csv")):
    print("  ", path.relative_to(ROOT))
print(f"   {len(list(FIG_DIR.rglob('*.png')))} figures under {FIG_DIR.relative_to(ROOT)}/")
print(f"   {len(list_feature_files(FEATURE_DIR))} feature files (.npy / .npz / .h5) under {FEATURE_DIR.relative_to(ROOT)}/")
"""))

    cells.append(md(CONCLUSIONS))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells, {len(sources)} embedded module files)")


if __name__ == "__main__":
    if "--refresh-sources" in sys.argv:
        refresh_sources()
    else:
        build()
