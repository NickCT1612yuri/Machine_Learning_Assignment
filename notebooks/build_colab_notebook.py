"""
Generate notebooks/text_classification_colab.ipynb from the current modules/ sources.

The notebook embeds every modules/*.py file, so when it is opened on Google Colab
without the repository it can recreate the package itself (no Drive mount).

    python notebooks/build_colab_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 notebooks/text_classification_colab.ipynb

    # after editing modules/ (or the conclusions) without re-running the notebook
    python notebooks/build_colab_notebook.py --refresh-sources
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "text_classification_colab.ipynb"

GROUP_TABLE = """| Full name | Student ID | Email | Main responsibilities | Contribution |
| --- | --- | --- | --- | --- |
| [Member 1 full name] | [Student ID] | [member1@hcmut.edu.vn] | Data loading, boolean preprocessing, EDA, traditional features (BoW / TF-IDF), Naive Bayes / LogReg / SVM, ablation study, README | 50% |
| [Member 2 full name] | [Student ID] | [member2@hcmut.edu.vn] | Word2Vec / DistilBERT / BGE-small, .npy / .h5 persistence, XGBoost, BiLSTM + TextCNN + SetFit 3x3, evaluation figures, report | 50% |"""


CONCLUSIONS = """
## 7. Conclusions

Numbers are Macro-F1 on the test sets of the run saved in this notebook (single seed 42). Cherry-pick headline rows for the PDF report after inspecting `reports/master_comparison.csv`.

1. **Task regime still decides whether contextual representations pay off** (AG News vs Rotten Tomatoes / Banking77).
2. **Preprocessing must match the task**: stop-word removal, tokenization, padding, and the new **negation marking** switch are ablated one at a time.
3. **Static Word2Vec + BiLSTM/TextCNN** vs **frozen DistilBERT/BGE** vs **SetFit** covers classical, sequence, and few-shot Transformer fine-tuning under one configurable API (all 9 DL embedding x model cells are runnable).
4. **XGBoost** replaces TabPFN as the nonlinear classical head on full sparse/dense features; PCA-30 + LogReg remains the projection control.
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
    """Update the embedded module sources and the conclusions in the executed notebook, keeping all outputs."""
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
# Configurable Text Classification Pipeline
### From sparse lexical baselines to Word2Vec, frozen Transformers, XGBoost and SetFit

**Course:** Machine Learning ([Course code, e.g. CO3117]) - Semester I, Academic Year 2026-2027
**Faculty:** Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
**Instructor:** [Instructor full name and title] ([instructor email])
**Track:** Task 2 - Machine Learning with Text Data (+ bonus deep-learning pipeline)
**GitHub repository:** [https://github.com/your-username/your-repo](https://github.com/your-username/your-repo)

{GROUP_TABLE}

**What this notebook does** (Runtime -> Run all, no Google Drive, every dataset downloaded from a public Hugging Face link):

| Section | Content |
| --- | --- |
| 0 | install missing libraries, recreate `modules/` if the notebook is opened alone |
| 1 | EDA on AG News, Rotten Tomatoes and Banking77 (word counts, length distribution, word frequencies, class balance) |
| 2 | boolean preprocessing switches `stop_word_removal`, `tokenization`, `padding` and their ablation |
| 3 | feature extraction: BoW, TF-IDF (+SVD-30), DistilBERT (768-d), BGE-small (384-d) (+PCA-30), saved as `.npy` **and** `.h5` |
| 4 | classifiers: Naive Bayes, Logistic Regression, calibrated LinearSVC, XGBoost; accuracy, macro P/R/F1, Brier score, ECE |
| 5 | deep learning 3x3: BiLSTM / TextCNN / SetFit x Word2Vec / DistilBERT / BGE-small |
| 6 | cross-dataset synthesis, calibration, confusion matrices, error and negation analysis, conclusions |

All logic lives in the `modules/` package; the notebook only calls it, so the same code path is used by `python main.py`.
"""))

    cells.append(md("""
## 0. Setup

The first cell installs only the libraries that are missing (Colab already ships `torch`, `transformers`, `pandas`, `scikit-learn`).
The second cell finds the `modules/` package: it uses the repository checkout if present, otherwise clones the GitHub repository (when `REPO_URL` is set) or writes the module files embedded in this notebook. A GPU runtime (T4) is recommended; everything also runs on CPU or Apple MPS.
"""))

    cells.append(code("""
import importlib.util, subprocess, sys

REQUIRED = {
    "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "sklearn": "scikit-learn", "scipy": "scipy", "h5py": "h5py", "torch": "torch",
    "transformers": "transformers", "datasets": "datasets",
    "sentence_transformers": "sentence-transformers", "setfit": "setfit",
    "xgboost": "xgboost",
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
    print(f"{cfg['title']:16s} https://huggingface.co/datasets/{cfg['hf_id']}  "
          f"train cap={cfg['max_train']}, test cap={cfg['max_test']}")
"""))

    cells.append(md("""
## 1. Exploratory data analysis

| Dataset (Hugging Face) | Task | Classes | Split used | Role in the comparison |
| --- | --- | --- | --- | --- |
| `fancyzhx/ag_news` | news topic | 4 | 10,000 / 2,000 (stratified sample of the official split) | lexical regime: topic keywords separate the classes |
| `cornell-movie-review-data/rotten_tomatoes` | sentence sentiment | 2 | 8,530 / 1,066 (official) | compositional regime: short critic sentences, negation, irony |
| `mteb/banking77` | customer intent | 77 | 9,993 / 3,076 (official) | high-cardinality regime: 77 intents sharing the same nouns |

For each dataset we report the class balance, word-count statistics (mean, std, median, 95th / 99th percentile), the vocabulary size with and without stop words, and the top-15 words with and without stop words. The 95th percentile is the padding cut-off used later by the neural models.
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
| `stop_word_removal` | keep every word, including negations (*not, no, never*) and prepositions (*into, from*) | drop scikit-learn's `ENGLISH_STOP_WORDS` from the text |
| `tokenization` | plain whitespace split: unigrams for BoW/TF-IDF, whole-word lookup for Transformers | regex words + bigrams for BoW/TF-IDF, WordPiece for Transformers |
| `padding` | truncate to 10 tokens and pool / recur over padding without a mask | pad/truncate to the 95th-percentile length and mask padding |
| `negation_marking` | leave tokens unchanged | join negation cues with the next word (`not good` -> `not_good`) before stop-word removal |

The default (main) configuration is `sw0_tok1_pad1_neg0`. The ablation flips **one switch at a time** for TF-IDF + LogReg, BGE-small + LogReg and Word2Vec + BiLSTM.
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

| Key | Family | Representation | 30-d projection |
| --- | --- | --- | --- |
| `bow` | traditional | CountVectorizer, up to 25,000 uni+bigrams, `min_df=2` (sparse `.npz`) | TruncatedSVD(30) |
| `tfidf` | traditional | sublinear TF-IDF, same vocabulary settings (sparse `.npz`) | TruncatedSVD(30) |
| `word2vec` | static | skip-gram Word2Vec on train texts, mean-pooled 300-d (+ `word2vec.kv` for DL) | StandardScaler + PCA(30) |
| `distilbert` | modern | frozen `distilbert-base-uncased`, attention-masked mean pooling, 768-d | StandardScaler + PCA(30) |
| `bge_small` | modern | frozen `BAAI/bge-small-en-v1.5`, CLS pooling + L2 normalisation, 384-d | StandardScaler + PCA(30) |

Everything is fit on the training split only. Each folder `features/<dataset>/<embedding>_<tag>/` holds `X_train.npy`, `X_test.npy` (dense) or `X_*_sparse.npz` (sparse), `X_*_pca30.npy`, `y_*.npy`, a single `features.h5` with the same arrays, and `meta.json`. The classifiers in Section 4 read these files.
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

# The .h5 copy holds exactly the same arrays as the .npy files.
folder = FEATURE_DIR / "rotten_tomatoes" / "bge_small_sw0_tok1_pad1_neg0"
h5 = load_h5_features(folder / "features.h5")
print("h5 keys:", sorted(k for k in h5 if k != "meta"))
print("X_train.npy == features.h5['X_train']:", np.array_equal(np.load(folder / "X_train.npy"), h5["X_train"]))
"""))

    cells.append(md("""
## 4. Classifiers on the saved features

| Classifier | Sparse input (BoW / TF-IDF) | Dense input (embeddings) |
| --- | --- | --- |
| Naive Bayes | `MultinomialNB(alpha=0.5)` | `GaussianNB()` |
| Logistic Regression | `LogisticRegression(C=2.0, lbfgs, class_weight="balanced")` | StandardScaler + same |
| Linear SVM | `CalibratedClassifierCV(LinearSVC(C=1.0), cv=3, sigmoid)` (Platt scaling gives probabilities) | L2 normaliser + same |
| XGBoost | `XGBClassifier` (hist, multi:softprob) on full features | same on dense / Word2Vec / Transformer vectors |

The "(PCA-30) + LogReg" rows use the 30-d projection as a control. Metrics: accuracy, macro precision / recall / F1, multiclass Brier score and expected calibration error (ECE).
"""))

    cells.append(code("""
show_cols = ["method", "input_dim", "accuracy", "f1", "brier_score", "ece", "train_time_sec", "infer_time_sec"]
main_tables = {}
for name in DATASET_ORDER:
    main_tables[name] = run_main_comparison(name)
    display(main_tables[name][show_cols].round(4))
"""))

    cells.append(md("""
## 5. Deep-learning pipeline (3 models x 3 embedding backends)

* **BiLSTM** / **TextCNN** over Word2Vec (trainable lookup), or frozen DistilBERT / BGE token hidden states.
* **SetFit** few-shot contrastive fine-tuning: native BGE-small, DistilBERT SentenceTransformer adapter, Word2Vec mean-pool adapter (all three runnable for API completeness).
* Shots: 16 per class when classes <= 10, else 8 (Banking77).
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

For every dataset: the full comparison table, Accuracy / Macro-F1 bars, Brier score vs Macro-F1, a reliability diagram, the ablation heat map, the confusion matrix of the best model, its most frequent confusions, and the accuracy on test documents with vs without a negation word.
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
