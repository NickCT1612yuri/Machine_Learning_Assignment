# Configurable Text Classification Pipeline: from Sparse Lexical Baselines to Word2Vec, Frozen Transformers, XGBoost and SetFit

## 1. Course and academic information

| | |
| --- | --- |
| University | Ho Chi Minh City University of Technology (HCMUT), VNU-HCM |
| Faculty | Faculty of Computer Science and Engineering |
| Course | Machine Learning |
| Course code | [Course code, e.g. CO3117] |
| Semester / academic year | Semester I, 2026-2027 |
| Track | Task 2 - Machine Learning with Text Data (+ bonus deep-learning pipeline) |
| Instructor | [Instructor full name and title] ([instructor email]) |

## 2. Group members and work distribution

Individual grade = group grade x contribution percentage (assignment Section 6).

| Full name | Student ID | Email | Main responsibilities | Contribution |
| --- | --- | --- | --- | --- |
| [Member 1 full name] | [Student ID] | [member1@hcmut.edu.vn] | Data loading and boolean preprocessing (`data_loader.py`, `config.py`), EDA (`eda.py`), traditional features (BoW / TF-IDF), Naive Bayes / LogReg / LinearSVC, preprocessing ablation, README and repository | 50% |
| [Member 2 full name] | [Student ID] | [member2@hcmut.edu.vn] | Word2Vec / DistilBERT / BGE-small and `.npy` / `.h5` persistence (`feature_extraction.py`), XGBoost (`classifiers.py`), bonus BiLSTM + TextCNN + SetFit 3×3 (`deep_learning.py`), evaluation figures (`evaluation.py`), PDF report | 50% |

Collaboration evidence (meeting minutes, photos, chat screenshots) goes in [`reports/evidence/`](reports/evidence/).

## 3. Objectives

The assignment asks for a complete traditional machine-learning pipeline on text (EDA, preprocessing, feature extraction, training, evaluation), at least one traditional and one deep-learning feature extractor whose outputs are saved as `.npy` or `.h5`, a pipeline whose preprocessing (stop-word removal, tokenization, padding), embedding and classifier can be configured, and a comparison of traditional and modern approaches. This repository implements all of that, plus the bonus deep-learning pipeline:

* **Three datasets that stress different abilities** (all public Hugging Face links, no Google Drive):

  | Dataset | Classes | Train / test | Regime |
  | --- | --- | --- | --- |
  | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | 4 news topics | 10,000 / 2,000 (stratified sample) | lexical: topic keywords separate the classes |
  | [`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | 2 (sentiment) | 8,530 / 1,066 (official) | compositional: short critic sentences with negation and irony |
  | [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | 77 intents | 9,993 / 3,076 (official) | high-cardinality: fine-grained intents sharing the same nouns |

* **Boolean preprocessing switches** `PreprocessConfig(stop_word_removal, tokenization, padding, negation_marking)`, each ablatable one at a time.
* **Five feature extractors, cached as `.npy` and `.h5`**: BoW and sublinear TF-IDF (sparse + SVD-30), Word2Vec skip-gram mean-pool (300-d + PCA-30, KV cached for DL), frozen DistilBERT (768-d) and BGE-small (384-d) with PCA-30.
* **Classifiers**: Naive Bayes (Multinomial / Gaussian), Logistic Regression, calibrated LinearSVC, and **XGBoost**.
* **Bonus deep learning (flexible 3×3)**: BiLSTM, TextCNN, and SetFit × Word2Vec / DistilBERT / BGE-small (all nine combinations runnable; SetFit×Word2Vec and SetFit×DistilBERT use SentenceTransformer adapters).
* **Metrics**: accuracy, macro precision / recall / F1, multiclass Brier score, expected calibration error, training and inference time.

Headline results: [`reports/master_comparison.csv`](reports/master_comparison.csv) (full matrix) and [`reports/master_comparison_cherry.csv`](reports/master_comparison_cherry.csv) (report tables). PDF: [`reports/ML_Assignment_Report.pdf`](reports/ML_Assignment_Report.pdf).

## 4. Project structure

Submission layout required by the course assignment (Section 5):

```
notebooks/                          Colab front end
modules/                            supporting Python package
reports/                            PDF report, CSV tables, figures, evidence/
features/                           extracted embeddings (.npy / .npz / .h5)
```

Expanded tree:

```
notebooks/
  text_classification_colab.ipynb   Colab front end (Runtime -> Run all)
  build_colab_notebook.py           regenerates the notebook from modules/
modules/
  config.py, data_loader.py, eda.py, feature_extraction.py,
  classifiers.py, deep_learning.py, evaluation.py, word2vec_utils.py, utils.py
features/<dataset>/<embedding>_<tag>/
  X_train.npy / X_test.npy          dense embeddings
  X_train_sparse.npz / X_test_sparse.npz   sparse BoW / TF-IDF
  X_*_pca30.npy, y_*.npy, features.h5, meta.json
reports/
  ML_Assignment_Report.pdf / .md
  master_comparison.csv, master_comparison_cherry.csv, master_ablation.csv, ...
  figures/pipeline_overview.png, figures/master_f1_heatmap.png, figures/<dataset>/*.png
  evidence/
scripts/                            optional helpers (cherry-pick plots, pipeline figure)
docs/                               course assignment PDF / planning notes (not part of the zip checklist)
main.py                             command-line runner (same code path as the notebook)
requirements.txt
```

`<tag>` encodes the preprocessing switches, e.g. `sw0_tok1_pad1_neg0` = stop words kept, subword/bigram tokenization on, padding with masking on, negation marking off (the default configuration).

## 5. How to run

### Option A: Google Colab (recommended for grading)

1. Open [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb) in Colab ([Colab link](https://colab.research.google.com/github/NickCT1612yuri/text-classification-ml-assignment/blob/main/notebooks/text_classification_colab.ipynb)) and select a GPU runtime (T4).
2. Click **Runtime -> Run all**. The notebook
   * installs only the missing libraries (`h5py`, `datasets`, `sentence-transformers`, `setfit`, `xgboost`, ...),
   * uses the repository's `modules/` if present, otherwise clones `REPO_URL` (set it in the second cell) or writes the module files embedded in the notebook,
   * downloads the three datasets from the public Hugging Face Hub (no Google Drive mount),
   * writes the features to `features/`, the tables and figures to `reports/`.

Pretrained weights are downloaded automatically from Hugging Face: `distilbert-base-uncased` and `BAAI/bge-small-en-v1.5`. Word2Vec is trained per dataset on the train split (or loaded from `WORD2VEC_KV_PATH` if set).

### Option B: local

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --quick               # smoke test on small subsets (writes to .quick/)
python main.py --dataset all         # full benchmark (same code as the notebook)
python main.py --dataset banking77 --skip-setfit --skip-textcnn

# or execute the notebook itself
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    notebooks/text_classification_colab.ipynb
```

If you change anything in `modules/`, run `python notebooks/build_colab_notebook.py` so the notebook embeds the new sources.

Full runtime depends on the 3×3 DL grid (especially DistilBERT/BGE → BiLSTM/TextCNN and SetFit). Extracted features and Word2Vec KVs are cached, so a re-run only retrains heads where needed.

On Python 3.13+ there may be no pre-built `h5py` wheel; install the HDF5 library first (`brew install hdf5`, then `HDF5_DIR=$(brew --prefix hdf5) pip install h5py`).

## 6. Deliverables

* Colab notebook: [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/NickCT1612yuri/text-classification-ml-assignment/blob/main/notebooks/text_classification_colab.ipynb))
* PDF report: [`reports/ML_Assignment_Report.pdf`](reports/ML_Assignment_Report.pdf)
* Extracted features (`.npy`, `.npz`, `.h5`): regenerate into [`features/`](features/) with `python main.py --dataset all` or Colab Run all (binaries are gitignored; see [`features/README.md`](features/README.md)). For the course zip, attach a packed `features/` archive if required.
* GitHub repository: [https://github.com/NickCT1612yuri/text-classification-ml-assignment](https://github.com/NickCT1612yuri/text-classification-ml-assignment)
