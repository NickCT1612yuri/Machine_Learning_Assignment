# Configurable Text Classification Pipeline on Banking77

## 1. Course and academic information

| | |
| --- | --- |
| University | Ho Chi Minh City University of Technology (HCMUT), VNU-HCM |
| Faculty | Faculty of Computer Science and Engineering |
| Course | Machine Learning |
| Course code | [Course code, e.g. CO3117] |
| Semester / academic year | Semester I, 2026-2027 |
| Track | Task 2 - Machine Learning with Text Data (+ bonus Word2Vec + BiLSTM) |
| Instructor | [Instructor full name and title] ([instructor email]) |

## 2. Group members and work distribution

Individual grade = group grade × contribution percentage (assignment Section 6).

| Full name | Student ID | Email | Main responsibilities | Contribution |
| --- | --- | --- | --- | --- |
| [Member 1 full name] | [Student ID] | [member1@hcmut.edu.vn] | Data loading and boolean preprocessing (`data_loader.py`, `config.py`), EDA (`eda.py`), TF-IDF, Naive Bayes / LogReg / LinearSVC, preprocessing ablation, README | 50% |
| [Member 2 full name] | [Student ID] | [member2@hcmut.edu.vn] | Word2Vec / DistilBERT and `.npy` / `.h5` persistence (`feature_extraction.py`), bonus Word2Vec + BiLSTM (`deep_learning.py`), evaluation figures (`evaluation.py`), PDF report | 50% |

Collaboration evidence: [`reports/evidence/`](reports/evidence/).

## 3. Objectives

Task 2 requires a complete traditional ML pipeline on text (EDA, preprocessing, feature extraction, training, evaluation), at least one traditional and one modern feature extractor with embeddings saved as `.npy` or `.h5`, a configurable pipeline (stop-word removal, tokenization, padding; embedding; classifier), and a comparison of traditional vs modern approaches. This repository implements that on **Banking77**, plus a bonus Word2Vec + BiLSTM path.

* **Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) — 77 online-banking intents, 9,993 / 3,076 train / test (public Hugging Face; no Google Drive).
* **Boolean preprocessing:** `PreprocessConfig(stop_word_removal, tokenization, padding, negation_marking)`.
* **Features (cached as `.npy` / `.h5`):** TF-IDF (uni+bigram), Word2Vec skip-gram mean-pool (300-d), frozen DistilBERT (768-d). BoW remains available as a plug-in embedding type.
* **Classifiers:** Naive Bayes, Logistic Regression, calibrated LinearSVC.
* **Bonus DL:** Word2Vec + BiLSTM.
* **Metrics:** accuracy, macro precision / recall / F1, multiclass Brier score, ECE, train / infer time.

Headline results: [`reports/master_comparison.csv`](reports/master_comparison.csv). PDF: [`reports/report/main.pdf`](reports/report/main.pdf).

## 4. Project structure

```
notebooks/
  text_classification_colab.ipynb   Colab front end (Runtime -> Run all)
  build_colab_notebook.py           regenerates the notebook from modules/
modules/
  config.py, data_loader.py, eda.py, feature_extraction.py,
  classifiers.py, deep_learning.py, evaluation.py, word2vec_utils.py, utils.py
features/banking77/<embedding>_<tag>/
  X_train.npy / X_test.npy | X_*_sparse.npz, features.h5, meta.json, ...
reports/
  comparison_banking77.csv, figures/, evidence/
  report/                           LaTeX source (HCMUT template) + main.pdf
main.py
requirements.txt
```

`<tag>` encodes preprocessing switches, e.g. `sw0_tok1_pad1_neg0` (default).

## 5. How to run

### Option A: Google Colab

1. Open [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb) in Colab and select a GPU runtime (T4).
2. Click **Runtime -> Run all**. The notebook installs missing libraries, recreates `modules/` if needed, downloads Banking77 from Hugging Face, and writes `features/` and `reports/`.

### Option B: local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --quick          # smoke test (writes to .quick/)
python main.py                  # full Banking77 benchmark
python main.py --skip-bilstm    # classical pipeline only
```

After editing `modules/`, regenerate the notebook:

```bash
python notebooks/build_colab_notebook.py
```

### Rebuild the PDF report (LaTeX)

```bash
cd reports/report
make          # XeLaTeX, two passes -> main.pdf
```

## 6. Deliverables

* Colab notebook: [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb)
* PDF report (LaTeX): [`reports/report/main.pdf`](reports/report/main.pdf) — sources in [`reports/report/`](reports/report/)
* Extracted features (`.npy`, `.npz`, `.h5`): regenerate with `python main.py` (binaries gitignored; see [`features/README.md`](features/README.md))
* GitHub: share the repository link in the notebook and report for bonus credit
