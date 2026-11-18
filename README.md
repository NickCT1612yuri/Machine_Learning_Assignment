# Text Classification — Machine Learning Assignment

HCMUT, Faculty of Computer Science and Engineering · Semester I, 2026–2027 · Task 2: text data.

| Member | Student ID | Email | Actual work | Contribution |
| --- | --- | --- | --- | --- |
| Dương Minh Nhật | 2452889 | [email] | Pipeline orchestration, classical TF-IDF path, EDA/report sections, README | 50% |
| Nguyễn Trọng Nhân | 2452883 | [email] | `assignment_pipeline.py`, BGE + TextCNN, Colab notebook, evaluation outputs | 50% |

**Still to fill before submission:** course code, instructor name/email, member emails, evidence of collaboration. Percentages must total 100%. Confirm datasets with the instructor (no duplicate groups).

## Design

We compare (1) TF-IDF and conventional classifiers, (2) frozen pretrained BGE-small embeddings with logistic regression, and (3) a TextCNN trained end-to-end from random word embeddings. The Transformer is a fixed feature extractor (not fine-tuned). TextCNN is the optional deep-learning pipeline. Metrics include accuracy, macro precision/recall/F1, plus Brier score / ECE and a negation vs non-negation accuracy split for academic analysis.

Public sources: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes), [Banking77](https://huggingface.co/datasets/mteb/banking77), and [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5).

The original training split is stratified into 85% fit and 15% validation (seed 42); the official test split is held out. Exact duplicate leakage across fit/val/test is filtered and logged. AG News is sampled to 10,000 train / 2,000 test; the other datasets use official splits. Validation Macro-F1 selects TF-IDF stop-word / n-gram / classifier settings, BGE logistic `C`, and TextCNN early-stopping epoch. Each selected family is evaluated on test once.

CLI knobs: `--embeddings tfidf bge`, `--classifiers nb logreg svm`, `--tokenization regex|whitespace`, `--padding dynamic|fixed`. Implementation: [`modules/assignment_pipeline.py`](modules/assignment_pipeline.py) (sole runtime entrypoint via [`main.py`](main.py)).

## Run in Google Colab

Upload [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb), prefer a T4 GPU, run cells in order. Default smoke: `QUICK=True` on Rotten Tomatoes; then set `QUICK=False` for full datasets. No personal Drive mount. After editing `modules/assignment_pipeline.py`, regenerate with `python notebooks/build_colab_notebook.py`.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest tests_pipeline -v
python main.py --quick              # smoke test -> outputs/quick
python main.py                      # full run -> outputs/full
python main.py --dataset banking77 --padding fixed --tokenization regex
```

First run needs Hugging Face network access (or a prepared `HF_HOME` cache).

## Output files

- `outputs/full/reports/<dataset>/validation.csv`, `test_results.csv`, classification reports, confusion matrices, errors, negation tables
- `outputs/full/features/<dataset>/tfidf.h5`, `bge_{fit,val,test}.npy`, `y_*.npy`
- Feature binaries are gitignored; include them or a download link in the course zip

## Academic report (LaTeX)

HCMUT-formatted report sources live in [`reports/report/`](reports/report/). Rebuild:

```bash
cd reports/report && make
```

PDF: [`reports/report/main.pdf`](reports/report/main.pdf). Numbers in the report must come from the **current** `outputs/` pipeline. Older Banking77-only CSVs/PDFs under `reports/` (if any) are historical and must not be presented as results of this pipeline.

Collaboration evidence: [`reports/evidence/`](reports/evidence/).

## Reproducibility and limits

Fitting and vocabulary use fit data only. No configuration selection uses test labels. Seed, settings and split sizes are stored with results. GPU nondeterminism and library/dataset version drift are possible; preserve the environment and generated artifacts with the submission.
