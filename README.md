# Text Classification — Machine Learning Assignment (Task 2)

| | |
| --- | --- |
| Course | Machine Learning — [course code] |
| Semester | Semester I, academic year 2026–2027 |
| University | Ho Chi Minh City University of Technology (HCMUT), VNU-HCM — Faculty of Computer Science and Engineering |
| Instructor | [instructor name, email] |
| Report | [`reports/report/main.pdf`](reports/report/main.pdf) |
| Colab notebook | [`notebooks/text_classification_colab.ipynb`](notebooks/text_classification_colab.ipynb) — [Open in Colab](https://colab.research.google.com/) |

| Member | Student ID | Email | Contribution |
| --- | --- | --- | --- |
| Dương Minh Nhật | 2452889 | [email] | 50% |
| Nguyễn Trọng Nhân | 2452883 | [email] | 50% |

## Objective

Build and compare complete text-classification pipelines on three public corpora with different structure:
AG News (4 news topics), Rotten Tomatoes (binary sentiment) and Banking77 (77 customer-service intents).

| Pipeline | Representation | Classifier |
| --- | --- | --- |
| Traditional (required) | TF-IDF, saved as `.h5` | Multinomial NB, Logistic Regression, Linear SVM |
| Modern features (required) | frozen BGE-small sentence embeddings, saved as `.npy` | Logistic Regression |
| Deep learning (bonus) | embeddings learned end-to-end | TextCNN |

Each pipeline covers EDA, preprocessing, feature extraction, training and evaluation. Configurations are chosen on a validation split
(macro-F1); each chosen model is evaluated once on the official test split.

## Run

**Colab:** open the notebook and choose *Runtime → Run all* (GPU recommended, ~10–15 min). It installs the missing libraries,
downloads the data from Hugging Face and needs no Google Drive.

**Locally** (Python ≥ 3.10):

```bash
pip install -r requirements.txt
python -m unittest -v                        # fast checks, no downloads
python main.py --quick                       # smoke test on small subsets -> .smoke/
python main.py                               # full experiment -> features/ and reports/results/
python main.py --datasets banking77 --families tfidf bge --classifiers logreg svm --padding fixed
cd reports/report && make                    # rebuild tables, figures and main.pdf (needs XeLaTeX)
```

Datasets (downloaded automatically): [AG News](https://huggingface.co/datasets/fancyzhx/ag_news),
[Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes),
[Banking77](https://huggingface.co/datasets/mteb/banking77). Encoder: [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5).

## Structure

```
main.py                 command-line entry point
modules/
  config.py             datasets, search grids, TextCNN settings (all tunable values)
  data.py               loading, cleaning, splitting, leakage filter, EDA
  features.py           TF-IDF (.h5) and BGE-small (.npy) extraction and storage
  models.py             NB / LogReg / SVM factory and TextCNN
  evaluation.py         metrics, calibration, bootstrap, error analysis
  pipeline.py           runs one corpus end to end and writes the results
notebooks/              Colab notebook and the script that regenerates it from modules/
features/               extracted features (generated; not tracked by git)
reports/
  results/<corpus>/     EDA, validation grid, test metrics, errors (generated)
  report/               LaTeX source of the report and main.pdf
  evidence/             collaboration evidence
tests/                  unit tests
docs/                   assignment brief and study notes
```

After editing `modules/`, run `python notebooks/build_notebook.py` so the notebook embeds the same code.

## Reproducibility

Seed 42 for splits, sampling, bootstrap and TextCNN. Library versions and settings of each run are stored in
`reports/results/<corpus>/run.json`. GPU/MPS kernels can introduce small numerical differences between machines.
