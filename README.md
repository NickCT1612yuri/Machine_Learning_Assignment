# Text Classification — Machine Learning Assignment

HCMUT, Faculty of Computer Science and Engineering · Semester I, 2026–2027 · Task 2: text data.

**To be completed by the team:** course code, instructor name and email, full names, student IDs, emails, actual responsibilities, contribution percentages, registration of datasets with the instructor, and evidence of collaboration. The percentages must total 100%. Do not submit placeholder details.

| Member | Student ID | Email | Actual work | Contribution |
| --- | --- | --- | --- | --- |
| [Name] | [ID] | [email] | [responsibilities] | [%] |
| [Name] | [ID] | [email] | [responsibilities] | [%] |

## Design

We compare (1) TF-IDF and conventional classifiers, (2) frozen pretrained BGE-small embeddings with logistic regression and (3) a TextCNN trained end-to-end from random word embeddings. The Transformer is a fixed feature extractor; it is not fine-tuned. TextCNN provides the optional deep-learning pipeline.

Public sources: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes), [Banking77](https://huggingface.co/datasets/mteb/banking77), and [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5). Confirm the datasets do not duplicate another group’s choice and are approved by the instructor.

The original training split is stratified into 85% fit and 15% validation (seed 42); the official test split is held out. To avoid exact duplicate leakage, validation records matching test text are excluded, and fit records matching validation or test text are excluded. The original dataset and test data are not changed; counts before filtering and excluded records are logged. AG News is sampled to 10,000 original train and 2,000 test instances for practical resource use; the other datasets use the complete official train/test splits. The official Rotten Tomatoes validation split is not used: validation is instead drawn from its official train split. Validation Macro-F1 selects TF-IDF stop-word setting, unigrams/bigrams, classifier and hyperparameter; BGE logistic-regression `C`; and TextCNN early-stopping epoch. Each selected family is evaluated on test once. Comparisons report accuracy and macro precision, recall and F1. Scores from one split/seed are estimates, not uncertainty intervals.

The default experiment uses both features and all three TF-IDF classifiers; `--embeddings tfidf bge` and `--classifiers nb logreg svm` can narrow the search for local checks. Keep both embeddings in the submission run. Preprocessing options: `--tokenization regex|whitespace` configures TF-IDF; the pretrained encoder always uses its own tokenizer. TF-IDF validates stop-word removal on/off. `--padding dynamic|fixed` controls Transformer batching; both options retain attention masks and identical truncation. TextCNN uses padded word sequences and a train-only vocabulary. Max lengths are explicit in `modules/assignment_pipeline.py`. The feature extractor, classifier settings and other options are deliberately modest, rather than a broad fragile model grid.

## Run in Google Colab

Open https://colab.research.google.com/, select File > Upload notebook, and choose the **local** file `notebooks/text_classification_colab.ipynb`. Alternatively, open the notebook from the feature branch on GitHub. Choose a T4 GPU if available and run each cell in order. The default configuration uses `QUICK=True` and Rotten Tomatoes for a smoke test. Once it runs successfully, set `QUICK=False` and select the desired datasets.

The notebook installs dependencies, downloads public data and the BGE model, and runs without a personal Drive. The output is in `experiment_outputs/features/` and `experiment_outputs/reports/` in the temporary runtime. These are feature files, EDA and evaluation results; no report or ZIP is generated. Download files you need before the runtime disconnects. After modifying `modules/assignment_pipeline.py`, regenerate the notebook with `python notebooks/build_colab_notebook.py` before uploading it.

## Run locally

```powershell
cd D:\baitaplonML\Machine_Learning_Assignment
python -m pip install -r requirements.txt
python -m unittest tests_pipeline -v
python main.py --quick              # smoke test, outputs/quick
python main.py                      # full configured datasets, outputs/full
python main.py --dataset banking77 --padding fixed --tokenization regex
```

First run needs internet access to Hugging Face. If files have already been downloaded to a chosen Hugging Face cache, set `HF_HOME` accordingly; optionally set `HF_HUB_OFFLINE=1` to avoid network access. On this machine a previously downloaded cache is at `D:\baitaplonML\hf_cache`. No test outputs are overwritten by quick runs.

## Output files

- `outputs/full/reports/<dataset>/validation.csv`: tested configurations and validation metrics.
- `outputs/full/reports/<dataset>/test_results.csv`: one held-out test result per selected family. `*_classification_report.csv`, confusion matrices, and `*_errors.csv` provide per-class/error analysis.
- `outputs/full/features/<dataset>/tfidf.h5`: fit/validation/test sparse TF-IDF matrices and labels. Classifier is fitted using the read-back matrices.
- `outputs/full/features/<dataset>/bge_{fit,val,test}.npy`: frozen embeddings read back as inputs to the downstream classifier, plus `y_*.npy` labels.
- `notebooks/` and `modules/`: notebook and source code. Output feature binaries are intentionally not committed to Git; include them or an accessible download link in the final submission.

The report is reserved for a later stage; existing reports are preserved. Legacy modules, figures and `reports/ML_Assignment_Report.pdf` are retained for historical comparison. They were produced by the earlier workflow and **must not be presented as results of the new pipeline**. The new workflow does not use legacy Word2Vec, XGBoost or SetFit.

## Reproducibility and limits

Fitting and vocabulary construction use fit data only. No configuration selection uses test labels. The code stores seed, settings and split sizes alongside results. GPU operations can have small nondeterministic differences. Dataset/model versions and library releases can change; preserve the exact execution environment, generated feature files and result tables with the submission. No model performance claim is guaranteed without executing the full notebook.
