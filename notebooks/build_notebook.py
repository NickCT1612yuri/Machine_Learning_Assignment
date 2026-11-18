"""Regenerate text_classification_colab.ipynb with the current modules/ embedded.

Run after editing anything in modules/:  python notebooks/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sources = {p.name: p.read_text() for p in sorted((ROOT / "modules").glob("*.py"))}


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


cells = [
    cell("markdown", """
# Text Classification — Machine Learning Assignment (Task 2)

HCMUT · Semester I, 2026–2027. **Runtime → Run all** reproduces every number in the report (about 10–15 min on a T4 GPU).

Three pipelines are compared on three public corpora:

| Pipeline | Features | Classifier |
| --- | --- | --- |
| Traditional | TF-IDF (saved as `.h5`) | Naive Bayes / Logistic Regression / Linear SVM |
| Modern features | frozen BGE-small sentence embeddings (saved as `.npy`) | Logistic Regression |
| Deep learning (bonus) | embeddings learned end-to-end | TextCNN |

Data are downloaded from Hugging Face: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news),
[Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes),
[Banking77](https://huggingface.co/datasets/mteb/banking77); encoder [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5).
No Google Drive is mounted.
"""),
    cell("markdown", "## 1. Install libraries\nColab already ships NumPy, pandas, scikit-learn, PyTorch and matplotlib; this adds the rest."),
    cell("code", """
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets>=2.19", "transformers>=4.40", "h5py>=3.10"])
"""),
    cell("markdown", "## 2. Write the project modules\nThe cell below is generated from `modules/` in the repository (`notebooks/build_notebook.py`); do not edit it by hand."),
    cell("code", f"""
from pathlib import Path
SOURCES = {json.dumps(sources, indent=1)}
Path("modules").mkdir(exist_ok=True)
for name, code in SOURCES.items():
    (Path("modules") / name).write_text(code)
print("Wrote", ", ".join(SOURCES))
"""),
    cell("markdown", """
## 3. Configure
`QUICK = True` runs a two-minute smoke test on small subsets (results go to `.smoke/` and are **not** the reported results).
Other options: `FAMILIES` ⊆ {tfidf, bge, textcnn}, `CLASSIFIERS` ⊆ {nb, logreg, svm}, `PADDING` ∈ {dynamic, fixed}.
Grids (tokenizer, stop words, n-grams, C, alpha, TextCNN settings) live in `modules/config.py`.
"""),
    cell("code", """
QUICK = False
DATASETS = ["ag_news", "rotten_tomatoes", "banking77"]
FAMILIES = ["tfidf", "bge", "textcnn"]
CLASSIFIERS = ["nb", "logreg", "svm"]
PADDING = "dynamic"
"""),
    cell("markdown", "## 4. Run EDA, feature extraction, model selection and evaluation"),
    cell("code", """
import pandas as pd, torch
from modules import run_dataset
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
OUT = Path(".smoke" if QUICK else ".").resolve()
summary = pd.concat([run_dataset(d, OUT, QUICK, FAMILIES, CLASSIFIERS, PADDING).assign(dataset=d) for d in DATASETS])
summary.to_csv(OUT / "reports/results/summary.csv", index=False)
"""),
    cell("markdown", "## 5. Results\nHeld-out test scores (macro-F1 with a 95% bootstrap interval), the EDA figure and the five best validation configurations per corpus."),
    cell("code", """
from IPython.display import Image, display
cols = ["dataset", "family", "accuracy", "f1_macro", "f1_ci_low", "f1_ci_high", "brier", "ece"]
display(summary[cols].round(4))
for d in DATASETS:
    res = OUT / "reports/results" / d
    print("\\n" + d)
    display(Image(str(res / "eda.png")))
    display(pd.read_csv(res / "validation.csv").nlargest(5, "f1_macro").round(4))
"""),
    cell("markdown", """
## 6. Output files
* `features/<dataset>/` — `tfidf.h5` (sparse TF-IDF, fit/val/test), `bge_{fit,val,test}.npy` (384-d embeddings), `y_*.npy`.
* `reports/results/<dataset>/` — `eda.png|json`, `validation.csv` (all tested configurations), `test_results.csv`,
  `paired_bootstrap.csv`, `per_class_f1.csv`, `top_confusions.csv`, `negation.csv`, `predictions.csv`, `textcnn_history.csv`, `run.json`.

Files in a Colab runtime are temporary; download them before disconnecting.
"""),
]

notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}
(ROOT / "notebooks" / "text_classification_colab.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print("notebook written")
