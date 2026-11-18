"""Build the self-contained Colab notebook from the current pipeline code."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'modules' / 'assignment_pipeline.py').read_text(encoding='utf-8')

def cell(kind, content):
    result = {'cell_type': kind, 'metadata': {}, 'source': content.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

cells = [
    cell('markdown', '''# Machine Learning Assignment - Text Classification

This notebook focuses on the code and explanation of the methods. It downloads public datasets directly and does not require mounting Google Drive. It does not create a report or a ZIP archive.

## Methods

1. **TF-IDF + traditional classifier:** learn vocabulary and TF-IDF weights from the fit split. Validation compares stop-word removal on/off, unigrams versus unigrams+bigrams, Multinomial Naive Bayes, Logistic Regression and linear SVM. Naive Bayes uses `alpha=1`; Logistic Regression and SVM try `C=1` and `C=4`.
2. **BGE-small + Logistic Regression:** use a pretrained, frozen Transformer to extract CLS embeddings and L2-normalize them. Validation chooses Logistic Regression `C` from `0.5`, `2` and `8`. The Transformer is not fine-tuned.
3. **TextCNN:** train word embeddings and convolutional classifier end-to-end from scratch using cross-entropy and Adam. Filters cover 3, 4 and 5 tokens; dropout is 0.5. The best epoch is selected by validation Macro-F1, with early stopping.

The original training split is divided into 85% fit and 15% validation; the original test set is kept for final evaluation. AG News is sampled to 10,000 train and 2,000 test instances; the other two use their official train/test splits. Exact duplicate text across splits is filtered out of the development sets and counted; source data and the test set are not modified. EDA covers class counts, text lengths and frequent words.

Public data: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes), [Banking77](https://huggingface.co/datasets/mteb/banking77). Model: [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5).
'''),
    cell('markdown', '''## Step 1 - Install dependencies

Run the code cell below. Internet access is needed to install dependencies and download the datasets/model. A GPU runtime is recommended, but CPU is supported.'''),
    cell('code', '''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'numpy>=1.26', 'pandas>=2.1', 'matplotlib>=3.8', 'scipy>=1.12',
    'scikit-learn>=1.4', 'h5py>=3.10', 'datasets>=2.19',
    'transformers>=4.40', 'torch>=2.2'])
print('Dependency installation completed.')
'''),
    cell('markdown', '''## Step 2 - Load the pipeline

The current Python code is embedded in this notebook, so a fresh Colab session does not depend on an outdated GitHub checkout. Do not modify this cell manually.'''),
    cell('code', f'''from pathlib import Path
import sys
root = Path.cwd()
code_dir = root / '_notebook_code'
code_dir.mkdir(exist_ok=True)
(code_dir / 'assignment_pipeline.py').write_text({source!r}, encoding='utf-8')
sys.path.insert(0, str(code_dir))
from assignment_pipeline import REGISTRY, run_dataset
import torch
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
'''),
    cell('markdown', '''## Step 3 - Configure the experiment

The initial settings run a **quick smoke test on Rotten Tomatoes**. These results use a smaller sample and are only for checking whether the notebook works. Once that succeeds, set `QUICK = False` and select the datasets required by the group.

- TF-IDF varies stop-word removal, n-grams and the selected classifiers; `TOKENIZATION` controls its tokenizer.
- Both padding options use attention masks and the same truncation length for BGE.
- Keep both TF-IDF and BGE in `EMBEDDINGS` for the assignment's traditional and modern features.
'''),
    cell('code', '''QUICK = True
DATASETS = ['rotten_tomatoes']  # options: ag_news, rotten_tomatoes, banking77
PADDING = 'dynamic'             # dynamic or fixed, for BGE
TOKENIZATION = 'regex'          # regex or whitespace, for TF-IDF
CLASSIFIERS = ['nb', 'logreg', 'svm']
EMBEDDINGS = ['tfidf', 'bge']
BATCH_SIZE = 32
EPOCHS = 7                     # quick mode uses 2 epochs
OUTPUT = root / 'experiment_outputs'
'''),
    cell('markdown', '''## Step 4 - Train and evaluate

The pipeline performs EDA, selects parameters on validation, evaluates the selected configuration from each family on the test set, and saves extracted features. It may take several minutes; a full run can take considerably longer.'''),
    cell('code', '''for name in DATASETS:
    run_dataset(name, OUTPUT, quick=QUICK, epochs=EPOCHS,
                batch_size=BATCH_SIZE, padding=PADDING,
                tokenization=TOKENIZATION, classifiers=CLASSIFIERS,
                embeddings=EMBEDDINGS)
print('Pipeline completed. Output folder:', OUTPUT)
'''),
    cell('markdown', '''## Step 5 - Inspect results

The first table reports held-out test metrics. The second table contains validation configurations. Macro-F1 gives each class equal weight. Inspect EDA and class-level errors before drawing conclusions.'''),
    cell('code', '''import pandas as pd
from IPython.display import display, Image
for name in DATASETS:
    print('Dataset:', name)
    display(pd.read_csv(OUTPUT / 'reports' / name / 'test_results.csv'))
    display(pd.read_csv(OUTPUT / 'reports' / name / 'validation.csv'))
    display(Image(filename=str(OUTPUT / 'reports' / name / 'eda.png')))
print('Features:', OUTPUT / 'features')
print('Metrics and analysis:', OUTPUT / 'reports')
'''),
    cell('markdown', '''## Output files

The Files panel on the left contains `experiment_outputs/features/` with `.h5` and `.npy` feature files and `experiment_outputs/reports/` with metrics, EDA figures and error-analysis CSVs. These files are in the temporary Colab runtime: download any files you need before the runtime is disconnected. The report will be completed separately later.''')
]
nb = {
    'cells': cells,
    'metadata': {
        'colab': {'name': 'text_classification_colab.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}
(root / 'notebooks' / 'text_classification_colab.ipynb').write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8'
)
print('Notebook refreshed')
