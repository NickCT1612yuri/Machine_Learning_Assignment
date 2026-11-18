"""All experiment settings in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VAL_FRACTION = 0.15  # stratified share of the official training split used for validation

# Each corpus: Hugging Face id, optional stratified caps on train/test, and the
# token budget used by BGE and TextCNN (chosen above the 95th-percentile length).
DATASETS = {
    "ag_news": dict(hf_id="fancyzhx/ag_news", train_cap=10_000, test_cap=2_000, max_len=96),
    "rotten_tomatoes": dict(hf_id="cornell-movie-review-data/rotten_tomatoes", train_cap=None, test_cap=None, max_len=64),
    "banking77": dict(hf_id="mteb/banking77", train_cap=None, test_cap=None, max_len=64),
}
QUICK_CAPS = (1_500, 600)  # train/test caps for the --quick smoke test

# Traditional pipeline: every preprocessing combination x every classifier setting.
TFIDF_GRID = dict(tokenizer=("regex", "whitespace"), stop_words=(False, True), max_ngram=(1, 2))
TFIDF_PARAMS = dict(min_df=2, max_features=25_000, sublinear_tf=True)
CLASSIFIER_GRID = {
    "nb": (0.01, 0.03, 0.1, 0.3, 1, 3),  # Multinomial Naive Bayes smoothing alpha
    "logreg": (1, 4, 16, 64),      # inverse regularisation strength C
    "svm": (0.1, 0.3, 1, 3),       # LinearSVC C
}

# Modern pipeline: frozen sentence encoder + logistic regression.
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

# Deep-learning pipeline (bonus): TextCNN trained end-to-end.
TEXTCNN = dict(vocab_size=20_000, min_count=2, emb_dim=100, n_filters=64, widths=(3, 4, 5),
               dropout=0.5, lr=1e-3, max_epochs=20, patience=3)

N_BOOTSTRAP = 1_000  # resamples for test-set confidence intervals
