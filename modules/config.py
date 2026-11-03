"""
Pipeline configuration: datasets, boolean preprocessing, experiment knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

# Assignment folder layout: notebooks/, modules/, features/, reports/.
# Smoke tests (TC_QUICK=1) write to .quick/ so they never overwrite real results.
ROOT = Path(__file__).resolve().parents[1]
_OUT = ROOT / ".quick" if os.environ.get("TC_QUICK", "0") == "1" else ROOT
FEATURE_DIR = _OUT / "features"
REPORT_DIR = _OUT / "reports"
FIG_DIR = REPORT_DIR / "figures"

DATASET_REGISTRY = {
    "ag_news": {
        "title": "AG News",
        "hf_id": "fancyzhx/ag_news",
        "max_train": 10000,
        "max_test": 2000,
        "max_len_cap": 128,
    },
    "rotten_tomatoes": {
        "title": "Rotten Tomatoes",
        "hf_id": "cornell-movie-review-data/rotten_tomatoes",
        "max_train": None,
        "max_test": None,
        "max_len_cap": 64,
    },
    "banking77": {
        "title": "Banking77",
        "hf_id": "mteb/banking77",
        "max_train": None,
        "max_test": None,
        "max_len_cap": 64,
    },
}

DATASET_ORDER = ["ag_news", "rotten_tomatoes", "banking77"]

# Sparse lexical features
MAX_FEATURES = 25000

# Dense projection for PCA/SVD-30 + LogReg control rows
PCA_DIM = 30

# Frozen Transformer encoders
BERT_MODEL = "distilbert-base-uncased"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BERT_BATCH_SIZE = 32

# Word2Vec (train on each dataset's train split; optional KEYEDVECTORS override)
WORD2VEC = {
    "vector_size": 300,
    "window": 5,
    "min_count": 2,
    "epochs": 5,
    "sg": 1,  # skip-gram
    "workers": 4,
    "pretrained_env": "WORD2VEC_KV_PATH",
}

XGBOOST = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
}

# Sequence models (BiLSTM / TextCNN)
BILSTM = {
    "max_vocab": 20000,
    "embed_dim": 128,  # used only for random-init fallback
    "hidden_dim": 192,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 7,
    "lr": 1e-3,
    "patience": 2,
    "max_len_cap": 200,
}

TEXTCNN = {
    "filter_sizes": (3, 4, 5),
    "num_filters": 100,
    "dropout": 0.5,
    "batch_size": 64,
    "epochs": 7,
    "lr": 1e-3,
    "patience": 2,
    "max_len_cap": 200,
}

# SetFit few-shot: shots / iterations for (<= 10 classes, > 10 classes)
SETFIT = {
    "model_id": BGE_MODEL,
    "shots": (16, 8),
    "num_iterations": (20, 5),
    "batch_size": 16,
    "num_epochs": 1,
}

DL_EMBEDDINGS = ("word2vec", "distilbert", "bge_small")
DL_MODELS = ("bilstm", "textcnn", "setfit")


@dataclass
class PreprocessConfig:
    """Boolean preprocessing switches required by the assignment."""

    stop_word_removal: bool = False
    tokenization: bool = True
    padding: bool = True
    negation_marking: bool = False

    def tag(self) -> str:
        return (
            f"sw{int(self.stop_word_removal)}_"
            f"tok{int(self.tokenization)}_"
            f"pad{int(self.padding)}_"
            f"neg{int(self.negation_marking)}"
        )


@dataclass
class ExperimentConfig:
    dataset_name: str = "rotten_tomatoes"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding_type: str = "bge_small"  # bow | tfidf | word2vec | distilbert | bge_small
    classifier_type: str = "logreg"  # naive_bayes | logreg | svm | xgboost
    dl_model: str = "bilstm"  # bilstm | textcnn | setfit
    dl_embedding: str = "word2vec"  # word2vec | distilbert | bge_small
    save_format: str = "both"  # npy | h5 | both


DEFAULT_PREPROCESS = PreprocessConfig(
    stop_word_removal=False,
    tokenization=True,
    padding=True,
    negation_marking=False,
)
