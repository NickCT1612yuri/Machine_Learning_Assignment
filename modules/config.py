"""
Pipeline configuration: dataset, boolean preprocessing, experiment knobs.
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
    "banking77": {
        "title": "Banking77",
        "hf_id": "mteb/banking77",
        "max_train": None,
        "max_test": None,
        "max_len_cap": 64,
    },
}

DATASET_ORDER = ["banking77"]

# Sparse lexical features
MAX_FEATURES = 25000

# Dense projection for PCA/SVD-30 (saved with features; not used in the default grid)
PCA_DIM = 30

# Frozen Transformer encoder
BERT_MODEL = "distilbert-base-uncased"
BERT_BATCH_SIZE = 32

# Word2Vec (train on the train split; optional KEYEDVECTORS override)
WORD2VEC = {
    "vector_size": 300,
    "window": 5,
    "min_count": 2,
    "epochs": 5,
    "sg": 1,  # skip-gram
    "workers": 4,
    "pretrained_env": "WORD2VEC_KV_PATH",
}

# Bonus sequence model
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

DL_EMBEDDINGS = ("word2vec",)
DL_MODELS = ("bilstm",)


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
    dataset_name: str = "banking77"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding_type: str = "tfidf"  # bow | tfidf | word2vec | distilbert
    classifier_type: str = "logreg"  # naive_bayes | logreg | svm
    dl_model: str = "bilstm"
    dl_embedding: str = "word2vec"
    save_format: str = "both"  # npy | h5 | both


DEFAULT_PREPROCESS = PreprocessConfig(
    stop_word_removal=False,
    tokenization=True,
    padding=True,
    negation_marking=False,
)
