"""
Configurable text-classification pipeline (HCMUT Machine Learning assignment, Task 2).

    config              PreprocessConfig, ExperimentConfig, dataset registry, hyperparameters
    data_loader         Hugging Face download + boolean preprocessing
    eda                 word-count statistics, length distribution, word frequencies
    feature_extraction  TF-IDF / Word2Vec / DistilBERT (.npy + .h5); BoW available as plug-in
    classifiers         Naive Bayes, Logistic Regression, calibrated LinearSVC
    deep_learning       Word2Vec + BiLSTM (bonus)
    evaluation          ablation + comparison suites, CSV tables, figures

Heavy libraries (torch, transformers) are imported lazily inside the functions that need them.
"""

from modules.config import (
    DATASET_ORDER,
    DATASET_REGISTRY,
    FEATURE_DIR,
    FIG_DIR,
    REPORT_DIR,
    SEED,
    ExperimentConfig,
    PreprocessConfig,
)
from modules.data_loader import clean_text, load_and_preprocess_dataset, load_raw_dataset
from modules.eda import run_eda
from modules.feature_extraction import extract_and_save_features, list_feature_files, load_h5_features
from modules.classifiers import compute_metrics, train_and_evaluate_classifier
from modules.deep_learning import run_bilstm_pipeline, run_dl_pipeline
from modules.evaluation import (
    build_master_table,
    error_analysis,
    negation_analysis,
    plot_suite_results,
    run_all_eda,
    run_dataset_benchmark_suite,
    run_deep_learning,
    run_feature_extraction,
    run_main_comparison,
    run_preprocessing_ablation,
)
from modules.utils import get_device, seed_everything

__all__ = [
    "DATASET_ORDER", "DATASET_REGISTRY", "FEATURE_DIR", "FIG_DIR", "REPORT_DIR", "SEED",
    "ExperimentConfig", "PreprocessConfig",
    "clean_text", "load_and_preprocess_dataset", "load_raw_dataset",
    "run_eda",
    "extract_and_save_features", "list_feature_files", "load_h5_features",
    "compute_metrics", "train_and_evaluate_classifier",
    "run_bilstm_pipeline", "run_dl_pipeline",
    "build_master_table", "error_analysis", "negation_analysis", "plot_suite_results", "run_all_eda",
    "run_dataset_benchmark_suite", "run_deep_learning", "run_feature_extraction", "run_main_comparison",
    "run_preprocessing_ablation",
    "get_device", "seed_everything",
]
