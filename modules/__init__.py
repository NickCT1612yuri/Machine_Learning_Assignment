"""Text classification pipeline for the HCMUT Machine Learning assignment (Task 2)."""
from .config import DATASETS
from .pipeline import FAMILIES, run_dataset

__all__ = ["DATASETS", "FAMILIES", "run_dataset"]
