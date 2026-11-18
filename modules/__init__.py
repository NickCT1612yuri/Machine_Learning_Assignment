"""
HCMUT Machine Learning assignment — Task 2 (text classification).

Runtime entrypoint: ``modules.assignment_pipeline`` (invoked by ``main.py``).

Legacy multi-module suite (Word2Vec / DistilBERT / BiLSTM) is archived under
``modules/_legacy/`` and is **not** used by the submission pipeline.
"""

from modules.assignment_pipeline import REGISTRY, run_dataset, make_report

__all__ = ["REGISTRY", "run_dataset", "make_report"]
