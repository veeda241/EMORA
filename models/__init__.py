"""Model comparison grid and training-plan generators for the MELD stress/burnout tracker."""

from .comparison_grid import compare_table, DATASETS, MODEL_REGISTRY, MODALITY_ORDER
from .training_plan import PROJECT_DIM, training_plan

__all__ = [
    "compare_table",
    "DATASETS",
    "MODEL_REGISTRY",
    "MODALITY_ORDER",
    "PROJECT_DIM",
    "training_plan",
]

__version__ = "0.1.0"
