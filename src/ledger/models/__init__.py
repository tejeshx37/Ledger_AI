"""Detection models: a common :class:`~ledger.models.base.Detector`
interface, implementations, and a name -> class registry.
"""

from ledger.models.base import Detector
from ledger.models.registry import MODEL_REGISTRY, get_detector
from ledger.models.xgboost_baseline import DetectorNotFittedError, XGBoostBaselineDetector

__all__ = [
    "Detector",
    "DetectorNotFittedError",
    "XGBoostBaselineDetector",
    "MODEL_REGISTRY",
    "get_detector",
]
