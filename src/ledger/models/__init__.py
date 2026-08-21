"""Detection models: a common :class:`~ledger.models.base.Detector`
interface, implementations, and a name -> class registry.
"""

from ledger.models.base import Detector
from ledger.models.graph_common import GraphDetector, GraphDetectorNotFittedError
from ledger.models.graphsage import GraphSAGEDetector
from ledger.models.registry import MODEL_REGISTRY, get_detector
from ledger.models.tgat import TGATDetector
from ledger.models.xgboost_baseline import DetectorNotFittedError, XGBoostBaselineDetector

__all__ = [
    "Detector",
    "DetectorNotFittedError",
    "XGBoostBaselineDetector",
    "GraphDetector",
    "GraphDetectorNotFittedError",
    "GraphSAGEDetector",
    "TGATDetector",
    "MODEL_REGISTRY",
    "get_detector",
]
