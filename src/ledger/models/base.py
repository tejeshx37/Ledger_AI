"""Abstract detector interface: every model LEDGER trains implements this,
so the evaluation harness (:mod:`ledger.evaluation`) and the training
orchestration (:mod:`ledger.training`) are model-agnostic. The baseline
(Phase 3), GraphSAGE, and TGAT (Phase 4) all implement the same five
methods; nothing downstream needs to know which one it is holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy.typing as npt
import pandas as pd


class Detector(ABC):
    """A binary classifier over account (or transaction) feature vectors.

    ``fit``/``predict_proba`` operate on the feature matrix produced by
    :class:`ledger.features.pipeline.FeaturePipeline` — a
    ``DataFrame`` of purely numeric columns, indexed by entity id. Models
    are responsible for their own persistence (``save``/``load``) so the
    training pipeline never needs to know a model's internal structure.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Detector:
        """Fit on a training fold. ``y`` is a 0/1 Series aligned to ``X``'s index."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Return the predicted probability of the positive class for each row of ``X``."""

    @abstractmethod
    def save(self, path: Path | str) -> Path:
        """Persist the fitted model (including any preprocessing state) to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> Detector:
        """Load a model previously written by :meth:`save`."""

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return per-feature importance, indexed by feature name, descending."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict describing this model instance
        (architecture, hyperparameters, and anything fit() derived, such
        as a computed class-imbalance weight).
        """


__all__ = ["Detector"]
