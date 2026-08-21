"""XGBoost baseline detector: flat features only, no graph structure.

This represents what a bank runs today (a tabular classifier over
per-account features) and is the floor every later model (Phase 4's
GraphSAGE and TGAT) is measured against — the project's central claim is
that graph structure beats this, not that it can match it.

Class imbalance is handled by ``scale_pos_weight``, computed from whatever
training fold :meth:`fit` actually receives — never a constant in config —
so it's always correct for the exact class balance of that fold's data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy.typing as npt
import pandas as pd

from ledger.config.models import ModelConfig
from ledger.models.base import Detector


class DetectorNotFittedError(RuntimeError):
    """Raised when predict_proba/feature_importance is called before fit()."""


@dataclass
class XGBoostBaselineDetector(Detector):
    """Gradient-boosted trees over flat per-account features (``xgboost.XGBClassifier``)."""

    config: ModelConfig
    _model: Any = field(default=None, repr=False)
    _feature_names: list[str] | None = field(default=None, repr=False)
    _scale_pos_weight: float | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostBaselineDetector:
        import xgboost as xgb

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0:
            raise ValueError("cannot fit: training fold has zero positive examples")
        if n_neg == 0:
            raise ValueError("cannot fit: training fold has zero negative examples")
        self._scale_pos_weight = n_neg / n_pos
        self._feature_names = list(X.columns)

        self._model = xgb.XGBClassifier(
            max_depth=self.config.xgboost_max_depth,
            n_estimators=self.config.xgboost_n_estimators,
            scale_pos_weight=self._scale_pos_weight,
            random_state=self.config.random_seed,
            eval_metric="aucpr",
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        if self._model is None or self._feature_names is None:
            raise DetectorNotFittedError("predict_proba() called before fit()")
        scores: npt.NDArray[Any] = self._model.predict_proba(X[self._feature_names].to_numpy())[
            :, 1
        ]
        return scores

    def feature_importance(self) -> pd.Series:
        if self._model is None or self._feature_names is None:
            raise DetectorNotFittedError("feature_importance() called before fit()")
        return pd.Series(self._model.feature_importances_, index=self._feature_names).sort_values(
            ascending=False
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.config.name,
            "xgboost_max_depth": self.config.xgboost_max_depth,
            "xgboost_n_estimators": self.config.xgboost_n_estimators,
            "random_seed": self.config.random_seed,
            "scale_pos_weight": self._scale_pos_weight,
            "n_features": len(self._feature_names) if self._feature_names else None,
        }

    def save(self, path: Path | str) -> Path:
        import joblib

        if self._model is None:
            raise DetectorNotFittedError("save() called before fit()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> XGBoostBaselineDetector:
        import joblib

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded


__all__ = ["XGBoostBaselineDetector", "DetectorNotFittedError"]
