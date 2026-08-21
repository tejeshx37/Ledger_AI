"""Model registry: maps ``ModelConfig.name`` to the :class:`Detector` that
implements it. GraphSAGE and TGAT are registered in the config's allowed
model names (see :class:`ledger.config.models.ModelConfig`) but not yet
implemented — requesting them raises ``NotImplementedError`` rather than
silently falling back to the baseline or fabricating a result.
"""

from __future__ import annotations

from collections.abc import Callable

from ledger.config.models import ModelConfig
from ledger.models.base import Detector
from ledger.models.xgboost_baseline import XGBoostBaselineDetector

MODEL_REGISTRY: dict[str, Callable[[ModelConfig], Detector]] = {
    "xgboost_baseline": XGBoostBaselineDetector,
}

_NOT_YET_IMPLEMENTED = {"graphsage", "tgat"}


def get_detector(config: ModelConfig) -> Detector:
    """Instantiate the detector registered for ``config.name``."""
    if config.name in _NOT_YET_IMPLEMENTED:
        raise NotImplementedError(
            f"model {config.name!r} is not implemented yet (Phase 4 of the project "
            "brief). Only 'xgboost_baseline' is available."
        )
    try:
        detector_cls = MODEL_REGISTRY[config.name]
    except KeyError as exc:
        raise KeyError(f"No detector registered for model {config.name!r}") from exc
    return detector_cls(config)


__all__ = ["MODEL_REGISTRY", "get_detector"]
