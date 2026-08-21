"""Model registry: maps ``ModelConfig.name`` to the :class:`Detector` that
implements it.

XGBoost only ever needs its own hyperparameters (:class:`ModelConfig`).
GraphSAGE and TGAT additionally need the optimisation loop's
hyperparameters (:class:`TrainingConfig`), the feature-family config used
to compute motif ground truth (:class:`FeaturesConfig`), and the
attributed graph itself — none of which a purely tabular model like
XGBoost has any use for. :func:`get_detector` accepts all of it and
dispatches by name, raising a clear error if a graph model is requested
without its required context rather than constructing something broken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ledger.config.models import FeaturesConfig, ModelConfig, TrainingConfig
from ledger.models.base import Detector
from ledger.models.graphsage import GraphSAGEDetector
from ledger.models.tgat import TGATDetector
from ledger.models.xgboost_baseline import XGBoostBaselineDetector

if TYPE_CHECKING:
    import networkx as nx

MODEL_REGISTRY: dict[str, type[Detector]] = {
    "xgboost_baseline": XGBoostBaselineDetector,
    "graphsage": GraphSAGEDetector,
    "tgat": TGATDetector,
}

GRAPH_MODEL_NAMES = frozenset({"graphsage", "tgat"})


def get_detector(
    model_config: ModelConfig,
    training_config: TrainingConfig | None = None,
    features_config: FeaturesConfig | None = None,
    graph: nx.MultiDiGraph[str] | None = None,
) -> Detector:
    """Instantiate the detector registered for ``model_config.name``.

    ``training_config``, ``features_config``, and ``graph`` are required
    for ``graphsage``/``tgat`` (raises :class:`ValueError` naming whichever
    is missing) and ignored for ``xgboost_baseline``.
    """
    try:
        detector_cls = MODEL_REGISTRY[model_config.name]
    except KeyError as exc:
        raise KeyError(f"No detector registered for model {model_config.name!r}") from exc

    if model_config.name not in GRAPH_MODEL_NAMES:
        return detector_cls(model_config)  # type: ignore[call-arg]

    missing = [
        name
        for name, value in (
            ("training_config", training_config),
            ("features_config", features_config),
            ("graph", graph),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"model {model_config.name!r} requires {missing} to be provided to get_detector()"
        )
    return detector_cls(
        config=model_config,
        training_config=training_config,
        features_config=features_config,
        graph=graph,
    )  # type: ignore[call-arg]


__all__ = ["MODEL_REGISTRY", "GRAPH_MODEL_NAMES", "get_detector"]
