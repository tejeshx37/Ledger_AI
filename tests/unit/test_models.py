"""Unit tests for the Detector interface, the XGBoost baseline, and the model registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ledger.config.models import ModelConfig
from ledger.models.base import Detector
from ledger.models.registry import MODEL_REGISTRY, get_detector
from ledger.models.xgboost_baseline import DetectorNotFittedError, XGBoostBaselineDetector


def _toy_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.2).astype(int), name="label")
    # feature correlated with label so the model has real signal
    x1 = rng.normal(loc=y * 2.0, scale=1.0)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"f1": x1, "f2": x2})
    return X, y


def test_xgboost_baseline_is_a_detector() -> None:
    assert issubclass(XGBoostBaselineDetector, Detector)


def test_fit_computes_scale_pos_weight_from_fold_not_constant() -> None:
    X, y = _toy_data(n=300, seed=1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    detector = XGBoostBaselineDetector(config=ModelConfig())
    detector.fit(X, y)
    assert detector.metadata()["scale_pos_weight"] == pytest.approx(n_neg / n_pos)


def test_fit_rejects_zero_positive_examples() -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0])
    detector = XGBoostBaselineDetector(config=ModelConfig())
    with pytest.raises(ValueError, match="zero positive"):
        detector.fit(X, y)


def test_fit_rejects_zero_negative_examples() -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([1, 1, 1])
    detector = XGBoostBaselineDetector(config=ModelConfig())
    with pytest.raises(ValueError, match="zero negative"):
        detector.fit(X, y)


def test_predict_proba_before_fit_raises() -> None:
    detector = XGBoostBaselineDetector(config=ModelConfig())
    with pytest.raises(DetectorNotFittedError):
        detector.predict_proba(pd.DataFrame({"f1": [1.0]}))


def test_feature_importance_before_fit_raises() -> None:
    detector = XGBoostBaselineDetector(config=ModelConfig())
    with pytest.raises(DetectorNotFittedError):
        detector.feature_importance()


def test_save_before_fit_raises(tmp_path: Path) -> None:
    detector = XGBoostBaselineDetector(config=ModelConfig())
    with pytest.raises(DetectorNotFittedError):
        detector.save(tmp_path / "model.joblib")


def test_fit_predict_produces_probabilities_in_range() -> None:
    X, y = _toy_data(n=300, seed=2)
    detector = XGBoostBaselineDetector(config=ModelConfig())
    detector.fit(X, y)
    scores = detector.predict_proba(X)
    assert len(scores) == len(X)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_feature_importance_returns_series_indexed_by_column_descending() -> None:
    X, y = _toy_data(n=300, seed=3)
    detector = XGBoostBaselineDetector(config=ModelConfig())
    detector.fit(X, y)
    importance = detector.feature_importance()
    assert set(importance.index) == set(X.columns)
    assert list(importance.values) == sorted(importance.values, reverse=True)


def test_metadata_reflects_config() -> None:
    config = ModelConfig(xgboost_max_depth=3, xgboost_n_estimators=42, random_seed=7)
    X, y = _toy_data(n=200, seed=4)
    detector = XGBoostBaselineDetector(config=config)
    detector.fit(X, y)
    meta = detector.metadata()
    assert meta["xgboost_max_depth"] == 3
    assert meta["xgboost_n_estimators"] == 42
    assert meta["random_seed"] == 7
    assert meta["n_features"] == 2


def test_save_load_roundtrip_produces_identical_predictions(tmp_path: Path) -> None:
    X, y = _toy_data(n=300, seed=5)
    detector = XGBoostBaselineDetector(config=ModelConfig())
    detector.fit(X, y)
    before = detector.predict_proba(X)

    path = detector.save(tmp_path / "model.joblib")
    loaded = XGBoostBaselineDetector.load(path)
    after = loaded.predict_proba(X)

    assert np.allclose(before, after)
    assert loaded.metadata() == detector.metadata()


def test_load_rejects_non_detector_file(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError):
        XGBoostBaselineDetector.load(path)


def test_get_detector_returns_registered_model() -> None:
    detector = get_detector(ModelConfig(name="xgboost_baseline"))
    assert isinstance(detector, MODEL_REGISTRY["xgboost_baseline"])


def test_get_detector_xgboost_ignores_graph_context() -> None:
    # XGBoost has no use for training_config/features_config/graph; passing
    # them (or not) must not matter.
    detector = get_detector(ModelConfig(name="xgboost_baseline"), graph=object())
    assert isinstance(detector, MODEL_REGISTRY["xgboost_baseline"])


def test_get_detector_graph_model_without_context_raises_value_error() -> None:
    with pytest.raises(ValueError, match="training_config"):
        get_detector(ModelConfig(name="graphsage"))


def test_get_detector_graph_model_with_context_succeeds(elliptic_graph_fixture) -> None:
    from ledger.config.models import TrainingConfig
    from ledger.models.graphsage import GraphSAGEDetector

    detector = get_detector(
        ModelConfig(name="graphsage"),
        training_config=TrainingConfig(),
        features_config=elliptic_graph_fixture.features_config,
        graph=elliptic_graph_fixture.attributed_graph,
    )
    assert isinstance(detector, GraphSAGEDetector)
    assert isinstance(detector, MODEL_REGISTRY["graphsage"])


def test_get_detector_tgat_with_context_succeeds(elliptic_graph_fixture) -> None:
    from ledger.config.models import TrainingConfig
    from ledger.models.tgat import TGATDetector

    detector = get_detector(
        ModelConfig(name="tgat"),
        training_config=TrainingConfig(),
        features_config=elliptic_graph_fixture.features_config,
        graph=elliptic_graph_fixture.attributed_graph,
    )
    assert isinstance(detector, TGATDetector)
