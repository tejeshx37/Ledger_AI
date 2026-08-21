"""Unit tests for GraphSAGEDetector: inductive fit/predict, persistence,
feature importance, and the not-fitted guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ledger.config.models import ModelConfig, TrainingConfig
from ledger.models.graph_common import GraphDetectorNotFittedError
from ledger.models.graphsage import GraphSAGEDetector

pytest.importorskip("torch_geometric")

_MODEL_CONFIG = ModelConfig(
    name="graphsage",
    hidden_dim=8,
    num_layers=2,
    dropout=0.1,
    fanout_per_layer=[4, 4],
    motif_loss_weight=0.0,
    random_seed=42,
)
_TRAINING_CONFIG = TrainingConfig(
    learning_rate=0.01,
    weight_decay=0.0,
    batch_size=16,
    max_epochs=3,
    early_stopping_patience=2,
)


def _make_detector(fixture) -> GraphSAGEDetector:
    return GraphSAGEDetector(
        config=_MODEL_CONFIG,
        training_config=_TRAINING_CONFIG,
        features_config=fixture.features_config,
        graph=fixture.attributed_graph,
    )


def test_predict_before_fit_raises(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    with pytest.raises(GraphDetectorNotFittedError):
        detector.predict_proba(elliptic_graph_fixture.X_test)


def test_feature_importance_before_fit_raises(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    with pytest.raises(GraphDetectorNotFittedError):
        detector.feature_importance()


def test_save_before_fit_raises(elliptic_graph_fixture, tmp_path: Path) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    with pytest.raises(GraphDetectorNotFittedError):
        detector.save(tmp_path / "model.joblib")


def test_fit_rejects_zero_positive_examples(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    y_all_negative = elliptic_graph_fixture.y_train * 0
    with pytest.raises(ValueError, match="zero positive"):
        detector.fit(elliptic_graph_fixture.X_train, y_all_negative)


def test_fit_predict_produces_valid_probabilities(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    scores = detector.predict_proba(elliptic_graph_fixture.X_test)
    assert len(scores) == len(elliptic_graph_fixture.X_test)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_predict_is_inductive_over_unseen_test_node(elliptic_graph_fixture) -> None:
    """The model must score a test-set account whose weights it never
    backpropagated through (it was excluded from the training subgraph).
    """
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)

    unseen_id = elliptic_graph_fixture.test_ids[0]
    assert unseen_id not in set(elliptic_graph_fixture.train_ids)

    single = elliptic_graph_fixture.X_test.loc[[unseen_id]]
    score = detector.predict_proba(single)
    assert len(score) == 1
    assert 0.0 <= score[0] <= 1.0


def test_predict_unknown_account_id_raises(elliptic_graph_fixture) -> None:
    import pandas as pd

    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    bogus = pd.DataFrame(
        {c: [0.0] for c in elliptic_graph_fixture.X_test.columns}, index=["not_a_real_account"]
    )
    with pytest.raises(KeyError):
        detector.predict_proba(bogus)


def test_feature_importance_indexed_by_columns(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    importance = detector.feature_importance()
    assert set(importance.index) == set(elliptic_graph_fixture.X_train.columns)
    assert list(importance.values) == sorted(importance.values, reverse=True)


def test_metadata_reflects_config(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    meta = detector.metadata()
    assert meta["architecture"] == "graphsage"
    assert meta["hidden_dim"] == 8
    assert meta["num_layers"] == 2
    assert meta["fanout_per_layer"] == [4, 4]


def test_save_load_roundtrip_produces_identical_scores(
    elliptic_graph_fixture, tmp_path: Path
) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    before = detector.predict_proba(elliptic_graph_fixture.X_test)

    path = detector.save(tmp_path / "sage.joblib")
    loaded = GraphSAGEDetector.load(path)
    after = loaded.predict_proba(elliptic_graph_fixture.X_test)

    assert np.allclose(before, after)


def test_load_rejects_non_detector_file(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError):
        GraphSAGEDetector.load(path)


def test_graph_detector_cannot_be_instantiated_directly(elliptic_graph_fixture) -> None:
    from ledger.models.graph_common import GraphDetector

    with pytest.raises(TypeError):
        GraphDetector(  # type: ignore[abstract]
            config=_MODEL_CONFIG,
            training_config=_TRAINING_CONFIG,
            features_config=elliptic_graph_fixture.features_config,
            graph=elliptic_graph_fixture.attributed_graph,
        )
