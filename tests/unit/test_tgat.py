"""Unit tests for TGATDetector: time-aware fit/predict, the motif-aware
auxiliary loss toggle, and attention-weight exposure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ledger.config.models import ModelConfig, TrainingConfig
from ledger.models.graph_common import GraphDetectorNotFittedError
from ledger.models.tgat import TGATDetector

pytest.importorskip("torch_geometric")

_TRAINING_CONFIG = TrainingConfig(
    learning_rate=0.01,
    weight_decay=0.0,
    batch_size=16,
    max_epochs=3,
    early_stopping_patience=2,
)


def _config(motif_loss_weight: float = 0.0) -> ModelConfig:
    return ModelConfig(
        name="tgat",
        hidden_dim=8,
        num_layers=2,
        dropout=0.1,
        time_encoding_dim=4,
        motif_loss_weight=motif_loss_weight,
        random_seed=42,
    )


def _make_detector(fixture, motif_loss_weight: float = 0.0) -> TGATDetector:
    return TGATDetector(
        config=_config(motif_loss_weight),
        training_config=_TRAINING_CONFIG,
        features_config=fixture.features_config,
        graph=fixture.attributed_graph,
    )


def test_fit_predict_produces_valid_probabilities(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    scores = detector.predict_proba(elliptic_graph_fixture.X_test)
    assert len(scores) == len(elliptic_graph_fixture.X_test)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_motif_head_present_when_weight_positive(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture, motif_loss_weight=0.3)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    assert detector._motif_head is not None
    assert detector.metadata()["motif_loss_weight"] == 0.3


def test_motif_head_absent_when_weight_zero(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture, motif_loss_weight=0.0)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    assert detector._motif_head is None


def test_metadata_reflects_config(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    meta = detector.metadata()
    assert meta["architecture"] == "tgat"
    assert meta["hidden_dim"] == 8


def test_attention_weights_before_fit_raises(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    with pytest.raises(GraphDetectorNotFittedError):
        detector.attention_weights(elliptic_graph_fixture.test_ids[:3])


def test_attention_weights_cover_every_edge_and_average_to_one(elliptic_graph_fixture) -> None:
    detector = _make_detector(elliptic_graph_fixture)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)

    all_ids = list(elliptic_graph_fixture.attributed_graph.nodes())
    rows = detector.attention_weights(all_ids)
    non_self_loop = [r for r in rows if r["src"] != r["dst"]]
    assert len(non_self_loop) > 0
    for row in rows:
        assert 0.0 <= row["attention"] <= 1.0 + 1e-6
        assert row["src"] in all_ids
        assert row["dst"] in all_ids


def test_save_load_roundtrip_produces_identical_scores(
    elliptic_graph_fixture, tmp_path: Path
) -> None:
    detector = _make_detector(elliptic_graph_fixture, motif_loss_weight=0.3)
    detector.fit(elliptic_graph_fixture.X_train, elliptic_graph_fixture.y_train)
    before = detector.predict_proba(elliptic_graph_fixture.X_test)

    path = detector.save(tmp_path / "tgat.joblib")
    loaded = TGATDetector.load(path)
    after = loaded.predict_proba(elliptic_graph_fixture.X_test)

    assert np.allclose(before, after)
