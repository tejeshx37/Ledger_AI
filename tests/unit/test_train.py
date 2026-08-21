"""End-to-end tests for the Phase 3 training orchestration and its CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ledger.cli.main import app
from ledger.config.settings import load_settings_from_overlay_path
from ledger.training.train import train_detector

runner = CliRunner()


def _write_configs(config_dir: Path, tmp_path: Path, dataset_name: str = "elliptic") -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "base.yaml").write_text(f"""
paths:
  data_raw_dir: "{tmp_path / 'raw'}"
  data_processed_dir: "{tmp_path / 'processed'}"
  runs_dir: "{tmp_path / 'runs'}"
data:
  dataset_name: {dataset_name}
""")
    overlay = config_dir / "model_baseline.yaml"
    overlay.write_text("""
model:
  name: xgboost_baseline
  xgboost_max_depth: 3
  xgboost_n_estimators: 30
features:
  use_graph_topological: false
  use_temporal: false
  use_motif_counts: false
""")
    return overlay


def test_train_detector_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)
    settings = load_settings_from_overlay_path(overlay)

    result = train_detector(settings)

    assert result.model_path.exists()
    assert result.feature_pipeline_path.exists()
    assert result.metrics_path.exists()
    assert result.pr_curve_path.exists()
    assert result.pr_curve_path.with_suffix(".json").exists()
    assert result.feature_importance_path.exists()
    assert result.manifest_path.exists()

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["dataset_name"] == "elliptic"
    assert metrics["test"]["n_samples"] > 0
    assert 0.0 <= metrics["test"]["roc_auc"] <= 1.0
    assert metrics["model_metadata"]["scale_pos_weight"] is not None

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["config_hash"] == settings.hash()


def test_train_detector_reloaded_model_matches_saved_scores(
    tmp_path: Path, elliptic_raw_dir: Path
) -> None:
    from ledger.data.canonical import CanonicalDataset
    from ledger.features.pipeline import FeaturePipeline
    from ledger.models.registry import MODEL_REGISTRY

    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)
    settings = load_settings_from_overlay_path(overlay)
    result = train_detector(settings)

    canonical = CanonicalDataset.read(settings.paths.data_processed_dir, "elliptic")
    pipeline = FeaturePipeline.load(result.feature_pipeline_path)
    detector = MODEL_REGISTRY["xgboost_baseline"].load(result.model_path)

    X = pipeline.transform(canonical).set_index("account_id")
    scores = detector.predict_proba(X)
    assert len(scores) == len(canonical.accounts)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_train_detector_ibm_aml_supported(tmp_path: Path, ibm_aml_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path, dataset_name="ibm_aml")
    settings = load_settings_from_overlay_path(overlay)
    res = train_detector(settings)
    assert res.run_id is not None


def test_train_detector_missing_raw_files_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)
    settings = load_settings_from_overlay_path(overlay)
    with pytest.raises(FileNotFoundError, match="missing file"):
        train_detector(settings)


def test_cli_train_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)

    result = runner.invoke(app, ["train", "--config", str(overlay)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_name"] == "elliptic"
    assert Path(payload["model_path"]).exists()
    assert "roc_auc" in payload["test_metrics"]


def test_cli_train_graphsage_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    pytest.importorskip("torch_geometric")
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)
    overlay.write_text("""
model:
  name: graphsage
  hidden_dim: 8
  num_layers: 2
  fanout_per_layer: [4, 4]
features:
  use_graph_topological: true
  use_temporal: true
  use_motif_counts: true
training:
  learning_rate: 0.01
  batch_size: 16
  max_epochs: 3
  early_stopping_patience: 2
""")

    result = runner.invoke(app, ["train", "--config", str(overlay)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_name"] == "elliptic"
    assert Path(payload["model_path"]).exists()
    assert "roc_auc" in payload["test_metrics"]


def test_cli_train_missing_files_exits_nonzero(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay = _write_configs(config_dir, tmp_path)
    result = runner.invoke(app, ["train", "--config", str(overlay)])
    assert result.exit_code == 1
    assert "ledger train failed" in result.output
