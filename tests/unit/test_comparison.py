"""Unit tests for the Phase 4.4 comparison protocol: identical splits
across models, the ring-participation breakdown, and the `ledger compare`
CLI command.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ledger.cli.main import app
from ledger.evaluation.comparison import run_model_comparison

pytest.importorskip("torch_geometric")

runner = CliRunner()


def _write_configs(config_dir: Path, tmp_path: Path) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "base.yaml").write_text(f"""
paths:
  data_raw_dir: "{tmp_path / 'raw'}"
  data_processed_dir: "{tmp_path / 'processed'}"
  runs_dir: "{tmp_path / 'runs'}"
data:
  dataset_name: elliptic
""")
    baseline = config_dir / "model_baseline.yaml"
    baseline.write_text("""
model:
  name: xgboost_baseline
  xgboost_max_depth: 3
  xgboost_n_estimators: 20
features:
  use_graph_topological: false
  use_temporal: false
  use_motif_counts: false
""")
    graphsage = config_dir / "model_graphsage.yaml"
    graphsage.write_text("""
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
    tgat = config_dir / "model_tgat.yaml"
    tgat.write_text("""
model:
  name: tgat
  hidden_dim: 8
  num_layers: 2
  time_encoding_dim: 4
  motif_loss_weight: 0.2
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
    return [baseline, graphsage, tgat]


def test_run_model_comparison_requires_at_least_two_configs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_model_comparison([tmp_path / "only_one.yaml"])


def test_run_model_comparison_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    overlay_paths = _write_configs(config_dir, tmp_path)

    result = run_model_comparison(overlay_paths)

    assert result.table_path.exists()
    assert result.table_csv_path.exists()
    assert result.manifest_path.exists()
    assert set(result.per_model_runs) == {"xgboost_baseline", "graphsage", "tgat"}
    assert len(result.table) == 3

    for row in result.table:
        assert row["n_test"] == row["n_ring"] + row["n_isolated"]
        assert "overall" in row and "ring" in row and "isolated" in row
        assert row["overall"]["n_samples"] == row["n_test"]

    # every model was trained on the identical test split
    test_id_sets = {frozenset(r.test_account_ids) for r in result.per_model_runs.values()}
    assert len(test_id_sets) == 1


def test_run_model_comparison_table_json_matches_returned_table(
    tmp_path: Path, elliptic_raw_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    overlay_paths = _write_configs(config_dir, tmp_path)
    result = run_model_comparison(overlay_paths)

    on_disk = json.loads(result.table_path.read_text())
    assert on_disk == result.table


def test_run_model_comparison_csv_has_one_row_per_model(
    tmp_path: Path, elliptic_raw_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    overlay_paths = _write_configs(config_dir, tmp_path)
    result = run_model_comparison(overlay_paths)

    with result.table_csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert {r["model_name"] for r in rows} == {"xgboost_baseline", "graphsage", "tgat"}


def test_run_model_comparison_mismatched_dataset_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("data:\n  dataset_name: elliptic\n")
    (config_dir / "a.yaml").write_text("model:\n  name: xgboost_baseline\n")
    other_dir = tmp_path / "other_configs"
    other_dir.mkdir()
    (other_dir / "base.yaml").write_text("data:\n  dataset_name: baf\n")
    (other_dir / "b.yaml").write_text("model:\n  name: xgboost_baseline\n")

    with pytest.raises(ValueError, match="same dataset"):
        run_model_comparison([config_dir / "a.yaml", other_dir / "b.yaml"])


def test_cli_compare_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_configs(config_dir, tmp_path)

    result = runner.invoke(app, ["compare", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_name"] == "elliptic"
    assert len(payload["table"]) == 3
    assert Path(payload["table_path"]).exists()


def test_cli_compare_missing_files_exits_nonzero(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_configs(config_dir, tmp_path)
    result = runner.invoke(app, ["compare", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "ledger compare failed" in result.output
