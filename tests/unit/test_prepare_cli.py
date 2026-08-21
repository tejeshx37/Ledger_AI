"""Unit tests for the Phase 2 orchestration (`prepare_dataset`) and its
`ledger data fetch` / `ledger data prepare` CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ledger.cli.main import app
from ledger.config.settings import load_settings
from ledger.data.prepare import prepare_dataset

runner = CliRunner()


def _write_config(config_dir: Path, tmp_path: Path) -> None:
    """Point paths.data_raw_dir at tmp_path/'raw', matching where the
    elliptic_raw_dir fixture (see conftest.py) already places its files —
    no copying needed.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "base.yaml").write_text(f"""
paths:
  data_raw_dir: "{tmp_path / 'raw'}"
  data_processed_dir: "{tmp_path / 'processed'}"
  runs_dir: "{tmp_path / 'runs'}"
data:
  dataset_name: elliptic
""")


def test_prepare_dataset_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)

    settings = load_settings(config_dir=config_dir)
    result = prepare_dataset(settings, dataset_name="elliptic")

    assert result.canonical_dir.exists()
    assert (result.canonical_dir / "accounts.parquet").exists()
    assert result.graph_path.exists()
    assert result.node_statistics_path.exists()
    assert result.split_manifest_path.exists()
    assert result.split_ids_path.exists()
    assert result.manifest_path.exists()

    split_ids = json.loads(result.split_ids_path.read_text())
    assert set(split_ids) == {"train", "val", "test"}
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_checksums"]


def test_prepare_dataset_missing_files_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)
    settings = load_settings(config_dir=config_dir)
    with pytest.raises(FileNotFoundError, match="missing file"):
        prepare_dataset(settings, dataset_name="elliptic")


def test_cli_fetch_unknown_dataset_exits_nonzero() -> None:
    result = runner.invoke(app, ["data", "fetch", "not_a_real_dataset"])
    assert result.exit_code == 1
    assert "Unknown dataset" in result.output


def test_cli_fetch_prints_instructions_and_reports_missing(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)
    result = runner.invoke(app, ["data", "fetch", "elliptic", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "elliptic_txs_features.csv" in result.output
    assert "missing" in result.output


def test_cli_fetch_succeeds_once_files_present(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)

    result = runner.invoke(app, ["data", "fetch", "elliptic", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    assert "All expected files present" in result.output


def test_cli_prepare_end_to_end(tmp_path: Path, elliptic_raw_dir: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)

    result = runner.invoke(app, ["data", "prepare", "elliptic", "--config-dir", str(config_dir)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_name"] == "elliptic"
    assert Path(payload["canonical_dir"]).exists()


def test_cli_prepare_missing_files_exits_nonzero(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir, tmp_path)
    result = runner.invoke(app, ["data", "prepare", "elliptic", "--config-dir", str(config_dir)])
    assert result.exit_code == 1
    assert "ledger data prepare failed" in result.output
