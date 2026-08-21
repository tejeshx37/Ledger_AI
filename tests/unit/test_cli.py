"""Unit tests for the `ledger` CLI entry point."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from ledger.cli.main import app

runner = CliRunner()


def test_config_show_prints_valid_json() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["dataset_name"] == "elliptic"


def test_config_hash_prints_a_sha256_hex_digest() -> None:
    result = runner.invoke(app, ["config", "hash"])
    assert result.exit_code == 0
    digest = result.stdout.strip()
    assert len(digest) == 64
    int(digest, 16)  # raises if not valid hex


def test_data_fetch_reports_missing_files_when_none_placed() -> None:
    # Phase 2 behavior test: see tests/unit/test_prepare_cli.py for the
    # full fetch/prepare flow with files actually placed.
    result = runner.invoke(app, ["data", "fetch", "elliptic"])
    assert result.exit_code == 1
    assert "missing" in result.output


def test_main_help_does_not_crash() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
