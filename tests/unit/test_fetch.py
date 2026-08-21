"""Unit tests for dataset acquisition instructions and checksum verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.data.fetch import (
    DatasetChecksumMismatchError,
    fetch_dataset,
    format_fetch_instructions,
    verify_dataset_files,
)
from ledger.data.registry import DATASET_REGISTRY, DatasetFile, DatasetSpec


@pytest.fixture
def registered_fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a throwaway dataset with a pinned checksum, restored after the test."""
    spec = DatasetSpec(
        name="fake",
        role="test fixture",
        license_name="test",
        source_description="test",
        required_credentials=(),
        fetch_instructions="copy a.csv into place",
        files=(DatasetFile("a.csv", "test file", expected_sha256="deadbeef"),),
    )
    registry = dict(DATASET_REGISTRY)
    registry["fake"] = spec
    monkeypatch.setattr("ledger.data.registry.DATASET_REGISTRY", registry)


def test_format_fetch_instructions_includes_role_license_and_files() -> None:
    text = format_fetch_instructions("elliptic")
    assert "elliptic_txs_features.csv" in text
    assert "Role:" in text
    assert "License:" in text


def test_format_fetch_instructions_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        format_fetch_instructions("not_a_real_dataset")


def test_verify_missing_file_reports_missing(tmp_path: Path) -> None:
    report = verify_dataset_files("elliptic", tmp_path / "nowhere")
    assert not report.all_present
    assert all(r.status == "missing" for r in report.results)


def test_verify_present_unpinned_file_is_ready(tmp_path: Path) -> None:
    raw_dir = tmp_path / "elliptic"
    raw_dir.mkdir()
    for f in DATASET_REGISTRY["elliptic"].files:
        (raw_dir / f.filename).write_text("some content")
    report = verify_dataset_files("elliptic", raw_dir)
    assert report.all_present
    assert not report.any_mismatch
    assert report.is_ready
    assert all(r.status == "unpinned" for r in report.results)
    assert all(r.actual_sha256 is not None for r in report.results)


def test_fetch_dataset_mismatch_raises(tmp_path: Path, registered_fake_dataset: None) -> None:
    raw_dir = tmp_path / "fake"
    raw_dir.mkdir()
    (raw_dir / "a.csv").write_text("not the expected content")
    with pytest.raises(DatasetChecksumMismatchError, match="Checksum mismatch"):
        fetch_dataset("fake", raw_dir)


def test_fetch_dataset_match_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "fake"
    raw_dir.mkdir()
    (raw_dir / "a.csv").write_bytes(b"exact content")
    from ledger.utils.checksums import sha256_file

    digest = sha256_file(raw_dir / "a.csv")
    spec = DatasetSpec(
        name="fake_match",
        role="test",
        license_name="test",
        source_description="test",
        required_credentials=(),
        fetch_instructions="test",
        files=(DatasetFile("a.csv", "test", expected_sha256=digest),),
    )
    registry = dict(DATASET_REGISTRY)
    registry["fake_match"] = spec
    monkeypatch.setattr("ledger.data.registry.DATASET_REGISTRY", registry)

    report = fetch_dataset("fake_match", raw_dir)
    assert report.is_ready
    assert report.results[0].status == "match"


def test_fetch_dataset_missing_file_does_not_raise_but_reports(tmp_path: Path) -> None:
    report = fetch_dataset("elliptic", tmp_path / "nowhere")
    assert not report.all_present
    assert not report.is_ready
