"""Unit tests for RunManifest and checksumming."""

from __future__ import annotations

import json
from pathlib import Path

from ledger.utils.checksums import sha256_file
from ledger.utils.manifest import RunManifest


def test_sha256_file_matches_known_answer(tmp_path: Path) -> None:
    p = tmp_path / "sample.txt"
    p.write_text("hello world")
    # Known SHA-256 of the literal bytes b"hello world".
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha256_file(p) == expected


def test_sha256_file_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("content a")
    b.write_text("content b")
    assert sha256_file(a) != sha256_file(b)


def test_run_manifest_create_populates_fields() -> None:
    manifest = RunManifest.create(
        run_id="run_test_1",
        config_hash="deadbeef",
        dataset_checksums={"data/raw/elliptic.csv": "abc123"},
        seed=42,
        repo_root=Path("."),
    )
    assert manifest.run_id == "run_test_1"
    assert manifest.config_hash == "deadbeef"
    assert manifest.seed == 42
    assert manifest.dataset_checksums == {"data/raw/elliptic.csv": "abc123"}
    assert manifest.hostname
    assert manifest.timestamp
    assert manifest.python_version
    assert isinstance(manifest.git_dirty, bool)


def test_run_manifest_write_and_read_roundtrip(tmp_path: Path) -> None:
    manifest = RunManifest.create(
        run_id="run_test_2",
        config_hash="cafef00d",
        dataset_checksums={},
        seed=7,
        repo_root=Path("."),
    )
    path = manifest.write(runs_dir=tmp_path)
    assert path == tmp_path / "run_test_2" / "manifest.json"
    assert path.exists()

    with path.open() as f:
        payload = json.load(f)
    assert payload["run_id"] == "run_test_2"
    assert payload["seed"] == 7

    reloaded = RunManifest.read(path)
    assert reloaded == manifest


def test_run_manifest_git_sha_is_real_when_in_repo() -> None:
    manifest = RunManifest.create(
        run_id="run_test_3",
        config_hash="x",
        dataset_checksums={},
        seed=0,
        repo_root=Path(__file__).resolve().parents[2],
    )
    assert manifest.git_sha != "unknown"
    assert len(manifest.git_sha) == 40
