"""Dataset acquisition: manual-instructions printing plus checksum
verification of files the operator has placed on disk.

LEDGER never downloads a dataset silently — every dataset here is gated
behind either an account (Kaggle) or a release page, so acquisition is a
human action. This module's job is to make that action unambiguous
(:func:`format_fetch_instructions`) and to make it impossible to silently
proceed with the wrong or corrupted files
(:func:`verify_dataset_files`, :func:`fetch_dataset`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ledger.data.registry import DatasetFile, DatasetSpec, get_dataset_spec
from ledger.utils.checksums import sha256_file

ChecksumStatus = Literal["match", "mismatch", "unpinned", "missing"]


@dataclass(frozen=True)
class FileCheckResult:
    """Verification outcome for a single expected dataset file."""

    file: DatasetFile
    path: Path
    status: ChecksumStatus
    actual_sha256: str | None


@dataclass(frozen=True)
class FetchReport:
    """Verification outcome for an entire dataset's expected files."""

    dataset: DatasetSpec
    raw_dir: Path
    results: tuple[FileCheckResult, ...]

    @property
    def all_present(self) -> bool:
        return all(r.status != "missing" for r in self.results)

    @property
    def any_mismatch(self) -> bool:
        return any(r.status == "mismatch" for r in self.results)

    @property
    def is_ready(self) -> bool:
        """True if every file is present and no checksum mismatch exists.

        A file with an ``unpinned`` checksum still counts as ready: there
        is nothing to mismatch against yet, and refusing to proceed would
        make the registry's unpinned-checksum design pointless.
        """
        return self.all_present and not self.any_mismatch


class DatasetChecksumMismatchError(ValueError):
    """Raised when a placed file's digest does not match the pinned one."""


def format_fetch_instructions(name: str) -> str:
    """Return the human-readable acquisition steps for a dataset."""
    spec = get_dataset_spec(name)
    lines = [
        f"Dataset: {spec.name}",
        f"Role: {spec.role}",
        f"Source: {spec.source_description}",
        f"License: {spec.license_name}",
    ]
    if spec.required_credentials:
        lines.append(f"Required credentials: {', '.join(spec.required_credentials)}")
    lines.append("")
    lines.append(spec.fetch_instructions)
    lines.append("")
    lines.append("Expected files:")
    for f in spec.files:
        lines.append(f"  - {f.filename}: {f.description}")
    return "\n".join(lines)


def verify_dataset_files(name: str, raw_dir: Path | str) -> FetchReport:
    """Check each expected file's presence and checksum under ``raw_dir``.

    A file with no pinned ``expected_sha256`` is reported as ``unpinned``
    (its actual digest is still computed and returned, so an operator can
    choose to pin it) rather than passed or failed — the registry ships
    without a value for datasets where no stable published digest exists.
    """
    spec = get_dataset_spec(name)
    raw_dir = Path(raw_dir)
    results: list[FileCheckResult] = []
    for f in spec.files:
        path = raw_dir / f.filename
        if not path.exists():
            results.append(FileCheckResult(file=f, path=path, status="missing", actual_sha256=None))
            continue
        actual = sha256_file(path)
        if f.expected_sha256 is None:
            status: ChecksumStatus = "unpinned"
        elif actual == f.expected_sha256:
            status = "match"
        else:
            status = "mismatch"
        results.append(FileCheckResult(file=f, path=path, status=status, actual_sha256=actual))
    return FetchReport(dataset=spec, raw_dir=raw_dir, results=tuple(results))


def fetch_dataset(name: str, raw_dir: Path | str) -> FetchReport:
    """Verify a dataset's placed files, refusing to proceed on mismatch.

    Does not download anything. Raises :class:`DatasetChecksumMismatchError`
    if any placed file's digest contradicts a pinned checksum in the
    registry. Missing files are reported, not raised on, so callers (e.g.
    the CLI) can print next steps instead of a traceback.
    """
    report = verify_dataset_files(name, raw_dir)
    if report.any_mismatch:
        bad = [r for r in report.results if r.status == "mismatch"]
        detail = "; ".join(
            f"{r.file.filename}: expected {r.file.expected_sha256}, got {r.actual_sha256}"
            for r in bad
        )
        raise DatasetChecksumMismatchError(
            f"Checksum mismatch for dataset {name!r}: {detail}. Refusing to proceed — "
            "re-download the affected file(s) rather than trusting a corrupted copy."
        )
    return report


__all__ = [
    "ChecksumStatus",
    "FileCheckResult",
    "FetchReport",
    "DatasetChecksumMismatchError",
    "format_fetch_instructions",
    "verify_dataset_files",
    "fetch_dataset",
]
