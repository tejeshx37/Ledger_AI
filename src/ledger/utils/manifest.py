"""Run manifests: the audit trail that makes every LEDGER result traceable.

Every experiment or scoring run writes a ``RunManifest`` to
``runs/<run_id>/manifest.json`` before doing any real work. Given only that
file, someone should be able to reconstruct exactly what produced a given
metric: which code (git SHA, dirty flag), which configuration (hash), which
data (per-file checksums), which seed, and which library versions.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_TRACKED_PACKAGES = (
    "torch",
    "torch-geometric",
    "xgboost",
    "scikit-learn",
    "flwr",
    "pydantic",
    "fastapi",
    "sqlalchemy",
    "numpy",
    "pandas",
)


def _git_sha(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty(cwd: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to reproduce and audit a single run.

    Construct via :meth:`create`, which fills in everything derivable from
    the environment (git state, hostname, timestamp, package versions);
    the caller only supplies run-specific identifiers.
    """

    run_id: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    dataset_checksums: dict[str, str]
    seed: int
    hostname: str
    timestamp: str
    python_version: str
    package_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_id: str,
        config_hash: str,
        dataset_checksums: dict[str, str],
        seed: int,
        repo_root: Path | str = Path("."),
    ) -> RunManifest:
        """Build a manifest, deriving git/host/package state automatically.

        Args:
            run_id: caller-assigned unique identifier for this run.
            config_hash: output of ``Settings.hash()`` for the config that
                governs this run.
            dataset_checksums: mapping of dataset file path -> SHA-256, as
                produced by :func:`ledger.utils.checksums.sha256_file`.
            seed: the seed passed to ``seed_everything`` for this run.
            repo_root: repository root used to resolve git state from.
        """
        repo_root = Path(repo_root)
        return cls(
            run_id=run_id,
            git_sha=_git_sha(repo_root),
            git_dirty=_git_dirty(repo_root),
            config_hash=config_hash,
            dataset_checksums=dict(dataset_checksums),
            seed=seed,
            hostname=socket.gethostname(),
            timestamp=datetime.now(UTC).isoformat(),
            python_version=platform.python_version(),
            package_versions=_package_versions(),
        )

    def write(self, runs_dir: Path | str) -> Path:
        """Serialise to ``<runs_dir>/<run_id>/manifest.json`` and return the path."""
        run_dir = Path(runs_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")
        return manifest_path

    @classmethod
    def read(cls, manifest_path: Path | str) -> RunManifest:
        """Load a previously written manifest back into a ``RunManifest``."""
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)
