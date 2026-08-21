"""Phase 2 orchestration: fetch-check -> adapt -> validate -> graph -> split
-> manifest.

``ledger data prepare --dataset <name>`` is the Phase 2 deliverable: it
turns a dataset's raw files (already placed on disk per
``ledger data fetch``) into validated canonical tables, a graph artifact,
and a split, all tied together by one :class:`~ledger.utils.manifest.RunManifest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from ledger.config.settings import Settings
from ledger.data.adapters import get_adapter
from ledger.data.fetch import fetch_dataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.data.splitting import split_dataset
from ledger.utils.manifest import RunManifest


@dataclass(frozen=True)
class PrepareResult:
    """Paths to everything a Phase 2 prepare run produced."""

    dataset_name: str
    run_id: str
    canonical_dir: Path
    graph_path: Path
    node_statistics_path: Path
    split_manifest_path: Path
    split_ids_path: Path
    manifest_path: Path


def prepare_dataset(settings: Settings, dataset_name: str | None = None) -> PrepareResult:
    """Run the full Phase 2 pipeline for one dataset.

    Raises whatever :func:`ledger.data.fetch.fetch_dataset` raises if a
    placed file fails a pinned checksum, and :class:`FileNotFoundError` if
    any expected raw file is simply absent — this never proceeds on an
    unverified or missing download.
    """
    name = dataset_name or settings.data.dataset_name
    raw_dir = settings.paths.data_raw_dir / name
    fetch_report = fetch_dataset(name, raw_dir)
    if not fetch_report.all_present:
        missing = [r.file.filename for r in fetch_report.results if r.status == "missing"]
        raise FileNotFoundError(
            f"Cannot prepare dataset {name!r}: missing file(s) {missing} under {raw_dir}. "
            f"Run `ledger data fetch {name}` for acquisition instructions."
        )

    adapter = get_adapter(name)
    canonical = adapter.run(raw_dir)
    canonical_dir = canonical.write(settings.paths.data_processed_dir)

    graph_builder = TemporalGraphBuilder(canonical)
    graph_path = settings.paths.data_processed_dir / name / "graph.joblib"
    joblib.dump(graph_builder.build_static_graph(), graph_path)

    node_stats = graph_builder.node_statistics()
    node_statistics_path = settings.paths.data_processed_dir / name / "node_statistics.parquet"
    node_stats.to_parquet(node_statistics_path, index=False)

    (train_ids, val_ids, test_ids), split_manifest = split_dataset(
        canonical, settings.data.split, seed=settings.model.random_seed
    )
    split_manifest_path = settings.paths.data_processed_dir / name / "split_manifest.json"
    split_manifest.write(split_manifest_path)

    split_ids_path = settings.paths.data_processed_dir / name / "split_ids.json"
    with split_ids_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_ids.tolist(),
                "val": val_ids.tolist(),
                "test": test_ids.tolist(),
            },
            f,
            indent=2,
        )

    dataset_checksums = {
        r.file.filename: r.actual_sha256
        for r in fetch_report.results
        if r.actual_sha256 is not None
    }
    run_id = f"prepare_{name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_manifest = RunManifest.create(
        run_id=run_id,
        config_hash=settings.hash(),
        dataset_checksums=dataset_checksums,
        seed=settings.model.random_seed,
        repo_root=settings.paths.project_root,
    )
    manifest_path = run_manifest.write(settings.paths.runs_dir)

    return PrepareResult(
        dataset_name=name,
        run_id=run_id,
        canonical_dir=canonical_dir,
        graph_path=graph_path,
        node_statistics_path=node_statistics_path,
        split_manifest_path=split_manifest_path,
        split_ids_path=split_ids_path,
        manifest_path=manifest_path,
    )


__all__ = ["PrepareResult", "prepare_dataset"]
