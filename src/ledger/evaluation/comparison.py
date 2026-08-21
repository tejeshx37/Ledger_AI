"""Phase 4.4: comparison protocol — baseline vs GraphSAGE vs TGAT on
identical splits, seeds, and (where applicable) features, broken down by
ring-participating accounts versus isolated suspicious accounts.

That breakdown is the project's core scientific claim: a graph model's
advantage should show up specifically in the ring column, not uniformly
everywhere. No dataset LEDGER has an adapter for ships an explicit
ring/typology label (Phase 2's canonical ``rings`` table is empty for
every adapter — see :mod:`ledger.data.adapters`), so "ring-participating"
here is an honest, documented proxy rather than an invented one:
membership in *some* directed cycle of the transaction graph
(``motif_cycle_participation``, a real structural computation from
Phase 2's :func:`ledger.features.motif_features.compute_motif_features`,
not a fabricated label).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy.typing as npt
import pandas as pd

from ledger.config.settings import Settings, load_settings_from_overlay_path
from ledger.data.canonical import CanonicalDataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.evaluation.labels import build_binary_target
from ledger.evaluation.metrics import compute_metrics
from ledger.features.motif_features import compute_motif_features
from ledger.features.pipeline import FeaturePipeline
from ledger.models.registry import MODEL_REGISTRY
from ledger.training.train import TrainResult, train_detector
from ledger.utils.manifest import RunManifest

DEFAULT_MODEL_OVERLAYS = (
    "model_baseline.yaml",
    "model_graphsage.yaml",
    "model_tgat.yaml",
)

_CSV_COLUMNS = (
    "model_name",
    "architecture",
    "n_test",
    "n_ring",
    "n_isolated",
    "overall_precision",
    "overall_recall",
    "overall_roc_auc",
    "overall_average_precision",
    "ring_recall",
    "ring_roc_auc",
    "isolated_recall",
    "isolated_roc_auc",
)


@dataclass(frozen=True)
class ComparisonResult:
    """Paths and the assembled table from one comparison run."""

    run_id: str
    dataset_name: str
    table_path: Path
    table_csv_path: Path
    manifest_path: Path
    per_model_runs: dict[str, TrainResult]
    table: list[dict[str, Any]]


def _safe_metrics(
    y_true: npt.NDArray[Any],
    y_score: npt.NDArray[Any],
    evaluation_config: Any,
    training_config: Any,
) -> dict[str, Any]:
    """compute_metrics(), degrading to a count-only summary if the subset
    has only one class present (ring/isolated subsets can be small enough
    that this happens) rather than raising and aborting the whole table.
    """
    try:
        return compute_metrics(y_true, y_score, evaluation_config, training_config)
    except ValueError as exc:
        return {
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()) if len(y_true) else 0,
            "note": f"metrics unavailable: {exc}",
        }


def _select_scores(
    index: pd.Index, y_score: npt.NDArray[Any], wanted_ids: list[str]
) -> npt.NDArray[Any]:
    """Select the entries of ``y_score`` (aligned to ``index``, in order)
    corresponding to ``wanted_ids``.
    """
    result: npt.NDArray[Any] = pd.Series(y_score, index=index).loc[wanted_ids].to_numpy()
    return result


def run_model_comparison(overlay_paths: list[Path]) -> ComparisonResult:
    """Train every model named in ``overlay_paths`` (each independently
    loaded via :func:`~ledger.config.settings.load_settings_from_overlay_path`,
    so they all share whatever ``base.yaml`` sits alongside them — hence
    identical dataset, split, and seed) and build the comparison table.

    Raises :class:`ValueError` if fewer than two configs are given, if they
    target different datasets, or if — despite sharing a base config — they
    somehow resolved to different test splits (a defensive check: the
    "identical splits" guarantee is exactly what makes the resulting table
    meaningful).
    """
    if len(overlay_paths) < 2:
        raise ValueError("a comparison requires at least two model configs")

    # Resolve and validate every config up front — before training any of
    # them — so a mismatched dataset is reported immediately rather than
    # after burning time training on the first (correct) overlay.
    all_settings = [load_settings_from_overlay_path(p) for p in overlay_paths]
    dataset_name = all_settings[0].data.dataset_name
    for overlay_path, settings in zip(overlay_paths, all_settings, strict=True):
        if settings.data.dataset_name != dataset_name:
            raise ValueError(
                f"all model configs in a comparison must target the same dataset; "
                f"{overlay_path} targets {settings.data.dataset_name!r}, expected "
                f"{dataset_name!r}"
            )

    per_model_results: dict[str, TrainResult] = {}
    per_model_settings: dict[str, Settings] = {}
    for settings in all_settings:
        result = train_detector(settings)
        per_model_results[settings.model.name] = result
        per_model_settings[settings.model.name] = settings

    test_id_sets = {frozenset(r.test_account_ids) for r in per_model_results.values()}
    if len(test_id_sets) > 1:
        raise ValueError(
            "models resolved to different test splits despite sharing a base config — "
            "the comparison would not be apples-to-apples"
        )

    first_settings = next(iter(per_model_settings.values()))
    canonical = CanonicalDataset.read(first_settings.paths.data_processed_dir, dataset_name)
    y_all = build_binary_target(canonical)

    graph = TemporalGraphBuilder(canonical).build_static_graph()
    ring_membership = compute_motif_features(graph, first_settings.features).set_index(
        "account_id"
    )["motif_cycle_participation"]

    table: list[dict[str, Any]] = []
    for model_name, result in per_model_results.items():
        settings = per_model_settings[model_name]
        pipeline = FeaturePipeline.load(result.feature_pipeline_path)
        detector = MODEL_REGISTRY[model_name].load(result.model_path)

        test_ids = result.test_account_ids
        X_test = pipeline.transform(canonical, account_ids=test_ids).set_index("account_id")
        y_test = y_all.loc[X_test.index]
        y_score = detector.predict_proba(X_test)

        ring_mask = ring_membership.reindex(X_test.index).fillna(0).astype(int) == 1
        ring_ids = list(X_test.index[ring_mask])
        isolated_ids = list(X_test.index[~ring_mask])

        overall = _safe_metrics(y_test.to_numpy(), y_score, settings.evaluation, settings.training)
        ring = _safe_metrics(
            y_test.loc[ring_ids].to_numpy(),
            _select_scores(X_test.index, y_score, ring_ids),
            settings.evaluation,
            settings.training,
        )
        isolated = _safe_metrics(
            y_test.loc[isolated_ids].to_numpy(),
            _select_scores(X_test.index, y_score, isolated_ids),
            settings.evaluation,
            settings.training,
        )

        table.append(
            {
                "model_name": model_name,
                "architecture": detector.metadata().get("architecture", model_name),
                "n_test": len(test_ids),
                "n_ring": len(ring_ids),
                "n_isolated": len(isolated_ids),
                "overall": overall,
                "ring": ring,
                "isolated": isolated,
            }
        )

    run_id = f"compare_{dataset_name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_manifest = RunManifest.create(
        run_id=run_id,
        config_hash=first_settings.hash(),
        dataset_checksums={},
        seed=first_settings.model.random_seed,
        repo_root=first_settings.paths.project_root,
    )
    manifest_path = run_manifest.write(first_settings.paths.runs_dir)
    run_dir = manifest_path.parent

    table_path = run_dir / "comparison_table.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    table_csv_path = run_dir / "comparison_table.csv"
    _write_table_csv(table_csv_path, table)

    return ComparisonResult(
        run_id=run_id,
        dataset_name=dataset_name,
        table_path=table_path,
        table_csv_path=table_csv_path,
        manifest_path=manifest_path,
        per_model_runs=per_model_results,
        table=table,
    )


def _write_table_csv(path: Path, table: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in table:
            overall = row["overall"]
            ring = row["ring"]
            isolated = row["isolated"]
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "architecture": row["architecture"],
                    "n_test": row["n_test"],
                    "n_ring": row["n_ring"],
                    "n_isolated": row["n_isolated"],
                    "overall_precision": overall.get("precision"),
                    "overall_recall": overall.get("recall"),
                    "overall_roc_auc": overall.get("roc_auc"),
                    "overall_average_precision": overall.get("average_precision"),
                    "ring_recall": ring.get("recall"),
                    "ring_roc_auc": ring.get("roc_auc"),
                    "isolated_recall": isolated.get("recall"),
                    "isolated_roc_auc": isolated.get("roc_auc"),
                }
            )


__all__ = ["ComparisonResult", "DEFAULT_MODEL_OVERLAYS", "run_model_comparison"]
