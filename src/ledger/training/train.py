"""Phase 3 training orchestration: prepare -> feature pipeline -> fit ->
evaluate -> write model/metrics/curves/manifest to ``runs/``.

Reuses Phase 2's pipeline end to end (:func:`ledger.data.prepare.prepare_dataset`
for fetch-check/adapt/validate/graph/split, and
:class:`ledger.features.pipeline.FeaturePipeline` for features) rather than
re-deriving any of it, so a training run's split and features are always
exactly what ``ledger data prepare`` would have produced for the same
config — never a stale, possibly-mismatched artifact from an earlier run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledger.config.settings import Settings
from ledger.data.canonical import CanonicalDataset
from ledger.data.prepare import prepare_dataset
from ledger.evaluation.curves import save_pr_curve
from ledger.evaluation.labels import build_binary_target
from ledger.evaluation.metrics import compute_metrics
from ledger.features.pipeline import FeaturePipeline
from ledger.models.registry import get_detector
from ledger.utils.manifest import RunManifest
from ledger.utils.seeding import seed_everything


@dataclass(frozen=True)
class TrainResult:
    """Paths and headline metrics from one training run."""

    run_id: str
    dataset_name: str
    model_path: Path
    feature_pipeline_path: Path
    metrics_path: Path
    pr_curve_path: Path
    feature_importance_path: Path
    manifest_path: Path
    test_metrics: dict[str, Any]


def _labeled_subset(ids: list[str], y_index: set[str]) -> list[str]:
    return [i for i in ids if i in y_index]


def train_detector(settings: Settings) -> TrainResult:
    """Run the full Phase 3 pipeline for ``settings.data.dataset_name`` /
    ``settings.model.name``.

    Raises :class:`NotImplementedError` if the configured model or dataset
    is not yet supported (see :mod:`ledger.models.registry` and
    :mod:`ledger.evaluation.labels`), and :class:`ValueError` if a split
    ends up with no labeled examples — never a fabricated metric.
    """
    seed_everything(settings.model.random_seed)

    prepare_result = prepare_dataset(settings)
    canonical = CanonicalDataset.read(
        settings.paths.data_processed_dir, prepare_result.dataset_name
    )

    with prepare_result.split_ids_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    y_all = build_binary_target(canonical)
    y_index = set(y_all.index)

    train_ids = _labeled_subset(split_ids["train"], y_index)
    val_ids = _labeled_subset(split_ids["val"], y_index)
    test_ids = _labeled_subset(split_ids["test"], y_index)
    if not train_ids:
        raise ValueError(f"dataset {canonical.name!r} train split has no labeled examples")
    if not test_ids:
        raise ValueError(f"dataset {canonical.name!r} test split has no labeled examples")

    pipeline = FeaturePipeline(config=settings.features)
    pipeline.fit(canonical, train_account_ids=train_ids, held_out_account_ids=val_ids + test_ids)

    X_train = pipeline.transform(canonical, account_ids=train_ids).set_index("account_id")
    X_test = pipeline.transform(canonical, account_ids=test_ids).set_index("account_id")
    y_train = y_all.loc[X_train.index]
    y_test = y_all.loc[X_test.index]

    detector = get_detector(settings.model)
    detector.fit(X_train, y_train)

    y_score_test = detector.predict_proba(X_test)
    test_metrics = compute_metrics(
        y_test.to_numpy(), y_score_test, settings.evaluation, settings.training
    )

    val_metrics = None
    if val_ids:
        X_val = pipeline.transform(canonical, account_ids=val_ids).set_index("account_id")
        y_val = y_all.loc[X_val.index]
        y_score_val = detector.predict_proba(X_val)
        val_metrics = compute_metrics(
            y_val.to_numpy(), y_score_val, settings.evaluation, settings.training
        )

    run_id = f"train_{settings.model.name}_{canonical.name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_manifest = RunManifest.create(
        run_id=run_id,
        config_hash=settings.hash(),
        dataset_checksums={},
        seed=settings.model.random_seed,
        repo_root=settings.paths.project_root,
    )
    manifest_path = run_manifest.write(settings.paths.runs_dir)
    run_dir = manifest_path.parent

    model_path = detector.save(run_dir / "model.joblib")
    feature_pipeline_path = pipeline.save(run_dir / "feature_pipeline.joblib")

    metrics_payload = {
        "dataset_name": canonical.name,
        "model_metadata": detector.metadata(),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "test": test_metrics,
        "val": val_metrics,
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    pr_curve_path = save_pr_curve(y_test.to_numpy(), y_score_test, run_dir / "pr_curve_test.png")

    feature_importance_path = run_dir / "feature_importance.json"
    with feature_importance_path.open("w", encoding="utf-8") as f:
        json.dump(detector.feature_importance().to_dict(), f, indent=2)

    return TrainResult(
        run_id=run_id,
        dataset_name=canonical.name,
        model_path=model_path,
        feature_pipeline_path=feature_pipeline_path,
        metrics_path=metrics_path,
        pr_curve_path=pr_curve_path,
        feature_importance_path=feature_importance_path,
        manifest_path=manifest_path,
        test_metrics=test_metrics,
    )


__all__ = ["TrainResult", "train_detector"]
