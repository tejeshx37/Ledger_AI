"""Unit tests for the feature engineering pipeline: fit/transform semantics
and the leakage guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.config.models import FeaturesConfig, SplitConfig
from ledger.data.adapters import get_adapter
from ledger.data.canonical import CanonicalDataset
from ledger.data.splitting import split_dataset
from ledger.features.pipeline import (
    FeatureLeakageError,
    FeaturePipeline,
    FeaturePipelineNotFittedError,
)


@pytest.fixture
def elliptic_dataset(elliptic_raw_dir: Path) -> CanonicalDataset:
    return get_adapter("elliptic").run(elliptic_raw_dir)


@pytest.fixture
def elliptic_split(elliptic_dataset: CanonicalDataset):
    (train, val, test), _ = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    return train, val, test


def test_transform_before_fit_raises(elliptic_dataset: CanonicalDataset) -> None:
    pipeline = FeaturePipeline(config=FeaturesConfig())
    with pytest.raises(FeaturePipelineNotFittedError):
        pipeline.transform(elliptic_dataset)


def test_fit_with_overlapping_train_and_held_out_raises(
    elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig())
    with pytest.raises(FeatureLeakageError, match="leakage"):
        pipeline.fit(
            elliptic_dataset, train_account_ids=train, held_out_account_ids=list(train)[:1]
        )


def test_fit_transform_produces_expected_shape(
    elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig())
    pipeline.fit(elliptic_dataset, train_account_ids=train, held_out_account_ids=test)
    transformed = pipeline.transform(elliptic_dataset, account_ids=test)
    assert len(transformed) == len(test)
    assert "account_id" in transformed.columns
    assert any(c.startswith("graph_") for c in transformed.columns)
    assert any(c.startswith("motif_") for c in transformed.columns)
    assert any(c.startswith("raw_") for c in transformed.columns)


def test_transform_without_account_ids_returns_every_account(
    elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig())
    pipeline.fit(elliptic_dataset, train_account_ids=train)
    transformed = pipeline.transform(elliptic_dataset)
    assert len(transformed) == len(elliptic_dataset.accounts)


def test_scaler_none_leaves_values_unscaled(
    elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig(scaler="none"))
    pipeline.fit(elliptic_dataset, train_account_ids=train)
    transformed = pipeline.transform(elliptic_dataset, account_ids=train)
    assert (transformed["motif_fan_in_count"] >= 0).all()


def test_standard_scaler_centers_train_features_near_zero(
    elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, _test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig(scaler="standard"))
    pipeline.fit(elliptic_dataset, train_account_ids=train)
    transformed = pipeline.transform(elliptic_dataset, account_ids=train)
    feature_cols = [c for c in transformed.columns if c != "account_id"]
    means = transformed[feature_cols].mean()
    assert (means.abs() < 1e-6).all()


def test_save_load_roundtrip_produces_identical_output(
    tmp_path: Path, elliptic_dataset: CanonicalDataset, elliptic_split
) -> None:
    train, _val, test = elliptic_split
    pipeline = FeaturePipeline(config=FeaturesConfig())
    pipeline.fit(elliptic_dataset, train_account_ids=train, held_out_account_ids=test)
    before = pipeline.transform(elliptic_dataset, account_ids=test)

    path = pipeline.save(tmp_path / "pipeline.joblib")
    loaded = FeaturePipeline.load(path)
    after = loaded.transform(elliptic_dataset, account_ids=test)

    assert before.equals(after)


def test_load_rejects_non_pipeline_file(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"not": "a pipeline"}, path)
    with pytest.raises(TypeError):
        FeaturePipeline.load(path)


def test_fit_raises_on_empty_train_ids(elliptic_dataset: CanonicalDataset) -> None:
    pipeline = FeaturePipeline(config=FeaturesConfig())
    with pytest.raises(ValueError, match="no rows"):
        pipeline.fit(elliptic_dataset, train_account_ids=["not_a_real_account_id"])
