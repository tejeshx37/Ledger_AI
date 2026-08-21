"""Unit tests for temporal/random/institution splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.config.models import SplitConfig
from ledger.data.adapters import get_adapter
from ledger.data.canonical import CanonicalDataset
from ledger.data.splitting import SplitManifest, split_dataset, split_temporal


@pytest.fixture
def elliptic_dataset(elliptic_raw_dir: Path) -> CanonicalDataset:
    return get_adapter("elliptic").run(elliptic_raw_dir)


@pytest.fixture
def ibm_aml_dataset(ibm_aml_raw_dir: Path) -> CanonicalDataset:
    return get_adapter("ibm_aml").run(ibm_aml_raw_dir)


@pytest.fixture
def elliptic_pp_dataset(elliptic_pp_raw_dir: Path) -> CanonicalDataset:
    return get_adapter("elliptic_pp").run(elliptic_pp_raw_dir)


def _assert_disjoint_and_covers(train, val, test, total_n: int) -> None:
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    assert len(train) + len(val) + len(test) == total_n


def test_split_temporal_default_fractions(elliptic_dataset: CanonicalDataset) -> None:
    (train, val, test), manifest = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    _assert_disjoint_and_covers(train, val, test, len(elliptic_dataset.labels))
    assert manifest.strategy == "temporal"
    assert manifest.boundaries["train_end"] is not None
    assert manifest.n_train == len(train)


def test_split_temporal_is_deterministic(elliptic_dataset: CanonicalDataset) -> None:
    (train1, val1, test1), _ = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    (train2, val2, test2), _ = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    assert list(train1) == list(train2)
    assert list(val1) == list(val2)
    assert list(test1) == list(test2)


def test_split_temporal_raises_on_missing_timestamp(elliptic_pp_dataset: CanonicalDataset) -> None:
    # elliptic_pp accounts have no known opened_at
    with pytest.raises(ValueError, match="known timestamp"):
        split_dataset(elliptic_pp_dataset, SplitConfig(strategy="temporal"))


def test_split_random_is_seed_reproducible(elliptic_dataset: CanonicalDataset) -> None:
    (train1, val1, test1), m1 = split_dataset(
        elliptic_dataset, SplitConfig(strategy="random"), seed=123
    )
    (train2, val2, test2), m2 = split_dataset(
        elliptic_dataset, SplitConfig(strategy="random"), seed=123
    )
    assert list(train1) == list(train2)
    assert m1.seed == m2.seed == 123
    _assert_disjoint_and_covers(train1, val1, test1, len(elliptic_dataset.labels))


def test_split_random_different_seeds_differ(elliptic_dataset: CanonicalDataset) -> None:
    (train1, _, _), _ = split_dataset(elliptic_dataset, SplitConfig(strategy="random"), seed=1)
    (train2, _, _), _ = split_dataset(elliptic_dataset, SplitConfig(strategy="random"), seed=2)
    assert list(train1) != list(train2)


def test_split_institution_holds_out_bank(ibm_aml_dataset: CanonicalDataset) -> None:
    config = SplitConfig(strategy="institution", held_out_institution_id="2")
    (train, val, test), manifest = split_dataset(ibm_aml_dataset, config, seed=0)
    _assert_disjoint_and_covers(train, val, test, len(ibm_aml_dataset.labels))
    assert manifest.strategy == "institution"
    assert manifest.boundaries["held_out_institution_id"] == "2"
    assert manifest.n_test == len(test)


def test_split_institution_unknown_id_raises(ibm_aml_dataset: CanonicalDataset) -> None:
    config = SplitConfig(strategy="institution", held_out_institution_id="does_not_exist")
    with pytest.raises(ValueError, match="matches no entities"):
        split_dataset(ibm_aml_dataset, config, seed=0)


def test_split_institution_requires_no_institution_data_raises(
    elliptic_dataset: CanonicalDataset,
) -> None:
    config = SplitConfig(strategy="institution", held_out_institution_id="anything")
    with pytest.raises(ValueError, match="known institution_id"):
        split_dataset(elliptic_dataset, config, seed=0)


def test_split_manifest_class_balance_sums_to_one(elliptic_dataset: CanonicalDataset) -> None:
    entities = elliptic_dataset.labels
    del entities
    (train, val, test), manifest = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    for fold in ("train", "val", "test"):
        balance = manifest.class_balance[fold]
        if balance:
            assert abs(sum(balance.values()) - 1.0) < 1e-9


def test_split_manifest_write_read_roundtrip(
    tmp_path: Path, elliptic_dataset: CanonicalDataset
) -> None:
    _, manifest = split_dataset(elliptic_dataset, SplitConfig(strategy="temporal"))
    path = manifest.write(tmp_path / "split_manifest.json")
    reloaded = SplitManifest.read(path)
    assert reloaded == manifest


def test_split_temporal_function_directly(elliptic_dataset: CanonicalDataset) -> None:
    from ledger.data.splitting import build_entity_frame

    entities = build_entity_frame(elliptic_dataset)
    (train, val, test), manifest = split_temporal(
        entities, SplitConfig(strategy="temporal"), "elliptic"
    )
    assert manifest.dataset_name == "elliptic"
    _assert_disjoint_and_covers(train, val, test, len(entities))
