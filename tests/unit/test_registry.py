"""Unit tests for the dataset registry."""

from __future__ import annotations

import pytest

from ledger.data.registry import DATASET_REGISTRY, get_dataset_spec

EXPECTED_DATASETS = {"elliptic", "elliptic_pp", "ibm_aml", "paysim", "baf", "ieee_cis"}


def test_registry_has_every_expected_dataset() -> None:
    assert set(DATASET_REGISTRY) == EXPECTED_DATASETS


def test_get_dataset_spec_returns_matching_name() -> None:
    spec = get_dataset_spec("elliptic")
    assert spec.name == "elliptic"
    assert len(spec.files) == 3


def test_get_dataset_spec_unknown_raises_with_registered_list() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset_spec("not_a_real_dataset")


def test_every_dataset_has_at_least_one_expected_file() -> None:
    for name, spec in DATASET_REGISTRY.items():
        assert len(spec.files) >= 1, f"{name} has no expected files"


def test_every_dataset_has_role_license_and_instructions() -> None:
    for name, spec in DATASET_REGISTRY.items():
        assert spec.role, f"{name} missing role"
        assert spec.license_name, f"{name} missing license_name"
        assert spec.fetch_instructions, f"{name} missing fetch_instructions"
