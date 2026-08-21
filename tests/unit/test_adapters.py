"""Unit tests for every registered dataset adapter: raw files -> a
canonical dataset that passes its own schema and referential-integrity
validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.data.adapters import ADAPTER_REGISTRY, get_adapter
from ledger.data.adapters.base import RawSchemaError


def test_adapter_registry_covers_every_dataset() -> None:
    from ledger.data.registry import DATASET_REGISTRY

    assert set(ADAPTER_REGISTRY) == set(DATASET_REGISTRY)


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError, match="No adapter registered"):
        get_adapter("not_a_real_dataset")


def test_elliptic_adapter(elliptic_raw_dir: Path) -> None:
    ds = get_adapter("elliptic").run(elliptic_raw_dir)
    assert len(ds.accounts) == 40
    assert len(ds.transactions) == 80
    assert len(ds.labels) == 40
    assert set(ds.labels["label"]).issubset({"illicit", "licit", "unknown"})
    assert ds.accounts["institution_id"].isna().all()
    assert ds.accounts["opened_at"].notna().all()


def test_elliptic_pp_adapter(elliptic_pp_raw_dir: Path) -> None:
    ds = get_adapter("elliptic_pp").run(elliptic_pp_raw_dir)
    assert len(ds.accounts) == 10
    assert len(ds.transactions) == 4
    assert set(ds.labels["label"]).issubset({"illicit", "licit", "unknown"})


def test_ibm_aml_adapter(ibm_aml_raw_dir: Path) -> None:
    ds = get_adapter("ibm_aml").run(ibm_aml_raw_dir)
    assert len(ds.transactions) == 3
    assert set(ds.labels["label"]) == {"laundering", "not_laundering"}
    # this is the only dataset with real per-transaction institution ids
    assert ds.transactions["institution_src"].notna().all()
    assert ds.accounts["institution_id"].notna().all()
    assert set(ds.accounts["account_id"]) == {"1_A1", "2_B1", "3_B2", "2_A2"}


def test_paysim_adapter(paysim_raw_dir: Path) -> None:
    ds = get_adapter("paysim").run(paysim_raw_dir)
    assert len(ds.transactions) == 3
    assert set(ds.labels["label"]) == {"fraud", "not_fraud"}
    assert ds.accounts["institution_id"].isna().all()


def test_baf_adapter(baf_raw_dir: Path) -> None:
    ds = get_adapter("baf").run(baf_raw_dir)
    assert len(ds.accounts) == 3
    assert len(ds.transactions) == 0
    assert set(ds.labels["label"]) == {"fraud", "not_fraud"}
    assert (ds.labels["entity_type"] == "account").all()


def test_ieee_cis_adapter(ieee_cis_raw_dir: Path) -> None:
    ds = get_adapter("ieee_cis").run(ieee_cis_raw_dir)
    assert len(ds.transactions) == 3
    assert set(ds.labels["label"]) == {"fraud", "not_fraud"}
    # card1=1000 appears twice, card1=1001 once, plus 2 distinct ProductCD merchants
    assert set(ds.transactions["src_account"]) == {"card_1000", "card_1001"}


def test_elliptic_adapter_rejects_malformed_features_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "elliptic"
    raw_dir.mkdir(parents=True)
    import pandas as pd

    # class column has an invalid value; edgelist/features left minimal
    pd.DataFrame([[1, 1.0] + [0.0] * 165]).to_csv(
        raw_dir / "elliptic_txs_features.csv", header=False, index=False
    )
    pd.DataFrame({"txId": [1], "class": ["not_a_valid_class"]}).to_csv(
        raw_dir / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame(columns=["txId1", "txId2"]).to_csv(
        raw_dir / "elliptic_txs_edgelist.csv", index=False
    )
    with pytest.raises(RawSchemaError):
        get_adapter("elliptic").run(raw_dir)
