"""Unit tests for the canonical schema, validation, and referential integrity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ledger.data.canonical import (
    CanonicalDataset,
    CanonicalIntegrityError,
    CanonicalSchemaError,
    json_text,
)


def _empty_rings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ring_id", "member_accounts", "pattern_type", "time_window"]
    ).astype({"ring_id": str, "member_accounts": str, "pattern_type": str, "time_window": str})


def _valid_dataset() -> CanonicalDataset:
    accounts = pd.DataFrame(
        {
            "account_id": ["a1", "a2", "a3"],
            "institution_id": ["bank1", "bank1", None],
            "opened_at": pd.to_datetime(["2020-01-01", "2020-01-02", pd.NaT]),
            "attributes": [json_text({"x": 1}), json_text({"x": 2}), json_text({})],
        }
    )
    transactions = pd.DataFrame(
        {
            "tx_id": ["t1", "t2"],
            "src_account": ["a1", "a2"],
            "dst_account": ["a2", "a3"],
            "amount": [100.0, None],
            "currency": ["USD", "USD"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "channel": ["wire", "ach"],
            "institution_src": ["bank1", "bank1"],
            "institution_dst": ["bank1", None],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": ["t1", "a1"],
            "entity_type": ["tx", "account"],
            "label": ["laundering", "not_fraud"],
            "label_source": ["test", "test"],
        }
    )
    rings = pd.DataFrame(
        {
            "ring_id": ["r1"],
            "member_accounts": [json_text(["a1", "a2"])],
            "pattern_type": ["cycle"],
            "time_window": [json_text(["2020-01-01", "2020-01-02"])],
        }
    )
    return CanonicalDataset(
        name="test", accounts=accounts, transactions=transactions, labels=labels, rings=rings
    )


def test_valid_dataset_passes_validation() -> None:
    ds = _valid_dataset()
    validated = ds.validate()
    assert validated is ds


def test_schema_rejects_negative_amount() -> None:
    ds = _valid_dataset()
    ds.transactions.loc[0, "amount"] = -5.0
    with pytest.raises(CanonicalSchemaError) as exc_info:
        ds.validate()
    assert exc_info.value.table == "transactions"
    assert len(exc_info.value.failure_cases) >= 1


def test_schema_rejects_bad_entity_type() -> None:
    ds = _valid_dataset()
    ds.labels.loc[0, "entity_type"] = "not_a_real_type"
    with pytest.raises(CanonicalSchemaError):
        ds.validate()


def test_schema_rejects_duplicate_account_id() -> None:
    ds = _valid_dataset()
    ds.accounts.loc[1, "account_id"] = "a1"
    with pytest.raises(CanonicalSchemaError):
        ds.validate()


def test_integrity_rejects_unknown_src_account() -> None:
    ds = _valid_dataset()
    ds.transactions.loc[0, "src_account"] = "unknown_account"
    with pytest.raises(CanonicalIntegrityError, match="src_account"):
        ds.validate()


def test_integrity_rejects_unknown_dst_account() -> None:
    ds = _valid_dataset()
    ds.transactions.loc[0, "dst_account"] = "unknown_account"
    with pytest.raises(CanonicalIntegrityError, match="dst_account"):
        ds.validate()


def test_integrity_rejects_label_referencing_unknown_tx() -> None:
    ds = _valid_dataset()
    ds.labels.loc[0, "entity_id"] = "unknown_tx"
    with pytest.raises(CanonicalIntegrityError, match="tx id"):
        ds.validate()


def test_integrity_rejects_label_referencing_unknown_account() -> None:
    ds = _valid_dataset()
    ds.labels.loc[1, "entity_id"] = "unknown_account"
    with pytest.raises(CanonicalIntegrityError, match="account id"):
        ds.validate()


def test_integrity_rejects_ring_with_unknown_member() -> None:
    ds = _valid_dataset()
    ds.rings.loc[0, "member_accounts"] = json_text(["a1", "not_a_real_account"])
    with pytest.raises(CanonicalIntegrityError, match="member_accounts"):
        ds.validate()


def test_write_read_roundtrip(tmp_path: Path) -> None:
    ds = _valid_dataset()
    ds.validate()
    out_dir = ds.write(tmp_path)
    assert out_dir == tmp_path / "test"
    assert (out_dir / "accounts.parquet").exists()

    reloaded = CanonicalDataset.read(tmp_path, "test")
    assert reloaded.accounts["account_id"].tolist() == ds.accounts["account_id"].tolist()
    assert reloaded.transactions["tx_id"].tolist() == ds.transactions["tx_id"].tolist()
    reloaded.validate()


def test_json_text_is_deterministic_for_equal_dicts() -> None:
    assert json_text({"b": 2, "a": 1}) == json_text({"a": 1, "b": 2})
