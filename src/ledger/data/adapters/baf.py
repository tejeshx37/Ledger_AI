"""Adapter for the NeurIPS 2022 Bank Account Fraud dataset (Kaggle:
sgpjesus/bank-account-fraud-dataset-neurips-2022, ``Base.csv``).

BAF is account-*application*-level, not transaction-level: each row is one
account-opening decision, so this adapter produces an empty (but
schema-conformant) ``transactions`` table — there is nothing to populate it
with — and puts every feature column except the label itself into each
account's ``attributes`` JSON.

The release does not carry a full calendar date, only a relative ``month``
in ``[0, 7]``; canonical ``opened_at`` here is a deterministic function of
it (epoch + month * 30 days), not a real date. ``customer_age`` is kept as
an ordinary attribute rather than promoted to its own canonical column —
LEDGER's schema has no protected-attribute slot — but it is exactly the
column the dataset's own fairness benchmark uses as its protected
attribute, and Phase 7's bias audit reads it back out of ``attributes``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset, json_text

_EPOCH = pd.Timestamp("1970-01-01")

_RAW_SCHEMA = DataFrameSchema(
    {
        "fraud_bool": Column(int, Check.isin([0, 1]), nullable=False),
        "customer_age": Column(int, nullable=False),
        "month": Column(int, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
)

_LABEL = {0: "not_fraud", 1: "fraud"}


class BafAdapter(DatasetAdapter):
    name = "baf"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(raw_dir / "Base.csv")
        df = validate_raw(df, _RAW_SCHEMA, "Base.csv")
        return {"applications": df}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        df = raw["applications"].reset_index(drop=True)
        account_id = [f"baf_{i}" for i in range(len(df))]

        feature_cols = [c for c in df.columns if c != "fraud_bool"]
        feature_records = df[feature_cols].to_dict("records")
        attributes = [json_text(rec) for rec in feature_records]

        accounts = pd.DataFrame(
            {
                "account_id": account_id,
                "institution_id": pd.array([None] * len(df), dtype="str"),
                "opened_at": [_EPOCH + pd.Timedelta(days=30 * int(m)) for m in df["month"]],
                "attributes": attributes,
            }
        )

        transactions = pd.DataFrame(
            {
                "tx_id": pd.array([], dtype="str"),
                "src_account": pd.array([], dtype="str"),
                "dst_account": pd.array([], dtype="str"),
                "amount": pd.array([], dtype="float64"),
                "currency": pd.array([], dtype="str"),
                "timestamp": pd.array([], dtype="datetime64[ns]"),
                "channel": pd.array([], dtype="str"),
                "institution_src": pd.array([], dtype="str"),
                "institution_dst": pd.array([], dtype="str"),
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": account_id,
                "entity_type": "account",
                "label": df["fraud_bool"].map(_LABEL),
                "label_source": "baf_Base",
            }
        )

        rings = pd.DataFrame(
            columns=["ring_id", "member_accounts", "pattern_type", "time_window"]
        ).astype(
            {
                "ring_id": str,
                "member_accounts": str,
                "pattern_type": str,
                "time_window": str,
            }
        )

        return CanonicalDataset(
            name=self.name,
            accounts=accounts,
            transactions=transactions,
            labels=labels,
            rings=rings,
        )


__all__ = ["BafAdapter"]
