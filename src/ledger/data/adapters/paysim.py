"""Adapter for the PaySim mobile-money simulation (Kaggle: ealaxi/paysim1).

PaySim's ``step`` column is a synthetic hour offset (1 step = 1 simulated
hour over a 30-day run), not a real timestamp; canonical timestamps here
are a deterministic function of it (epoch + step hours), used purely so
temporal splitting and graph snapshots have something to order on.

PaySim's ``oldbalance*``/``newbalance*`` columns describe account balances
at the moment of each transaction, not a static per-account attribute, and
LEDGER's canonical ``accounts`` table has no per-timestep slot for them —
they are intentionally dropped here rather than forced into a schema that
does not fit them. ``isFlaggedFraud`` (the source simulation's own
rule-based flag) is likewise dropped: it is a *system output*, not a label
of ground truth, and mixing it into ``labels`` would contaminate the
target this project measures against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset

_EPOCH = pd.Timestamp("1970-01-01")

_RAW_SCHEMA = DataFrameSchema(
    {
        "step": Column(int, Check.ge(0), nullable=False),
        "type": Column(str, nullable=False),
        "amount": Column(float, Check.ge(0.0), nullable=False),
        "nameOrig": Column(str, nullable=False),
        "nameDest": Column(str, nullable=False),
        "isFraud": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=True,
)

_LABEL = {0: "not_fraud", 1: "fraud"}


class PaysimAdapter(DatasetAdapter):
    name = "paysim"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(raw_dir / "PS_20174392719_1491204439457_log.csv")
        df["nameOrig"] = df["nameOrig"].astype(str)
        df["nameDest"] = df["nameDest"].astype(str)
        df = validate_raw(df, _RAW_SCHEMA, "PS_20174392719_1491204439457_log.csv")
        return {"transactions": df}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        df = raw["transactions"]

        account_ids = pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True).unique()
        accounts = pd.DataFrame(
            {
                "account_id": account_ids,
                "institution_id": pd.array([None] * len(account_ids), dtype="str"),
                "opened_at": pd.NaT,
                "attributes": "{}",
            }
        )

        tx_id = [f"paysim_{i}" for i in range(len(df))]
        transactions = pd.DataFrame(
            {
                "tx_id": tx_id,
                "src_account": df["nameOrig"],
                "dst_account": df["nameDest"],
                "amount": df["amount"],
                "currency": "unknown",
                "timestamp": [_EPOCH + pd.Timedelta(hours=int(s)) for s in df["step"]],
                "channel": df["type"],
                "institution_src": pd.array([None] * len(df), dtype="str"),
                "institution_dst": pd.array([None] * len(df), dtype="str"),
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": tx_id,
                "entity_type": "tx",
                "label": df["isFraud"].map(_LABEL),
                "label_source": "paysim",
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


__all__ = ["PaysimAdapter"]
