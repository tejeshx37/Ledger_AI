"""The canonical internal representation every dataset adapter normalises
into.

No code outside :mod:`ledger.data.adapters` should ever depend on a
dataset's native column names or encoding. Everything downstream — graph
construction, feature engineering, splitting, models, serving — sees only
these four tables:

``accounts``
    account_id, institution_id, opened_at, attributes (JSON text)
``transactions``
    tx_id, src_account, dst_account, amount, currency, timestamp, channel,
    institution_src, institution_dst
``labels``
    entity_id, entity_type ('tx' | 'account' | 'ring'), label, label_source
``rings``
    ring_id, member_accounts (JSON array of account_id), pattern_type,
    time_window (JSON 2-element array of ISO-8601 timestamps or nulls)

Nested values (``attributes``, ``member_accounts``, ``time_window``) are
stored as JSON text rather than native Python objects so every column has a
flat, Parquet-friendly dtype and every table round-trips through disk
without a custom serialiser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

ENTITY_TYPES = ("tx", "account", "ring")

ACCOUNTS_SCHEMA = DataFrameSchema(
    {
        "account_id": Column(str, nullable=False),
        "institution_id": Column(str, nullable=True),
        "opened_at": Column("datetime64[ns]", nullable=True),
        "attributes": Column(str, nullable=False),
    },
    unique=["account_id"],
    strict=True,
    coerce=False,
)

TRANSACTIONS_SCHEMA = DataFrameSchema(
    {
        "tx_id": Column(str, nullable=False),
        "src_account": Column(str, nullable=False),
        "dst_account": Column(str, nullable=False),
        "amount": Column(float, Check.ge(0.0), nullable=True),
        "currency": Column(str, nullable=False),
        "timestamp": Column("datetime64[ns]", nullable=True),
        "channel": Column(str, nullable=False),
        "institution_src": Column(str, nullable=True),
        "institution_dst": Column(str, nullable=True),
    },
    unique=["tx_id"],
    strict=True,
    coerce=False,
)

LABELS_SCHEMA = DataFrameSchema(
    {
        "entity_id": Column(str, nullable=False),
        "entity_type": Column(str, Check.isin(ENTITY_TYPES), nullable=False),
        "label": Column(str, nullable=False),
        "label_source": Column(str, nullable=False),
    },
    unique=["entity_id", "entity_type", "label_source"],
    strict=True,
    coerce=False,
)

RINGS_SCHEMA = DataFrameSchema(
    {
        "ring_id": Column(str, nullable=False),
        "member_accounts": Column(str, nullable=False),
        "pattern_type": Column(str, nullable=False),
        "time_window": Column(str, nullable=False),
    },
    unique=["ring_id"],
    strict=True,
    coerce=False,
)

_TABLE_SCHEMAS: dict[str, DataFrameSchema] = {
    "accounts": ACCOUNTS_SCHEMA,
    "transactions": TRANSACTIONS_SCHEMA,
    "labels": LABELS_SCHEMA,
    "rings": RINGS_SCHEMA,
}


class CanonicalSchemaError(ValueError):
    """A table failed its Pandera schema. ``failure_cases`` holds the offending rows."""

    def __init__(self, table: str, failure_cases: pd.DataFrame) -> None:
        self.table = table
        self.failure_cases = failure_cases
        super().__init__(
            f"Canonical table {table!r} failed schema validation "
            f"({len(failure_cases)} offending row(s)):\n{failure_cases}"
        )


class CanonicalIntegrityError(ValueError):
    """A cross-table referential integrity check failed.

    ``offending_rows`` holds the specific rows (e.g. transactions whose
    endpoint is not a known account) that violated the constraint.
    """

    def __init__(self, message: str, offending_rows: pd.DataFrame) -> None:
        self.offending_rows = offending_rows
        super().__init__(f"{message} ({len(offending_rows)} offending row(s)):\n{offending_rows}")


def json_text(value: object) -> str:
    """Serialise a Python value to the JSON text stored in a canonical column."""
    return json.dumps(value, default=str, sort_keys=True)


@dataclass
class CanonicalDataset:
    """The four canonical tables for one dataset, plus its registry name."""

    name: str
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    labels: pd.DataFrame
    rings: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "accounts": self.accounts,
            "transactions": self.transactions,
            "labels": self.labels,
            "rings": self.rings,
        }

    def validate(self) -> CanonicalDataset:
        """Validate every table's schema, then cross-table referential integrity.

        Raises :class:`CanonicalSchemaError` (per-table) or
        :class:`CanonicalIntegrityError` (cross-table), both carrying the
        offending rows, on the first failure encountered.
        """
        for table_name, df in self.tables().items():
            schema = _TABLE_SCHEMAS[table_name]
            try:
                schema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as exc:
                raise CanonicalSchemaError(table_name, exc.failure_cases) from exc
        self._validate_referential_integrity()
        return self

    def _validate_referential_integrity(self) -> None:
        account_ids = set(self.accounts["account_id"])
        tx_ids = set(self.transactions["tx_id"])
        ring_ids = set(self.rings["ring_id"])

        bad_src = self.transactions[~self.transactions["src_account"].isin(account_ids)]
        if len(bad_src):
            raise CanonicalIntegrityError(
                "transactions.src_account references an unknown account_id", bad_src
            )
        bad_dst = self.transactions[~self.transactions["dst_account"].isin(account_ids)]
        if len(bad_dst):
            raise CanonicalIntegrityError(
                "transactions.dst_account references an unknown account_id", bad_dst
            )

        entity_id_sets = {"tx": tx_ids, "account": account_ids, "ring": ring_ids}
        for entity_type, valid_ids in entity_id_sets.items():
            subset = self.labels[self.labels["entity_type"] == entity_type]
            bad_labels = subset[~subset["entity_id"].isin(valid_ids)]
            if len(bad_labels):
                raise CanonicalIntegrityError(
                    f"labels.entity_id references an unknown {entity_type} id", bad_labels
                )

        for _, row in self.rings.iterrows():
            members = json.loads(row["member_accounts"])
            unknown = set(members) - account_ids
            if unknown:
                raise CanonicalIntegrityError(
                    f"rings.member_accounts for ring {row['ring_id']!r} references "
                    f"unknown account_id(s) {sorted(unknown)}",
                    self.rings[self.rings["ring_id"] == row["ring_id"]],
                )

    def write(self, processed_dir: Path | str) -> Path:
        """Write all four tables as Parquet under ``<processed_dir>/<name>/``."""
        out_dir = Path(processed_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for table_name, df in self.tables().items():
            df.to_parquet(out_dir / f"{table_name}.parquet", index=False)
        return out_dir

    @classmethod
    def read(cls, processed_dir: Path | str, name: str) -> CanonicalDataset:
        """Read a previously written canonical dataset back from Parquet."""
        in_dir = Path(processed_dir) / name
        return cls(
            name=name,
            accounts=pd.read_parquet(in_dir / "accounts.parquet"),
            transactions=pd.read_parquet(in_dir / "transactions.parquet"),
            labels=pd.read_parquet(in_dir / "labels.parquet"),
            rings=pd.read_parquet(in_dir / "rings.parquet"),
        )


__all__ = [
    "ENTITY_TYPES",
    "ACCOUNTS_SCHEMA",
    "TRANSACTIONS_SCHEMA",
    "LABELS_SCHEMA",
    "RINGS_SCHEMA",
    "CanonicalSchemaError",
    "CanonicalIntegrityError",
    "CanonicalDataset",
    "json_text",
]
