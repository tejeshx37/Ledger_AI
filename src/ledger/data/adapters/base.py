"""Abstract base for per-dataset adapters: raw files -> canonical tables.

Every adapter loads a dataset's native files, schema-validates them (so a
malformed download fails loudly at the boundary rather than corrupting
canonical data), and maps them onto the four canonical tables defined in
:mod:`ledger.data.canonical`. Downstream code (graph construction, feature
engineering, models) depends only on the canonical schema, never on any
adapter's internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema

from ledger.data.canonical import CanonicalDataset


class RawSchemaError(ValueError):
    """A dataset's native file failed its raw-format schema check.

    Raised at the earliest possible point (right after parsing, before any
    normalisation) so a corrupted or unexpectedly-shaped download is never
    silently carried into the canonical tables. ``failure_cases`` holds the
    offending rows, as returned by Pandera's lazy validation.
    """

    def __init__(self, source: str, failure_cases: pd.DataFrame) -> None:
        self.source = source
        self.failure_cases = failure_cases
        super().__init__(
            f"Raw file {source!r} failed schema validation "
            f"({len(failure_cases)} offending row(s)):\n{failure_cases}"
        )


def validate_raw(df: pd.DataFrame, schema: DataFrameSchema, source: str) -> pd.DataFrame:
    """Validate ``df`` against ``schema``, raising :class:`RawSchemaError` with
    the offending rows attached on failure. Returns the (possibly coerced)
    validated frame.
    """
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise RawSchemaError(source, exc.failure_cases) from exc


class DatasetAdapter(ABC):
    """Loads one dataset's raw files and normalises them into a
    :class:`~ledger.data.canonical.CanonicalDataset`.

    Subclasses implement :meth:`load_raw` (parse and schema-validate the
    dataset's native files) and :meth:`to_canonical` (map the validated raw
    tables onto the four canonical tables). :meth:`run` composes both steps
    and validates the canonical result, so a caller only ever needs to call
    ``run``.
    """

    name: str

    @abstractmethod
    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        """Parse and schema-validate this dataset's native files."""

    @abstractmethod
    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        """Map validated raw tables onto the four canonical tables."""

    def run(self, raw_dir: Path | str) -> CanonicalDataset:
        """Load, validate, and normalise this dataset, returning a
        canonical-schema-validated :class:`CanonicalDataset`.
        """
        raw = self.load_raw(Path(raw_dir))
        canonical = self.to_canonical(raw)
        return canonical.validate()


__all__ = ["DatasetAdapter", "RawSchemaError", "validate_raw"]
