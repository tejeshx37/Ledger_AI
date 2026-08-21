"""Per-dataset adapters: raw files -> canonical tables.

``ADAPTER_REGISTRY`` maps a registered dataset name (see
:mod:`ledger.data.registry`) to the adapter class that normalises it.
"""

from __future__ import annotations

from ledger.data.adapters.baf import BafAdapter
from ledger.data.adapters.base import DatasetAdapter, RawSchemaError
from ledger.data.adapters.elliptic import EllipticAdapter
from ledger.data.adapters.elliptic_pp import EllipticPlusPlusAdapter
from ledger.data.adapters.ibm_aml import IbmAmlAdapter
from ledger.data.adapters.ieee_cis import IeeeCisAdapter
from ledger.data.adapters.paysim import PaysimAdapter

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "elliptic": EllipticAdapter,
    "elliptic_pp": EllipticPlusPlusAdapter,
    "ibm_aml": IbmAmlAdapter,
    "paysim": PaysimAdapter,
    "baf": BafAdapter,
    "ieee_cis": IeeeCisAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate the adapter registered for ``name``."""
    try:
        adapter_cls = ADAPTER_REGISTRY[name]
    except KeyError as exc:
        allowed = sorted(ADAPTER_REGISTRY)
        raise KeyError(f"No adapter registered for dataset {name!r}. Known: {allowed}") from exc
    return adapter_cls()


__all__ = [
    "DatasetAdapter",
    "RawSchemaError",
    "ADAPTER_REGISTRY",
    "get_adapter",
]
