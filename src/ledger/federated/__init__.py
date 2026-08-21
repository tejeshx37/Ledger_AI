"""LEDGER federated module (see project brief for phase scope)."""

from __future__ import annotations

from ledger.federated.boundary import GLOBAL_BOUNDARY_REGISTRY
from ledger.federated.client import LedgerFlowerClient
from ledger.federated.experiment import run_federated_experiment
from ledger.federated.partitioning import PartitionStats, partition_dataset
from ledger.federated.server import LedgerFlowerStrategy

__all__ = [
    "partition_dataset",
    "PartitionStats",
    "LedgerFlowerClient",
    "LedgerFlowerStrategy",
    "GLOBAL_BOUNDARY_REGISTRY",
    "run_federated_experiment",
]
