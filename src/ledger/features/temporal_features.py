"""Temporal account features: velocity, burstiness, inter-arrival
statistics, and a rolling amount average.

Computed from every transaction incident to an account (as either sender
or receiver) with a known timestamp; accounts with fewer than two such
transactions get zeroed-out statistics rather than NaN or a divide error.
"""

from __future__ import annotations

import pandas as pd

from ledger.config.models import FeaturesConfig
from ledger.data.canonical import CanonicalDataset

_COLUMNS = [
    "account_id",
    "temporal_tx_count",
    "temporal_velocity",
    "temporal_mean_inter_arrival_seconds",
    "temporal_std_inter_arrival_seconds",
    "temporal_burstiness",
    "temporal_rolling_amount_mean",
]


def _incident_transactions(dataset: CanonicalDataset) -> pd.DataFrame:
    tx = dataset.transactions.dropna(subset=["timestamp"])
    return pd.concat(
        [
            tx[["src_account", "timestamp", "amount"]].rename(
                columns={"src_account": "account_id"}
            ),
            tx[["dst_account", "timestamp", "amount"]].rename(
                columns={"dst_account": "account_id"}
            ),
        ],
        ignore_index=True,
    )


def _burstiness(mean_inter_arrival: float, std_inter_arrival: float) -> float:
    """Goh & Barabasi's burstiness parameter B = (sigma - mu) / (sigma + mu).

    Ranges from -1 (perfectly regular) to +1 (maximally bursty); 0 for a
    Poisson process. Zero when there is no meaningful inter-arrival signal.
    """
    denom = std_inter_arrival + mean_inter_arrival
    if denom <= 0:
        return 0.0
    return (std_inter_arrival - mean_inter_arrival) / denom


def compute_temporal_features(dataset: CanonicalDataset, config: FeaturesConfig) -> pd.DataFrame:
    """Return one row per account with temporal activity statistics.

    Accounts with no timestamped incident transaction are absent from the
    result; callers reindex against the full account list and fill missing
    rows with zeros (see :mod:`ledger.features.pipeline`).
    """
    incident = _incident_transactions(dataset)
    if incident.empty:
        return pd.DataFrame(columns=_COLUMNS)

    rows: list[dict[str, object]] = []
    for account_id, group in incident.groupby("account_id"):
        g = group.sort_values("timestamp")
        n = len(g)
        span_seconds = (
            (g["timestamp"].max() - g["timestamp"].min()).total_seconds() if n > 1 else 0.0
        )
        velocity = n / span_seconds if span_seconds > 0 else 0.0

        inter_arrivals = g["timestamp"].diff().dropna().dt.total_seconds()
        mean_inter_arrival = float(inter_arrivals.mean()) if len(inter_arrivals) else 0.0
        std_inter_arrival = float(inter_arrivals.std(ddof=0)) if len(inter_arrivals) else 0.0

        rolling_amount_mean = float(
            g["amount"]
            .fillna(0.0)
            .rolling(window=config.rolling_window_steps, min_periods=1)
            .mean()
            .iloc[-1]
        )

        rows.append(
            {
                "account_id": account_id,
                "temporal_tx_count": n,
                "temporal_velocity": velocity,
                "temporal_mean_inter_arrival_seconds": mean_inter_arrival,
                "temporal_std_inter_arrival_seconds": std_inter_arrival,
                "temporal_burstiness": _burstiness(mean_inter_arrival, std_inter_arrival),
                "temporal_rolling_amount_mean": rolling_amount_mean,
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)


__all__ = ["compute_temporal_features"]
