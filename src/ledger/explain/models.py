"""Phase 6: Explanation layer models.

Defines the structured Evidence dataclass representing the explanation payload
produced for a given alert.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """Structured evidence schema for justifying a suspicious alert."""

    alert_id: str
    account_id: str
    contributing_accounts: list[str]
    contributing_transactions: list[str]
    motif_type_detected: str | None
    time_window: tuple[str, str] | None
    feature_attributions: dict[str, float]
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the evidence instance to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        """Convert the evidence instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
