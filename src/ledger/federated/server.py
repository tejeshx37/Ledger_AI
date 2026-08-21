"""Phase 5.3: Flower Federated Server Strategy.

This module implements a custom Flower strategy subclassing FedAvg. It tracks DP budget,
handles client update aggregation, and coordinates boundary embedding exchange at the
end of each round.
"""

from __future__ import annotations

from typing import Any, Callable

import flwr as fl
import numpy as np
import structlog
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from ledger.config.settings import Settings
from ledger.federated.privacy import PrivacyAccountant

logger = structlog.get_logger(__name__)


class LedgerFlowerStrategy(fl.server.strategy.FedAvg):
    """Custom Flower FedAvg strategy integrating privacy budget and boundary exchange."""

    def __init__(
        self,
        *args: Any,
        settings: Settings,
        accountant: PrivacyAccountant,
        on_round_end_callback: Callable[[int, list[np.ndarray]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.accountant = accountant
        self.on_round_end_callback = on_round_end_callback
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)

            if self.settings.privacy.dp_enabled:
                self.accountant.add_step(self.settings.privacy.noise_multiplier)
                cumulative_eps = self.accountant.get_epsilon()
                logger.info(
                    "DP accountant update",
                    round=server_round,
                    cumulative_epsilon=cumulative_eps,
                    target_epsilon=self.settings.privacy.target_epsilon,
                )
                if cumulative_eps > self.settings.privacy.target_epsilon:
                    raise ValueError(
                        f"Privacy budget exceeded: cumulative epsilon {cumulative_eps:.4f} "
                        f"exceeds target epsilon {self.settings.privacy.target_epsilon}"
                    )

            if self.on_round_end_callback is not None:
                self.on_round_end_callback(server_round, weights)

        return aggregated_parameters, metrics
