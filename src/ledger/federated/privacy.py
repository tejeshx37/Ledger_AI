"""Phase 5.4: Differential Privacy and Secure Aggregation.

This module implements:
1. Renyi Differential Privacy (RDP) accountant to track cumulative epsilon consumption.
2. Gradient clipping and calibrated Gaussian noise addition for model updates.
3. Secure Aggregation mask generation (pairwise client-to-client masks that sum to zero).
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PrivacyAccountant:
    """Tracks Renyi Differential Privacy (RDP) and converts to (epsilon, delta)."""

    def __init__(self, target_delta: float) -> None:
        if not (0.0 < target_delta < 1.0):
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        self.target_delta = target_delta
        self.steps = 0
        self.noise_multipliers: list[float] = []

    def add_step(self, noise_multiplier: float) -> None:
        """Record a step with a given noise multiplier."""
        self.steps += 1
        self.noise_multipliers.append(noise_multiplier)

    def get_epsilon(self) -> float:
        """Compute the cumulative epsilon using RDP Renyi composition."""
        if not self.noise_multipliers:
            return 0.0

        # Search over standard Renyi DP orders to find the tightest conversion to (epsilon, delta)
        alphas = np.linspace(1.1, 100.0, 100)
        best_eps = float("inf")

        for alpha in alphas:
            # Renyi divergence of order alpha for a Gaussian mechanism with noise scale sigma is alpha / (2 * sigma^2)
            rdp = sum(alpha / (2.0 * (sigma**2)) for sigma in self.noise_multipliers)
            # Conversion formula: epsilon = rdp + log(1/delta) / (alpha - 1)
            eps = rdp + (np.log(1.0 / self.target_delta) / (alpha - 1.0))
            if eps < best_eps:
                best_eps = eps

        return float(best_eps)


def apply_model_dp(
    weights: list[np.ndarray],
    initial_weights: list[np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
) -> list[np.ndarray]:
    """Clip and add Gaussian noise to a list of weights (representing model update)."""
    # 1. Compute weight difference (update delta)
    delta = [w - init_w for w, init_w in zip(weights, initial_weights)]

    # 2. Compute L2 norm of the update
    flat_delta = np.concatenate([d.flatten() for d in delta])
    norm = np.linalg.norm(flat_delta)

    # 3. Clip delta if norm exceeds threshold
    if norm > clip_norm:
        scale = clip_norm / (norm + 1e-9)
        delta = [d * scale for d in delta]

    # 4. Add calibrated Gaussian noise
    noise_scale = noise_multiplier * clip_norm
    delta_noised = []
    for d in delta:
        noise = np.random.normal(0.0, noise_scale, size=d.shape)
        delta_noised.append(d + noise)

    # 5. Return updated weights
    return [init_w + d_n for init_w, d_n in zip(initial_weights, delta_noised)]


def generate_pairwise_masks(
    num_clients: int, shapes: list[tuple[int, ...]], seed: int = 42
) -> dict[int, list[np.ndarray]]:
    """Simulate secure aggregation using pairwise additive zero-sum masks.

    For every pair of clients (i, j) with i < j, generates a random mask R_ij.
    Adds R_ij to client i's update and subtracts R_ij from client j's update.
    The masks exactly cancel out when aggregated at the server.
    """
    masks = {c: [np.zeros(shape) for shape in shapes] for c in range(num_clients)}
    rng = np.random.default_rng(seed)

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            pairwise_seed = int(rng.integers(0, 2**31 - 1))
            pair_rng = np.random.default_rng(pairwise_seed)
            for k, shape in enumerate(shapes):
                r = pair_rng.normal(0.0, 1.0, size=shape)
                masks[i][k] += r
                masks[j][k] -= r

    return masks
