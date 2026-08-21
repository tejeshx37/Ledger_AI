"""Deterministic seeding across every source of randomness LEDGER touches.

A detection result in this domain is only trustworthy if it is
reproducible: an examiner must be able to re-run an experiment and get the
same alerts. ``seed_everything`` is the single call every entry point
(CLI command, training script, API startup for stochastic components) makes
before doing anything else.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy, and (if installed) PyTorch.

    Also sets deterministic cuDNN flags so GPU convolution/attention
    kernels do not introduce run-to-run nondeterminism, and pins
    ``PYTHONHASHSEED`` for dict/set iteration order stability in
    subprocesses spawned after this call.

    Args:
        seed: the seed to apply everywhere. Must be a non-negative int;
            this is enforced by the caller's config validation
            (``ModelConfig.random_seed`` has no explicit bound here because
            any int, including negative, is a valid seed for these
            libraries — but callers should source it from config, never
            hardcode it inline).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
