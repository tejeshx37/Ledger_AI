"""Unit tests for deterministic seeding."""

from __future__ import annotations

import random

import numpy as np

from ledger.utils.seeding import seed_everything


def test_seed_everything_makes_python_random_reproducible() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_seed_everything_makes_numpy_reproducible() -> None:
    seed_everything(7)
    first = np.random.rand(10)
    seed_everything(7)
    second = np.random.rand(10)
    assert np.array_equal(first, second)


def test_seed_everything_different_seeds_differ() -> None:
    seed_everything(1)
    first = np.random.rand(10)
    seed_everything(2)
    second = np.random.rand(10)
    assert not np.array_equal(first, second)


def test_seed_everything_sets_pythonhashseed() -> None:
    import os

    seed_everything(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
