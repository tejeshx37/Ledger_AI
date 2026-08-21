"""Unit tests for the precision-recall curve artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ledger.evaluation.curves import save_pr_curve


def test_save_pr_curve_writes_png_and_json(tmp_path: Path) -> None:
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.6, 0.4, 0.3, 0.1])
    png_path = save_pr_curve(y_true, y_score, tmp_path / "curve.png")

    assert png_path.exists()
    assert png_path.stat().st_size > 0

    json_path = png_path.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert "precision" in payload
    assert "recall" in payload
    assert "thresholds" in payload
    assert 0.0 <= payload["average_precision"] <= 1.0
    assert len(payload["precision"]) == len(payload["recall"])


def test_save_pr_curve_creates_parent_directory(tmp_path: Path) -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.8, 0.2, 0.6, 0.4])
    nested = tmp_path / "a" / "b" / "curve.png"
    result = save_pr_curve(y_true, y_score, nested)
    assert result == nested
    assert nested.exists()
