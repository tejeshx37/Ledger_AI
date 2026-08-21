"""Precision-recall curve artifact: a PNG for human review, plus the raw
curve points as JSON so later phases (Phase 4's model comparison, Phase 7's
evaluation report) can re-plot without recomputing from scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def save_pr_curve(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], path: Path | str
) -> Path:
    """Render a precision-recall curve to ``path`` (PNG) and write the
    underlying (precision, recall, threshold) arrays to a sibling
    ``.json`` file with the same stem. Returns the PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve (AP={ap:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    curve_data: dict[str, Any] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": ap,
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    return path


__all__ = ["save_pr_curve"]
