"""Slice metrics for a detector that must stay quiet on real aircraft."""

from __future__ import annotations

import numpy as np

from ridcheck.sim import PHYSICS_SLICES, REAL_SLICES, SPOOF_SLICES


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC from the Mann–Whitney rank statistic."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # Average ranks so tied scores do not invent separation.
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        # ranks are 1-based
        average = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average
        start = end
    sum_pos = float(np.sum(ranks[labels == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_at_fpr(real_scores: np.ndarray, fpr: float) -> float:
    """Lowest score cut that keeps about `fpr` of real windows on the spoof side."""
    if len(real_scores) == 0:
        return 1.0
    return float(np.quantile(real_scores, 1.0 - fpr))


def summarize(labels, scores, slices, threshold: float) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    slices = np.asarray(slices)
    real = np.isin(slices, REAL_SLICES)
    rows = []
    for name in list(REAL_SLICES) + list(SPOOF_SLICES):
        mask = slices == name
        if not np.any(mask):
            continue
        part = scores[mask]
        if name in REAL_SLICES:
            rate = float(np.mean(part >= threshold))
            kind = "fpr"
        else:
            rate = float(np.mean(part >= threshold))
            kind = "recall"
        rows.append({
            "slice": name,
            "n": int(np.sum(mask)),
            "kind": kind,
            "rate": rate,
            "mean_score": float(np.mean(part)),
        })
    physics = np.isin(slices, PHYSICS_SLICES)
    return {
        "threshold": threshold,
        "auroc": auroc(labels, scores),
        "real_fpr": float(np.mean(scores[real] >= threshold)) if np.any(real) else float("nan"),
        "physics_recall": float(np.mean(scores[physics] >= threshold)) if np.any(physics) else float("nan"),
        "colocated_recall": float(np.mean(scores[slices == "spoof_colocated"] >= threshold))
        if np.any(slices == "spoof_colocated") else float("nan"),
        "slices": rows,
    }


def format_summary(summary: dict, title: str) -> str:
    lines = [
        f"{title}",
        f"  threshold {summary['threshold']:.3f}   AUROC {summary['auroc']:.3f}   "
        f"real FPR {summary['real_fpr']:.1%}   physics recall {summary['physics_recall']:.1%}   "
        f"colocated recall {summary.get('colocated_recall', float('nan')):.1%}",
        f"  {'slice':<24} {'n':>6} {'rate':>8} {'mean p':>8}",
    ]
    for row in summary["slices"]:
        lines.append(
            f"  {row['slice']:<24} {row['n']:>6} {row['rate']:>8.1%} {row['mean_score']:>8.3f}"
        )
    return "\n".join(lines)
