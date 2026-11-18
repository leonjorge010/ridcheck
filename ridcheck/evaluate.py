"""Score a fresh simulated set with the exported int8 model.

    python -m ridcheck evaluate
    python -m ridcheck evaluate --quant results/quant.npz --seed 77
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ridcheck.export import load_quant
from ridcheck.metrics import format_summary, summarize, threshold_at_fpr
from ridcheck.sim import REAL_SLICES, make_eval_set

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quant", type=Path, default=ROOT / "results" / "quant.npz")
    parser.add_argument("--tracks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--fpr", type=float, default=0.01)
    args = parser.parse_args()
    model = load_quant(args.quant)
    features, labels, slices = make_eval_set(args.tracks, args.seed)
    scores = model.score(features)
    real = scores[np.isin(slices, REAL_SLICES)]
    threshold = threshold_at_fpr(real, args.fpr)
    summary = summarize(labels, scores, slices, threshold)
    summary["shipped_threshold"] = model.threshold
    print(format_summary(summary, f"int8 eval seed {args.seed}"))
    print(f"  shipped spoof cut {model.threshold:.3f}   watch cut {model.watch:.3f}")


if __name__ == "__main__":
    main()
