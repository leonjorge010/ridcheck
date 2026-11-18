"""Train the spoof detector and export the int8 weights.

    python -m ridcheck train
    python -m ridcheck train --device cuda --scenes 4000 --epochs 16

Writes results/ (or --out): config.json, history.csv, model.pt, quant.npz,
and firmware/src/model_weights.h. Scratch runs can use --out runs/<name>.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ridcheck.export import save_quant, write_header
from ridcheck.features import FEATURE_NAMES, N_FEATURES, WINDOW
from ridcheck.metrics import format_summary, summarize, threshold_at_fpr
from ridcheck.model import SpoofNet
from ridcheck.quantize import feature_scales, quantize_network
from ridcheck.replay import windows_from_jsonl
from ridcheck.sim import REAL_SLICES, make_dataset, make_eval_set, training_counts

# Easy false tracks already saturate the loss. The colocated hover does not,
# so its windows pull harder, and real windows pull back so a slow hover
# stays on the real side of the cut.
SLICE_LOSS_WEIGHT = {
    "spoof_colocated": 2.0,
    "real": 1.5,
    "real_pair": 1.5,
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results"
HEADER_PATH = ROOT / "firmware" / "src" / "model_weights.h"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this machine has no visible GPU")
    return torch.device(name)


def _sample_weights(slices) -> np.ndarray:
    weights = np.ones(len(slices), dtype=np.float32)
    for name, weight in SLICE_LOSS_WEIGHT.items():
        weights[slices == name] = weight
    return weights


def _loader(features, labels, weights, codes, batch: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(labels.astype(np.float32))),
        torch.from_numpy(np.ascontiguousarray(weights)),
        torch.from_numpy(np.ascontiguousarray(codes.astype(np.int64))),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle)


def _predict(model, features, device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            batch = torch.from_numpy(features[start : start + 1024]).to(device)
            logit = model(batch)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float64)


def _selftest_pack(quant, features, count: int = 6):
    chosen = features[:count]
    if len(chosen) < count:
        pad = np.zeros((count - len(chosen), WINDOW, N_FEATURES), dtype=np.float32)
        chosen = np.concatenate([chosen, pad])
    quantized = quant.quantize_features(chosen)
    prob, acc = quant.forward_acc(quantized)
    return {
        "features": chosen.reshape(count, -1).astype(np.float32),
        "quantized": quantized.reshape(count, -1).astype(np.int8),
        "acc": acc.astype(np.int32),
        "prob": prob.astype(np.float64),
    }


def train(
    scenes: int,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    out: Path,
    eval_tracks: int,
    captures: Path | None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = training_counts(scenes)
    features, labels, slices = make_dataset(counts, seed)
    if captures is not None:
        cap_x, cap_y, cap_s = windows_from_jsonl(captures)
        features = np.concatenate([features, cap_x])
        labels = np.concatenate([labels, cap_y])
        slices = np.concatenate([slices, cap_s])
    eval_x, eval_y, eval_slices = make_eval_set(eval_tracks, seed + 10007)
    model = SpoofNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loader = _loader(features, labels, _sample_weights(slices), np.zeros(len(labels), dtype=np.int64), batch, shuffle=True)
    history = []
    best_state = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_x, batch_y, batch_w, _batch_code in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(batch_x)
            loss = (loss_fn(logit, batch_y) * batch_w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        scores = _predict(model, eval_x, device)
        real_scores = scores[np.isin(eval_slices, REAL_SLICES)]
        threshold = threshold_at_fpr(real_scores, 0.01)
        summary = summarize(eval_y, scores, eval_slices, threshold)
        row = {
            "epoch": epoch,
            "loss": total / max(seen, 1),
            "auroc": summary["auroc"],
            "real_fpr": summary["real_fpr"],
            "physics_recall": summary["physics_recall"],
            "colocated_recall": summary["colocated_recall"],
            "threshold": threshold,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  loss {row['loss']:.4f}  AUROC {row['auroc']:.3f}  "
            f"FPR {row['real_fpr']:.1%}  physics recall {row['physics_recall']:.1%}  "
            f"colocated recall {row['colocated_recall']:.1%}",
            flush=True,
        )
        # Keep the easy false tracks, then take whatever also catches the hover
        # that sits on the transmitter.
        score = row["physics_recall"]
        if row["physics_recall"] >= 0.98:
            score += row["colocated_recall"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    float_scores = _predict(model, eval_x, device)
    real_scores = float_scores[np.isin(eval_slices, REAL_SLICES)]
    threshold = threshold_at_fpr(real_scores, 0.01)
    scales = feature_scales(features)
    quant = quantize_network(best_state, scales, features[: min(len(features), 2048)])
    int8_scores = quant.score(eval_x)
    # Keep the operating point on the weights that actually ship.
    int8_real = int8_scores[np.isin(eval_slices, REAL_SLICES)]
    threshold = threshold_at_fpr(int8_real, 0.01)
    watch = threshold_at_fpr(int8_real, 0.05)
    quant.threshold = threshold
    quant.watch = min(watch, threshold)
    float_summary = summarize(eval_y, float_scores, eval_slices, threshold_at_fpr(real_scores, 0.01))
    int8_summary = summarize(eval_y, int8_scores, eval_slices, threshold)
    watch_summary = summarize(eval_y, int8_scores, eval_slices, quant.watch)
    print(format_summary(float_summary, "float32 held-out"))
    print(format_summary(int8_summary, "int8 held-out, 1% cut (spoof)"))
    print(format_summary(watch_summary, "int8 held-out, 5% cut (watch)"))

    torch.save({"state_dict": best_state, "hidden": model.hidden}, out / "model.pt")
    save_quant(quant, out / "quant.npz")
    write_header(quant, _selftest_pack(quant, eval_x), HEADER_PATH)
    config = {
        "seed": seed,
        "scenes": scenes,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "counts": counts,
        "eval_tracks": eval_tracks,
        "captures": str(captures) if captures else None,
        "window": WINDOW,
        "features": list(FEATURE_NAMES),
        "hidden": list(model.hidden),
        "threshold_fpr1": threshold,
        "watch_fpr5": quant.watch,
        "int8_colocated_recall": int8_summary["colocated_recall"],
        "int8_watch_colocated_recall": watch_summary["colocated_recall"],
        "int8_watch_real_fpr": watch_summary["real_fpr"],
        "int8_auroc": int8_summary["auroc"],
        "int8_physics_recall": int8_summary["physics_recall"],
        "int8_real_fpr": int8_summary["real_fpr"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "eval.json").write_text(json.dumps({
        "float32": float_summary,
        "int8": int8_summary,
        "int8_watch": watch_summary,
    }, indent=2) + "\n")
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"wrote {out} and {HEADER_PATH}")
    return {"config": config, "int8": int8_summary, "float32": float_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", type=int, default=2400, help="training-track budget")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-tracks", type=int, default=80, help="held-out tracks per slice")
    parser.add_argument("--captures", type=Path, default=None,
                        help="JSONL of labeled fixes to mix into the simulator set")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    device = resolve_device(args.device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print(f"training on {device}" + (f" ({gpu})" if gpu else ""), flush=True)
    train(
        scenes=args.scenes,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=device,
        out=args.out,
        eval_tracks=args.eval_tracks,
        captures=args.captures,
    )


if __name__ == "__main__":
    main()
