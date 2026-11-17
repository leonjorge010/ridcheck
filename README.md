# Spoofed Remote ID detector

A small classifier that scores a Remote ID track as a real aircraft or a false one, then runs that score on the ESP32-S3 board. Totem-air is the receiver: it already stores RSSI, the aircraft fix, speed, heading, the operator fix, and the node's own GNSS fix. Project Ghost is the lab source of false tracks, one radio claiming many aircraft. This project trains on those receiver-side patterns and exports integer weights for the board.

The training loop matches the CBU robot: a fast simulator stands in for the slow capture, a held-out table decides the operating point, and the same weights are what the device runs.

## What a window contains

Eight location fixes, oldest first, sixteen numbers each. On the Pi those numbers come from a Totem-air detection record. On the board, node speed comes from the LSM6DSV16X and the node fix comes from the SAM-M10Q. The Wio-S3's SX1262 is the LoRa radio and is not part of the score.

| Feature | Where it comes from |
|---|---|
| `rssi_norm`, `log_claimed_range`, `rssi_residual` | RSSI versus free-space loss at the range from `node_lat`/`node_lon` to the claimed fix |
| `speed_reported`, `speed_track`, `speed_residual`, `heading_error`, `step_ratio` | Reported speed and heading versus the fixes |
| `vspeed_residual`, `alt_gap_rate` | Vertical speed and `(altitude_geo - height_agl)` |
| `dt_center`, `clock_skew` | Gap between fixes, and the node clock versus `time_since_utc_hour_sec` |
| `log_operator_range` | Aircraft versus the operator fix |
| `addr_local` | Locally administered Wi-Fi address, or a BLE static-random address |
| `cohort_log` | Other tracks within 3 dB in the same second |
| `node_speed` | IMU speed; 0 when the node is fixed |

The network is 128 → 32 → 16 → 1. The hidden layers are int8. The ESP32-S3 runs that integer net.

## Train

GPU training runs as one Slurm job on the CBU compute lab (`jduran`, files in `/lab/home/jduran`). The lab WireGuard tunnel has to be up on the laptop first. Its private key stays on that laptop. The login command is `ssh -p 2222 jduran@10.77.0.1`, and the server fingerprint must be `SHA256:w3zDp2wg5VOp6Y8Z/yCv6LflPLi+cnkMTJqbJfvh5A8`. The lab accepts SSH keys only. Generate one on the laptop if `~\.ssh\id_ed25519.pub` is missing, open `http://10.77.0.1` with the tunnel on, and register that public key before the first login.

From the laptop, in this repo:

```bash
bash lab/submit.sh --wait
```

That copies the project to `/lab/home/jduran/spoofdet`, submits `lab/train.sbatch` (one GPU, one hour), and copies `results/` and `firmware/src/model_weights.h` back when the job ends. Without `--wait` it only submits. Watch the queue with `ssh -p 2222 jduran@10.77.0.1 squeue --me`.

The job trains with `--device cuda --scenes 8000 --epochs 20 --eval-tracks 120`. A machine that already has CUDA can run the same command directly:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m spoofdet train --device cuda --scenes 8000 --epochs 20 --eval-tracks 120
```

`--device auto` uses CUDA when it is visible and CPU otherwise. The run writes `results/` (`config.json`, `history.csv`, `eval.json`, `model.pt`, `quant.npz`) and `firmware/src/model_weights.h`. The weights currently checked in are a CPU baseline (`results/config.json` says `"device": "cpu"`). The lab job replaces them.

A labeled capture can be mixed in. Each line is one fix and needs `label` 0 or 1 plus a `track` id. Totem-air's local JSONL shape (`t`, `mac`, `rssi`, `lat`, `lon`, `decoded`, …) is accepted.

```bash
.venv/bin/python -m spoofdet train --device cuda --captures captures.jsonl
```

Score a capture with the shipped int8 weights:

```bash
.venv/bin/python -m spoofdet replay examples/sample_scene.jsonl
.venv/bin/python -m spoofdet evaluate --seed 77
```

## Held-out result

The weights in `results/` and `firmware/src/model_weights.h` are the int8 model. The threshold is set so about 1% of real windows are called spoof. Full numbers are in `results/eval.json`.

| Slice | What it is | Rate at the 1% threshold |
|---|---|---|
| real / real pair | Consistent flight, RSSI matches range | 1.5% / 0.5% false alarms |
| teleport, drift | Claimed motion does not match the fixes | 100% |
| waypoint far | Smooth path, power still comes from a nearby radio | 100% |
| swarm | Several tracks sharing one radio's RSSI | 100% |
| universal MAC | Drifted track with a burned-in address | 98% |
| colocated | Slow path sitting on the transmitter, ordinary GNSS noise | 2% |

A false aircraft that loiters on top of the transmitter, with a matching pilot fix, scores a bit higher than a real hover and almost never crosses the 1% line. That case needs another sensor or the Remote ID authentication page. The universal-MAC row is there so the motion and the power are doing the work, not the address bit.

## Firmware

`firmware/` is a PlatformIO sketch for the ESP32-S3. On boot it checks six baked windows against the host. Then it reads one line of 128 comma-separated floats (8 fixes × 16 features, oldest first) and prints `p=0.998 spoof` or `p=0.365 real`. `SELFTEST` runs the check again.

```bash
cd firmware && pio run -t upload && pio device monitor
```

The integer core was checked on the host with `make parity` (the C self-test must match the Python int8 model). The sketch has not been flashed; the board is not attached here.

## Tests

```bash
python -m pytest -q
make parity
```
