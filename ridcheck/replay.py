"""Score a JSONL capture with the same int8 model the board runs.

Each line is one location fix. Two shapes are accepted:

- Simulator / labeled capture: track, t, rssi, lat, lon, alt, height, speed,
  vspeed, hdg, operator_lat, operator_lon, node_lat, node_lon, and optional
  node_speed, clock_skew, cohort, addr_local, label.
- A Totem-air local record (`t`, `mac`, `rssi`, `lat`, `lon`, `alt`, `speed`,
  `hdg`, `node_lat`, `node_lon`, `decoded`). Rows without a fix are skipped.

    python -m ridcheck replay examples/sample_scene.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ridcheck.export import load_quant
from ridcheck.features import TrackArrays, addr_local_feature, network_disagreement, windows_from_track

ROOT = Path(__file__).resolve().parents[1]


def _clock_skew(record: dict) -> float:
    if "clock_skew" in record and record["clock_skew"] is not None:
        return float(record["clock_skew"])
    decoded = record.get("decoded") or {}
    drone = decoded.get("time_since_utc_hour_sec")
    if drone is None or "t" not in record:
        return 0.0
    delta = (float(record["t"]) % 3600.0) - float(drone)
    if delta > 1800.0:
        delta -= 3600.0
    if delta < -1800.0:
        delta += 3600.0
    return delta


def _num(record, *keys, default=0.0):
    decoded = record.get("decoded") or {}
    for key in keys:
        if key in record and record[key] is not None:
            return float(record[key])
        if key in decoded and decoded[key] is not None:
            return float(decoded[key])
    return float(default)


def record_usable(record: dict) -> bool:
    kind = record.get("type") or record.get("msg_type") or (record.get("decoded") or {}).get("message_type")
    if kind not in (None, "Location/Vector", "DJI Vendor"):
        return False
    return record.get("lat") is not None and record.get("lon") is not None and record.get("t") is not None


def aircraft_key(record: dict) -> str:
    if record.get("track"):
        return str(record["track"])
    mac = record.get("mac", "?")
    uas = record.get("id") or record.get("uas_id") or ""
    return f"{mac}|{uas}"


def track_key(record: dict) -> str:
    node = record.get("node_id")
    if node is None and record.get("node_lat") is not None and record.get("node_lon") is not None:
        node = f"{round(float(record['node_lat']), 4)},{round(float(record['node_lon']), 4)}"
    base = aircraft_key(record)
    return base if node is None else f"{base}@{node}"


def rows_to_tracks(rows: list[dict]) -> list[TrackArrays]:
    grouped: dict[str, list] = defaultdict(list)
    usable = [row for row in rows if record_usable(row)]
    for row in usable:
        grouped[track_key(row)].append(row)
    # Cohort from RSSI neighbours in the same one-second bucket when a row omits it.
    buckets: dict[int, list] = defaultdict(list)
    for row in usable:
        buckets[int(round(float(row["t"])))].append(row)
    for bucket in buckets.values():
        for row in bucket:
            if row.get("cohort") is not None:
                continue
            rssi = float(row.get("rssi") or -100)
            row["cohort"] = 1 + sum(
                1 for other in bucket
                if other is not row and abs(float(other.get("rssi") or -100) - rssi) <= 3.0
            )
    tracks = []
    for key, group in grouped.items():
        group.sort(key=lambda row: float(row["t"]))
        if len(group) < 2:
            continue
        def col(name, *alts, default=0.0):
            return np.array([_num(row, name, *alts, default=default) for row in group], dtype=np.float64)

        radio = str(group[0].get("radio") or "wifi_beacon")
        if group[0].get("addr_local") is not None:
            addr = float(group[0]["addr_local"])
        elif group[0].get("mac"):
            addr = addr_local_feature(group[0]["mac"], radio)
        else:
            addr = -1.0
        label = int(group[0]["label"]) if group[0].get("label") is not None else -1
        tracks.append(TrackArrays(
            t=col("t"),
            rssi=col("rssi", default=-90),
            lat=col("lat"),
            lon=col("lon"),
            alt=col("alt", "altitude_geo"),
            height=col("height", "height_agl", default=_num(group[0], "alt")),
            speed=col("speed", "ground_speed"),
            vspeed=col("vspeed", "vertical_speed"),
            hdg=col("hdg", "heading"),
            op_lat=col("operator_lat", default=_num(group[0], "lat")),
            op_lon=col("operator_lon", default=_num(group[0], "lon")),
            node_lat=col("node_lat", default=_num(group[0], "lat")),
            node_lon=col("node_lon", default=_num(group[0], "lon")),
            node_speed=col("node_speed"),
            clock_skew=np.array([_clock_skew(row) for row in group], dtype=np.float64),
            cohort=np.array([float(row.get("cohort") or 1) for row in group]),
            addr_local=addr,
            label=label,
            slice_name=str(group[0].get("slice") or aircraft_key(group[0])),
            emit_lat=col("lat"),
            emit_lon=col("lon"),
            net_db=np.zeros(len(group), dtype=np.float64),
        ))
    _attach_network(tracks)
    return tracks


def _attach_network(tracks: list[TrackArrays]) -> None:
    """Fill net_db when the same aircraft was heard at more than one node."""
    groups: dict[str, list[TrackArrays]] = defaultdict(list)
    for track in tracks:
        groups[track.slice_name].append(track)
    for group in groups.values():
        if len(group) < 2:
            continue
        for track in group:
            rms = np.zeros(len(track), dtype=np.float64)
            for index, moment in enumerate(track.t):
                node_lat, node_lon, rssi = [], [], []
                for other in group:
                    nearest = int(np.argmin(np.abs(other.t - moment)))
                    if abs(float(other.t[nearest]) - float(moment)) > 1.5:
                        continue
                    node_lat.append(other.node_lat[nearest])
                    node_lon.append(other.node_lon[nearest])
                    rssi.append(other.rssi[nearest])
                if len(rssi) < 2:
                    continue
                rms[index] = network_disagreement(
                    np.array(node_lat)[:, None],
                    np.array(node_lon)[:, None],
                    np.full((len(rssi), 1), track.lat[index]),
                    np.full((len(rssi), 1), track.lon[index]),
                    np.array(rssi)[:, None],
                )[0]
            track.net_db = rms


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def windows_from_jsonl(path: Path):
    """Labeled windows from a JSONL capture. Every track needs label 0 or 1."""
    from ridcheck.sim import tracks_to_windows

    tracks = rows_to_tracks(load_jsonl(path))
    unlabeled = [track.slice_name for track in tracks if track.label < 0]
    if unlabeled:
        names = ", ".join(unlabeled[:5])
        raise SystemExit(f"{path} is missing label on: {names}")
    return tracks_to_windows(tracks)


def score_file(path: Path, quant_path: Path) -> list[dict]:
    model = load_quant(quant_path)
    reports = []
    for track in rows_to_tracks(load_jsonl(path)):
        windows = windows_from_track(track)
        if len(windows) == 0:
            continue
        scores = model.score(windows)
        mean = float(np.mean(scores))
        reports.append({
            "track": track.slice_name,
            "label": track.label,
            "windows": int(len(windows)),
            "mean_p": mean,
            "verdict": model.verdict(mean),
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--quant", type=Path, default=ROOT / "results" / "quant.npz")
    args = parser.parse_args()
    reports = score_file(args.jsonl, args.quant)
    if not reports:
        raise SystemExit(f"no complete tracks in {args.jsonl}")
    for report in reports:
        known = "" if report["label"] < 0 else f"  label={report['label']}"
        print(
            f"{report['track']:<24} windows={report['windows']:<3} "
            f"p={report['mean_p']:.3f}  {report['verdict']}{known}"
        )


if __name__ == "__main__":
    main()
