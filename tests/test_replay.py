"""Replay scores a labeled capture with the shipped int8 weights."""

import json
from pathlib import Path

from spoofdet.replay import score_file
from spoofdet.sim import generate_tracks

ROOT = Path(__file__).resolve().parents[1]


def _dump(tracks, path: Path):
    lines = []
    for index, track in enumerate(tracks):
        for row in range(len(track)):
            lines.append({
                "track": f"{track.slice_name}-{index}",
                "slice": track.slice_name,
                "label": track.label,
                "t": float(track.t[row]),
                "rssi": float(track.rssi[row]),
                "lat": float(track.lat[row]),
                "lon": float(track.lon[row]),
                "alt": float(track.alt[row]),
                "height": float(track.height[row]),
                "speed": float(track.speed[row]),
                "vspeed": float(track.vspeed[row]),
                "hdg": float(track.hdg[row]),
                "operator_lat": float(track.op_lat[row]),
                "operator_lon": float(track.op_lon[row]),
                "node_lat": float(track.node_lat[row]),
                "node_lon": float(track.node_lon[row]),
                "node_speed": float(track.node_speed[row]),
                "clock_skew": float(track.clock_skew[row]),
                "cohort": float(track.cohort[row]),
                "addr_local": float(track.addr_local),
                "mac": "02:00:00:00:00:01" if track.addr_local > 0 else "60:60:1f:00:00:01",
                "radio": "wifi_beacon",
            })
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_replay_flags_a_teleport_and_keeps_a_real_track(tmp_path):
    tracks = generate_tracks("real", 1, seed=42) + generate_tracks("spoof_teleport", 1, seed=43)
    path = tmp_path / "scene.jsonl"
    _dump(tracks, path)
    reports = score_file(path, ROOT / "results" / "quant.npz")
    by_name = {report["track"]: report for report in reports}
    assert by_name["spoof_teleport"]["verdict"] == "spoof"
    assert by_name["spoof_teleport"]["mean_p"] > 0.9
    assert by_name["real"]["verdict"] == "real"
    assert by_name["real"]["mean_p"] < by_name["spoof_teleport"]["mean_p"]
