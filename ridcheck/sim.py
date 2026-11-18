"""Fast track simulator for spoofed-Remote-ID training.

This is the same role as the robot project's raycast gym: a lightweight stand-in
for the slow real capture. It emits the features a receiver would store, not
Remote ID frames.

Real tracks integrate speed and heading, and their RSSI follows the range from
the node's GNSS fix. Spoofed tracks reuse the inconsistencies a single lab
transmitter produces: claimed motion that does not match the fixes, and received
power that follows the transmitter instead of the claimed position. Every
reported fix gets GNSS scatter, including the smooth ones. The colocated slice
is the hard case: a slow path that stays on top of the transmitter, so the
power and the claimed range agree.
"""

from __future__ import annotations

import numpy as np

from ridcheck.features import (
    N_FEATURES,
    TrackArrays,
    expected_rssi,
    haversine_m,
    windows_from_track,
)

# Stand-in for the receiver's own fix.
DEFAULT_NODE = (34.0, -117.3)
N_SAMPLES = 28

SLICES = (
    "real",
    "real_pair",
    "spoof_teleport",
    "spoof_drift",
    "spoof_waypoint_far",
    "spoof_colocated",
    "spoof_swarm",
    "spoof_universal_mac",
)

SPOOF_SLICES = tuple(name for name in SLICES if name.startswith("spoof_"))
REAL_SLICES = ("real", "real_pair")

# Physics slices the address bit cannot answer on its own.
PHYSICS_SLICES = (
    "spoof_teleport",
    "spoof_drift",
    "spoof_waypoint_far",
    "spoof_universal_mac",
    "spoof_swarm",
)

TRAIN_WEIGHTS = {
    "real": 8,
    "real_pair": 4,
    "spoof_teleport": 3,
    "spoof_drift": 3,
    "spoof_waypoint_far": 3,
    "spoof_colocated": 3,
    "spoof_swarm": 3,
    "spoof_universal_mac": 3,
}


def meters_to_offset(lat: float, north_m: float, east_m: float):
    dlat = north_m / 111_320.0
    denom = 111_320.0 * np.cos(np.radians(lat))
    dlon = east_m / (denom if abs(denom) > 1.0 else 111_320.0)
    return float(dlat), float(dlon)


def move_by(lat: float, lon: float, distance_m: float, bearing_deg: float):
    bearing = np.radians(bearing_deg)
    dlat, dlon = meters_to_offset(lat, distance_m * np.cos(bearing), distance_m * np.sin(bearing))
    return lat + dlat, lon + dlon


def gnss_noise(rng: np.random.Generator, lat, lon, sigma_m: float = 2.5):
    """Scatter a reported fix the way a decoded GNSS position is scattered."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    north = rng.normal(0.0, sigma_m, lat.shape)
    east = rng.normal(0.0, sigma_m, lat.shape)
    lat_out = lat + north / 111_320.0
    denom = 111_320.0 * np.cos(np.radians(float(np.mean(lat))))
    lon_out = lon + east / (denom if abs(denom) > 1.0 else 111_320.0)
    return lat_out, lon_out


def move_ne(lat: float, lon: float, north_m, east_m):
    north = np.asarray(north_m, dtype=np.float64)
    east = np.asarray(east_m, dtype=np.float64)
    lat_out = float(lat) + north / 111_320.0
    denom = 111_320.0 * np.cos(np.radians(float(lat)))
    lon_out = float(lon) + east / (denom if abs(float(denom)) > 1.0 else 111_320.0)
    return lat_out, lon_out


def _timebase(rng: np.random.Generator, n: int, jitter: float) -> np.ndarray:
    dt = np.clip(1.0 + rng.normal(0.0, jitter, n - 1), 0.35, 2.2)
    return np.concatenate([[0.0], np.cumsum(dt)])


def _integrate(lat0: float, lon0: float, t: np.ndarray, speed: np.ndarray, heading: np.ndarray):
    lat = np.empty(len(t), dtype=np.float64)
    lon = np.empty(len(t), dtype=np.float64)
    lat[0], lon[0] = lat0, lon0
    for i in range(1, len(t)):
        dist = float(speed[i] * (t[i] - t[i - 1]))
        lat[i], lon[i] = move_by(lat[i - 1], lon[i - 1], dist, float(heading[i]))
    return lat, lon


def _node(rng: np.random.Generator, t: np.ndarray):
    speed = 0.0 if rng.random() < 0.75 else float(rng.uniform(0.5, 1.5))
    heading = np.full(len(t), rng.uniform(0.0, 360.0))
    lat, lon = _integrate(DEFAULT_NODE[0], DEFAULT_NODE[1], t, np.full(len(t), speed), heading)
    measured = np.clip(speed + rng.normal(0.0, 0.04, len(t)), 0.0, None)
    return lat, lon, measured


def _tx_dbm(rng: np.random.Generator) -> float:
    return float(rng.normal(12.0, 3.0))


def _rssi_at(range_m, tx_dbm: float, rng: np.random.Generator, n: int, fade: bool):
    rssi = expected_rssi(range_m, tx_dbm) + rng.normal(0.0, 2.0, n)
    if fade and rng.random() < 0.35:
        rssi[int(rng.integers(0, n))] -= float(rng.uniform(6.0, 12.0))
    return np.clip(rssi, -110.0, -20.0)


def _skew(rng: np.random.Generator, n: int, spread: str) -> np.ndarray:
    if spread == "large":
        base = float(rng.uniform(-25.0, 25.0))
    else:
        base = float(rng.normal(0.0, 0.35))
    return base + rng.normal(0.0, 0.05, n)


def _pack(
    t, rssi, lat, lon, alt, height, speed, vspeed, hdg,
    op_lat, op_lon, node_lat, node_lon, node_speed, clock_skew,
    addr_local, label, slice_name,
) -> TrackArrays:
    n = len(t)
    return TrackArrays(
        t=np.asarray(t, dtype=np.float64),
        rssi=np.asarray(rssi, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
        alt=np.asarray(alt, dtype=np.float64),
        height=np.asarray(height, dtype=np.float64),
        speed=np.asarray(speed, dtype=np.float64),
        vspeed=np.asarray(vspeed, dtype=np.float64),
        hdg=np.asarray(hdg, dtype=np.float64) % 360.0,
        op_lat=np.asarray(op_lat, dtype=np.float64),
        op_lon=np.asarray(op_lon, dtype=np.float64),
        node_lat=np.asarray(node_lat, dtype=np.float64),
        node_lon=np.asarray(node_lon, dtype=np.float64),
        node_speed=np.asarray(node_speed, dtype=np.float64),
        clock_skew=np.asarray(clock_skew, dtype=np.float64),
        cohort=np.ones(n, dtype=np.float64),
        addr_local=float(addr_local),
        label=int(label),
        slice_name=slice_name,
    )


def _real_loiter(rng, t, node_lat, node_lon, node_speed, slice_name: str):
    """A real aircraft loitering, including close to the receiver.

    This keeps a slow nearby track from being treated as a spoof by default.
    """
    n = len(t)
    center = move_by(
        DEFAULT_NODE[0], DEFAULT_NODE[1], float(rng.uniform(15.0, 180.0)), float(rng.uniform(0, 360))
    )
    lat, lon, alt, height, speed, vspeed, hdg, op_lat, op_lon = _loiter_motion(
        rng, n, t, center[0], center[1]
    )
    true_range = haversine_m(node_lat, node_lon, lat, lon)
    rssi = _rssi_at(true_range, _tx_dbm(rng), rng, n, fade=True)
    reported_lat, reported_lon = gnss_noise(rng, lat, lon)
    addr = 1.0 if rng.random() < 0.20 else -1.0
    return _pack(
        t, rssi, reported_lat, reported_lon, alt, height, speed, vspeed, hdg,
        op_lat, op_lon, node_lat, node_lon, node_speed, _skew(rng, n, "small"),
        addr, 0, slice_name,
    )


def _real_track(rng: np.random.Generator, t, node_lat, node_lon, node_speed, slice_name: str):
    if rng.random() < 0.45:
        return _real_loiter(rng, t, node_lat, node_lon, node_speed, slice_name)
    n = len(t)
    lat0, lon0 = move_by(DEFAULT_NODE[0], DEFAULT_NODE[1], float(rng.uniform(30.0, 700.0)), float(rng.uniform(0, 360)))
    heading = np.empty(n)
    heading[0] = rng.uniform(0.0, 360.0)
    for i in range(1, n):
        heading[i] = (heading[i - 1] + rng.normal(0.0, 6.0)) % 360.0
    speed = np.clip(rng.uniform(0.0, 14.0) + np.cumsum(rng.normal(0.0, 0.12, n)), 0.0, 22.0)
    lat, lon = _integrate(lat0, lon0, t, speed, heading)
    gps_n = rng.normal(0.0, 2.5, n)
    gps_e = rng.normal(0.0, 2.5, n)
    dlat, dlon = meters_to_offset(float(np.mean(lat)), 1.0, 1.0)
    lat_r = lat + gps_n * dlat
    lon_r = lon + gps_e * dlon
    takeoff = float(rng.uniform(200.0, 400.0))
    height = np.clip(rng.uniform(15.0, 90.0) + np.cumsum(rng.normal(0.0, 0.35, n)), 5.0, 140.0)
    alt = takeoff + height + rng.normal(0.0, 0.8, n)
    dt = np.gradient(t)
    vspeed = np.gradient(height) / np.maximum(dt, 1e-3) + rng.normal(0.0, 0.15, n)
    reported = np.clip(speed + rng.normal(0.0, 0.25, n), 0.0, None)
    hdg = (heading + rng.normal(0.0, 3.0, n)) % 360.0
    op_lat = np.full(n, lat[0])
    op_lon = np.full(n, lon[0])
    # Pilot stands at the launch point, a few metres off the first fix.
    op_lat = op_lat + rng.normal(0.0, 12.0) * dlat
    op_lon = op_lon + rng.normal(0.0, 12.0) * dlon
    true_range = haversine_m(node_lat, node_lon, lat, lon)
    rssi = _rssi_at(true_range, _tx_dbm(rng), rng, n, fade=True)
    addr = 1.0 if rng.random() < 0.20 else -1.0
    return _pack(
        t, rssi, lat_r, lon_r, alt, height, reported, vspeed, hdg,
        op_lat, op_lon, node_lat, node_lon, node_speed, _skew(rng, n, "small"),
        addr, 0, slice_name,
    )


def _circle_motion(rng, n, t, center_lat, center_lon, radius):
    omega = float(rng.uniform(0.12, 0.45))
    theta = rng.uniform(0.0, 2.0 * np.pi) + omega * (t - t[0])
    north = radius * np.cos(theta)
    east = radius * np.sin(theta)
    lat, lon = move_ne(center_lat, center_lon, north, east)
    speed = np.full(n, omega * radius)
    # Velocity of (r cos, r sin) is (-r sin, r cos) * omega.
    hdg = (np.degrees(np.arctan2(np.cos(theta), -np.sin(theta))) + 360.0) % 360.0
    climb = float(rng.normal(0.0, 0.15))
    height = np.clip(rng.uniform(25.0, 70.0) + climb * (t - t[0]), 8.0, 120.0)
    takeoff = float(rng.uniform(220.0, 360.0))
    alt = takeoff + height + rng.normal(0.0, 0.4, n)
    vspeed = np.full(n, climb) + rng.normal(0.0, 0.05, n)
    op_lat = np.full(n, center_lat)
    op_lon = np.full(n, center_lon)
    return lat, lon, alt, height, speed, vspeed, hdg, op_lat, op_lon


def _inconsistent(rng, n, t, kind: str):
    """Claimed track whose reported kinematics do not match the fixes."""
    lat = np.empty(n)
    lon = np.empty(n)
    lat[0], lon[0] = move_by(
        DEFAULT_NODE[0], DEFAULT_NODE[1], float(rng.uniform(80.0, 900.0)), float(rng.uniform(0, 360))
    )
    if kind == "teleport":
        for i in range(1, n):
            lat[i], lon[i] = move_by(lat[i - 1], lon[i - 1], float(rng.uniform(80.0, 900.0)), float(rng.uniform(0, 360)))
        speed = rng.uniform(0.0, 22.0, n)
        hdg = rng.uniform(0.0, 360.0, n)
    else:
        for i in range(1, n):
            lat[i], lon[i] = move_by(lat[i - 1], lon[i - 1], float(rng.uniform(2.0, 28.0)), float(rng.uniform(0, 360)))
        speed = np.clip(rng.uniform(0.0, 18.0) + np.cumsum(rng.normal(0.0, 1.2, n)), 0.0, 40.0)
        hdg = np.cumsum(rng.uniform(-25.0, 25.0, n)) % 360.0
    alt = np.clip(rng.uniform(40.0, 250.0) + np.cumsum(rng.normal(0.0, 6.0, n)), 0.0, 500.0)
    height = np.clip(rng.uniform(10.0, 80.0) + np.cumsum(rng.normal(0.0, 5.0, n)), 0.0, 220.0)
    vspeed = rng.normal(0.0, 2.5, n)
    op_lat = np.empty(n)
    op_lon = np.empty(n)
    op_lat[0], op_lon[0] = lat[0], lon[0]
    for i in range(1, n):
        if rng.random() < 0.15:
            op_lat[i], op_lon[i] = move_by(lat[i], lon[i], float(rng.uniform(50.0, 400.0)), float(rng.uniform(0, 360)))
        else:
            op_lat[i], op_lon[i] = op_lat[i - 1], op_lon[i - 1]
    return lat, lon, alt, height, speed, vspeed, hdg, op_lat, op_lon


def _emitter(rng, t, node_lat, node_lon, near_m: float = 10.0, far_m: float = 80.0):
    em_lat, em_lon = move_by(
        DEFAULT_NODE[0], DEFAULT_NODE[1], float(rng.uniform(near_m, far_m)), float(rng.uniform(0, 360))
    )
    em_range = haversine_m(node_lat, node_lon, np.full(len(t), em_lat), np.full(len(t), em_lon))
    rssi = _rssi_at(em_range, _tx_dbm(rng), rng, len(t), fade=True)
    return em_lat, em_lon, rssi


def _loiter_motion(rng, n, t, center_lat, center_lon):
    """Slow motion that stays on top of a fixed transmitter.

    Reported speed matches that motion, and the fix then gets the same GNSS
    scatter as a real aircraft. Received power still comes from the transmitter.
    """
    lat = np.empty(n)
    lon = np.empty(n)
    lat[0], lon[0] = center_lat, center_lon
    heading = np.empty(n)
    speed = np.empty(n)
    heading[0] = rng.uniform(0.0, 360.0)
    speed[0] = rng.uniform(0.0, 1.2)
    for i in range(1, n):
        speed[i] = float(np.clip(speed[i - 1] + rng.normal(0.0, 0.25), 0.0, 3.0))
        heading[i] = (heading[i - 1] + rng.normal(0.0, 8.0)) % 360.0
        step = speed[i] * float(t[i] - t[i - 1])
        lat[i], lon[i] = move_by(lat[i - 1], lon[i - 1], step, float(heading[i]))
        if float(haversine_m(center_lat, center_lon, lat[i], lon[i])) > 12.0:
            lat[i], lon[i] = lat[i - 1], lon[i - 1]
            speed[i] = 0.0
    height = np.clip(rng.uniform(12.0, 45.0) + np.cumsum(rng.normal(0.0, 0.2, n)), 5.0, 80.0)
    alt = float(rng.uniform(220.0, 360.0)) + height + rng.normal(0.0, 0.8, n)
    vspeed = np.gradient(height) / np.maximum(np.gradient(t), 1e-3) + rng.normal(0.0, 0.12, n)
    reported = np.clip(speed + rng.normal(0.0, 0.25, n), 0.0, None)
    hdg = (heading + rng.normal(0.0, 3.0, n)) % 360.0
    return lat, lon, alt, height, reported, vspeed, hdg, np.full(n, center_lat), np.full(n, center_lon)


def _spoof_track(
    rng, t, node_lat, node_lon, node_speed, rssi_base, center, radius, kind, slice_name, addr, skew_spread,
):
    n = len(t)
    if kind == "loiter":
        motion = _loiter_motion(rng, n, t, center[0], center[1])
    elif kind in ("teleport", "drift"):
        motion = _inconsistent(rng, n, t, kind)
    else:
        motion = _circle_motion(rng, n, t, center[0], center[1], radius)
    lat, lon, alt, height, speed, vspeed, hdg, op_lat, op_lon = motion
    if kind in ("circle", "loiter"):
        # Same fix scatter a real GNSS report carries, so a smooth or slow path
        # is not labeled spoof just for how tidy the motion is.
        lat, lon = gnss_noise(rng, lat, lon)
        if kind == "circle":
            speed = np.clip(speed + rng.normal(0.0, 0.25, n), 0.0, None)
            hdg = (hdg + rng.normal(0.0, 3.0, n)) % 360.0
    rssi = np.clip(rssi_base + rng.normal(0.0, 0.6, n), -110.0, -20.0)
    return _pack(
        t, rssi, lat, lon, alt, height, speed, vspeed, hdg,
        op_lat, op_lon, node_lat, node_lon, node_speed, _skew(rng, n, skew_spread),
        addr, 1, slice_name,
    )


def _assign_cohort(tracks: list[TrackArrays]) -> None:
    if len(tracks) < 2:
        return
    n = len(tracks[0])
    rssi = np.column_stack([tr.rssi for tr in tracks])
    for index in range(len(tracks)):
        delta = np.abs(rssi - rssi[:, index : index + 1])
        tracks[index].cohort = np.sum(delta <= 3.0, axis=1).astype(np.float64)
        if tracks[index].cohort.shape[0] != n:
            raise RuntimeError("cohort length drifted")


def make_scene(rng: np.random.Generator, kind: str) -> list[TrackArrays]:
    """Build one scene of the requested kind."""
    jitter = 0.045 if kind in ("real", "real_pair", "spoof_colocated") else 0.012
    t = _timebase(rng, N_SAMPLES, jitter)
    node_lat, node_lon, node_speed = _node(rng, t)
    if kind == "real":
        return [_real_track(rng, t, node_lat, node_lon, node_speed, "real")]
    if kind == "real_pair":
        tracks = [
            _real_track(rng, t, node_lat, node_lon, node_speed, "real_pair"),
            _real_track(rng, t, node_lat, node_lon, node_speed, "real_pair"),
        ]
        _assign_cohort(tracks)
        return tracks

    if kind == "spoof_colocated":
        em_lat, em_lon, rssi_base = _emitter(rng, t, node_lat, node_lon, 15.0, 180.0)
    else:
        em_lat, em_lon, rssi_base = _emitter(rng, t, node_lat, node_lon)
    if kind == "spoof_teleport":
        tracks = [_one_spoof(rng, t, node_lat, node_lon, node_speed, rssi_base, em_lat, em_lon, "teleport", kind)]
    elif kind == "spoof_drift":
        tracks = [_one_spoof(rng, t, node_lat, node_lon, node_speed, rssi_base, em_lat, em_lon, "drift", kind)]
    elif kind == "spoof_waypoint_far":
        center = move_by(em_lat, em_lon, float(rng.uniform(1200.0, 2800.0)), float(rng.uniform(0, 360)))
        tracks = [
            _spoof_track(
                rng, t, node_lat, node_lon, node_speed, rssi_base, center,
                float(rng.uniform(60.0, 180.0)), "circle", kind,
                1.0 if rng.random() < 0.8 else -1.0, "large" if rng.random() < 0.5 else "small",
            )
        ]
    elif kind == "spoof_colocated":
        tracks = [
            _spoof_track(
                rng, t, node_lat, node_lon, node_speed, rssi_base, (em_lat, em_lon),
                0.0, "loiter", kind, -1.0, "small",
            )
        ]
    elif kind == "spoof_universal_mac":
        tracks = [
            _spoof_track(
                rng, t, node_lat, node_lon, node_speed, rssi_base, (em_lat, em_lon),
                0.0, "drift", kind, -1.0, "small",
            )
        ]
    elif kind == "spoof_swarm":
        count = int(rng.integers(4, 8))
        tracks = []
        for _ in range(count):
            mode = "teleport" if rng.random() < 0.5 else "drift"
            tracks.append(
                _one_spoof(
                    rng, t, node_lat, node_lon, node_speed, rssi_base, em_lat, em_lon, mode, kind
                )
            )
    else:
        raise ValueError(f"unknown slice {kind}")
    _assign_cohort(tracks)
    return tracks


def _one_spoof(rng, t, node_lat, node_lon, node_speed, rssi_base, em_lat, em_lon, mode, slice_name):
    addr = 1.0 if rng.random() < 0.8 else -1.0
    spread = "large" if rng.random() < 0.5 else "small"
    return _spoof_track(
        rng, t, node_lat, node_lon, node_speed, rssi_base, (em_lat, em_lon),
        0.0, mode, slice_name, addr, spread,
    )


def generate_tracks(kind: str, n_tracks: int, seed: int) -> list[TrackArrays]:
    """Draw scenes until `n_tracks` tracks of this slice exist."""
    rng = np.random.default_rng(seed)
    tracks: list[TrackArrays] = []
    guard = 0
    while len(tracks) < n_tracks:
        tracks.extend(make_scene(rng, kind))
        guard += 1
        if guard > n_tracks * 4 + 10:
            raise RuntimeError(f"could not fill slice {kind}")
    return tracks[:n_tracks]


def tracks_to_windows(tracks: list[TrackArrays]):
    features = []
    labels = []
    slices = []
    for track in tracks:
        windows = windows_from_track(track)
        if len(windows) == 0:
            continue
        features.append(windows)
        labels.append(np.full(len(windows), track.label, dtype=np.int64))
        slices.append(np.full(len(windows), track.slice_name))
    if not features:
        return (
            np.zeros((0, 8, N_FEATURES), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=str),
        )
    return np.concatenate(features), np.concatenate(labels), np.concatenate(slices)


def training_counts(scenes: int) -> dict[str, int]:
    total = sum(TRAIN_WEIGHTS.values())
    scale = max(scenes, total) / total
    return {name: max(1, int(round(weight * scale))) for name, weight in TRAIN_WEIGHTS.items()}


def make_dataset(counts: dict[str, int], seed: int):
    tracks: list[TrackArrays] = []
    for index, (name, count) in enumerate(counts.items()):
        tracks.extend(generate_tracks(name, count, seed + 1009 * (index + 1)))
    return tracks_to_windows(tracks)


def make_eval_set(n_per_slice: int, seed: int):
    return make_dataset({name: n_per_slice for name in SLICES}, seed)
