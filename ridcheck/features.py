"""Features a Remote ID receiver can compute for one track.

The columns follow the Totem-air detection record: RSSI, aircraft fix, speed,
heading, operator fix, and the receiver's own fix. Nothing here is a
transmitted frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW = 8
STRIDE = 4
N_FEATURES = 16
FEATURE_CLIP = 4.0

# 2.4 GHz free-space loss at 1 m. RSSI is compared with a 12 dBm reference,
# which is a typical beacon level, so the residual is zero when the claimed
# range and the received power agree.
TX_REF_DBM = 12.0
FSPL_1M_DB = 40.2

FEATURE_NAMES = (
    "rssi_norm",
    "log_claimed_range",
    "rssi_residual",
    "speed_reported",
    "speed_track",
    "speed_residual",
    "heading_error",
    "vspeed_residual",
    "step_ratio",
    "alt_gap_rate",
    "dt_center",
    "clock_skew",
    "log_operator_range",
    "addr_local",
    "cohort_log",
    "node_speed",
)


@dataclass
class TrackArrays:
    """One aircraft track, one row per received location fix."""

    t: np.ndarray
    rssi: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    height: np.ndarray
    speed: np.ndarray
    vspeed: np.ndarray
    hdg: np.ndarray
    op_lat: np.ndarray
    op_lon: np.ndarray
    node_lat: np.ndarray
    node_lon: np.ndarray
    node_speed: np.ndarray
    clock_skew: np.ndarray
    cohort: np.ndarray
    addr_local: float
    label: int
    slice_name: str

    def __len__(self) -> int:
        return int(self.t.shape[0])


def expected_rssi(range_m, tx_dbm: float = TX_REF_DBM):
    """Free-space RSSI (dBm) at 2.4 GHz for a claimed horizontal range."""
    distance = np.maximum(np.asarray(range_m, dtype=np.float64), 1.0)
    return tx_dbm - (FSPL_1M_DB + 20.0 * np.log10(distance))


def haversine_m(lat1, lon1, lat2, lon2):
    """Ground distance in metres between WGS84 points in degrees."""
    radius = 6_371_000.0
    phi1 = np.radians(np.asarray(lat1, dtype=np.float64))
    phi2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dphi = np.radians(np.asarray(lat2, dtype=np.float64) - np.asarray(lat1, dtype=np.float64))
    dlmb = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees, 0 = north, 90 = east."""
    phi1 = np.radians(np.asarray(lat1, dtype=np.float64))
    phi2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dlmb = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    east = np.sin(dlmb) * np.cos(phi2)
    north = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0


def wrap_deg(diff):
    """Wrap a degree difference into [-180, 180]."""
    return (np.asarray(diff, dtype=np.float64) + 180.0) % 360.0 - 180.0


def step_features(track: TrackArrays) -> np.ndarray:
    """Per-step features, shape (len(track) - 1, 16). Row i steps into sample i+1."""
    dt = np.diff(track.t.astype(np.float64))
    dt = np.clip(dt, 1e-3, None)
    lat0, lat1 = track.lat[:-1], track.lat[1:]
    lon0, lon1 = track.lon[:-1], track.lon[1:]
    step = haversine_m(lat0, lon0, lat1, lon1)
    bearing = bearing_deg(lat0, lon0, lat1, lon1)
    speed_track = step / dt
    reported = track.speed[1:].astype(np.float64)
    moving = step >= 1.5
    heading_error = np.where(moving, wrap_deg(track.hdg[1:] - bearing) / 180.0, 0.0)
    vertical_track = np.diff(track.alt.astype(np.float64)) / dt
    vspeed_residual = (track.vspeed[1:] - vertical_track) / 8.0
    step_ratio = (
        np.log1p(step) - np.log1p(np.maximum(reported, 0.0) * dt)
    ) / 3.0
    alt_gap_rate = np.diff((track.alt - track.height).astype(np.float64)) / 20.0
    claimed = haversine_m(track.node_lat[1:], track.node_lon[1:], lat1, lon1)
    rssi = track.rssi[1:].astype(np.float64)
    residual = (rssi - expected_rssi(claimed)) / 20.0
    operator_range = haversine_m(lat1, lon1, track.op_lat[1:], track.op_lon[1:])
    cohort = np.maximum(track.cohort[1:].astype(np.float64) - 1.0, 0.0)
    columns = [
        (rssi + 70.0) / 30.0,
        np.log1p(claimed) / np.log1p(2000.0),
        residual,
        reported / 20.0,
        speed_track / 20.0,
        (reported - speed_track) / 20.0,
        heading_error,
        vspeed_residual,
        step_ratio,
        alt_gap_rate,
        dt - 1.0,
        track.clock_skew[1:].astype(np.float64) / 10.0,
        np.log1p(operator_range) / np.log1p(2000.0),
        np.full(dt.shape, float(track.addr_local), dtype=np.float64),
        np.log1p(cohort) / np.log1p(7.0),
        track.node_speed[1:].astype(np.float64) / 2.0,
    ]
    features = np.column_stack(columns)
    if features.shape[1] != N_FEATURES:
        raise RuntimeError(f"expected {N_FEATURES} features, got {features.shape[1]}")
    return np.nan_to_num(features, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)


def clip_features(features: np.ndarray) -> np.ndarray:
    return np.clip(features, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)


def windows_from_features(
    features: np.ndarray, window: int = WINDOW, stride: int = STRIDE
) -> np.ndarray:
    """Stack consecutive steps into (n, window, 16) blocks, oldest first."""
    count = features.shape[0]
    if count < window:
        return np.zeros((0, window, N_FEATURES), dtype=np.float32)
    starts = range(0, count - window + 1, stride)
    stacked = np.stack([features[start : start + window] for start in starts])
    return clip_features(stacked)


def windows_from_track(track: TrackArrays) -> np.ndarray:
    return windows_from_features(step_features(track))


def addr_local_feature(mac: str, radio: str) -> float:
    """+1 when the address is locally assigned, -1 when it is a universal OUI.

    Wi-Fi: the locally administered bit. BLE: a static random address, whose
    top two bits are 11. This reads an address the receiver already heard.
    """
    octet = int(str(mac).split(":")[0], 16)
    if radio == "ble":
        return 1.0 if (octet & 0xC0) == 0xC0 else -1.0
    return 1.0 if (octet & 0x02) else -1.0
