"""Simulator invariants: motion tells and RSSI tells point the right way."""

import numpy as np

from ridcheck.features import N_FEATURES, WINDOW, haversine_m, step_features, windows_from_track
from ridcheck.sim import generate_tracks


def _mean_column(tracks, index):
    columns = []
    for track in tracks:
        features = step_features(track)
        columns.append(features[:, index])
    return np.concatenate(columns)


def test_distance_of_one_degree_latitude():
    distance = haversine_m(34.0, -117.3, 35.0, -117.3)
    assert 110_000 < distance < 112_000
    assert haversine_m(34.0, -117.3, 34.0, -117.3) == 0.0


def test_window_shape_and_finite_features():
    tracks = generate_tracks("real", 3, seed=1)
    windows = windows_from_track(tracks[0])
    assert windows.shape[1:] == (WINDOW, N_FEATURES)
    assert windows.shape[0] >= 1
    assert np.isfinite(windows).all()


def test_teleport_steps_exceed_real_motion():
    real = generate_tracks("real", 12, seed=2)
    teleport = generate_tracks("spoof_teleport", 12, seed=3)
    # step_ratio is column 8
    assert _mean_column(teleport, 8).mean() > _mean_column(real, 8).mean() + 0.4


def test_far_claim_breaks_rssi_more_than_a_colocated_path():
    far = generate_tracks("spoof_waypoint_far", 12, seed=4)
    near = generate_tracks("spoof_colocated", 12, seed=5)
    # rssi_residual is column 2
    assert np.mean(np.abs(_mean_column(far, 2))) > np.mean(np.abs(_mean_column(near, 2))) + 0.4


def test_colocated_rssi_matches_a_real_aircraft():
    real = generate_tracks("real", 12, seed=8)
    near = generate_tracks("spoof_colocated", 12, seed=9)
    real_residual = np.mean(np.abs(_mean_column(real, 2)))
    near_residual = np.mean(np.abs(_mean_column(near, 2)))
    assert abs(near_residual - real_residual) < 0.35


def test_scattered_nodes_disagree_when_the_claim_is_not_the_radio():
    real = generate_tracks("real", 8, seed=14)
    far = generate_tracks("spoof_waypoint_far", 8, seed=15)
    near = generate_tracks("spoof_colocated", 8, seed=16)
    real_db = np.mean([track.net_db.mean() for track in real])
    far_db = np.mean([track.net_db.mean() for track in far])
    near_db = np.mean([track.net_db.mean() for track in near])
    assert far_db > real_db + 4.0
    assert near_db < real_db + 2.0


def test_colocated_address_is_universal_and_labeled_spoof():
    tracks = generate_tracks("spoof_colocated", 6, seed=6)
    assert all(track.addr_local == -1.0 for track in tracks)
    assert all(track.label == 1 for track in tracks)


def test_swarm_shares_a_loud_cohort():
    tracks = generate_tracks("spoof_swarm", 8, seed=7)
    # One scene is 4–7 tracks; generate_tracks may span scenes. Cohort of a
    # swarm member is at least 2 somewhere in the drawn set.
    assert max(track.cohort.max() for track in tracks) >= 3
