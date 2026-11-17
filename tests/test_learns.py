"""A short train should catch obvious false tracks and stay quiet on real ones."""

import numpy as np
import torch

from spoofdet.metrics import auroc
from spoofdet.model import SpoofNet
from spoofdet.sim import generate_tracks, tracks_to_windows


def test_short_train_separates_teleports_from_real_tracks():
    torch.manual_seed(0)
    real = generate_tracks("real", 40, seed=10)
    teleport = generate_tracks("spoof_teleport", 40, seed=11)
    features, labels, _ = tracks_to_windows(real + teleport)
    model = SpoofNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels.astype(np.float32))
    first = None
    for _ in range(6):
        opt.zero_grad()
        logit = model(x)
        loss = loss_fn(logit, y)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
    assert float(loss.item()) < first
    with torch.no_grad():
        scores = torch.sigmoid(model(x)).numpy()
    assert auroc(labels, scores) > 0.9
