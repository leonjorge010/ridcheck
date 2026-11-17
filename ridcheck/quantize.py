"""Post-training int8 export.

Hidden layers are integer. The last layer turns the int32 accumulator into a
float logit and a probability. Python and the C firmware share the rounding
rule in `qround`: half away from zero.
"""

from __future__ import annotations

import numpy as np

from ridcheck.features import N_FEATURES, WINDOW

Q_SHIFT = 16


def qround(values) -> np.ndarray:
    """Round half away from zero. Used when baking weights on the host."""
    data = np.asarray(values, dtype=np.float64)
    rounded = np.where(data >= 0.0, np.floor(data + 0.5), np.ceil(data - 0.5))
    return rounded.astype(np.int64)


def qround_f32(values) -> np.ndarray:
    """float32 half-away-from-zero, matching floorf/ceilf on the device."""
    data = np.asarray(values, dtype=np.float32)
    half = np.float32(0.5)
    bumped = np.where(data >= np.float32(0.0), data + half, data - half).astype(np.float32)
    rounded = np.where(data >= np.float32(0.0), np.floor(bumped), np.ceil(bumped))
    return rounded.astype(np.int64)


def requant(acc: np.ndarray, mult: int) -> np.ndarray:
    scaled = (acc.astype(np.int64) * np.int64(mult) + (1 << (Q_SHIFT - 1))) >> Q_SHIFT
    return scaled


class QuantLayer:
    def __init__(self, weight, bias, mult, scale_acc, relu: bool):
        self.weight = np.asarray(weight, dtype=np.int8)
        self.bias = np.asarray(bias, dtype=np.int32)
        self.mult = int(mult)
        self.scale_acc = float(scale_acc)
        self.relu = relu

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])


class QuantModel:
    def __init__(self, layers: list[QuantLayer], feature_scale: np.ndarray, threshold: float):
        self.layers = layers
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.threshold = float(threshold)

    def quantize_features(self, features: np.ndarray) -> np.ndarray:
        flat = np.asarray(features, dtype=np.float32).reshape(-1, WINDOW * N_FEATURES)
        scale = np.tile(self.feature_scale.astype(np.float32), WINDOW)
        ratio = (flat / scale).astype(np.float32)
        quantized = qround_f32(ratio)
        return np.clip(quantized, -127, 127).astype(np.int8)

    def forward_acc(self, quantized: np.ndarray):
        """Return probabilities and the last-layer int32 accumulator."""
        current = quantized
        last_acc = None
        for index, layer in enumerate(self.layers):
            acc = current.astype(np.int32) @ layer.weight.astype(np.int32).T
            acc = acc + layer.bias
            last = index == len(self.layers) - 1
            if last:
                last_acc = acc.reshape(-1).astype(np.int32)
                logit = last_acc.astype(np.float32) * np.float32(layer.scale_acc)
                prob = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
                return prob.astype(np.float64), last_acc
            activated = requant(acc, layer.mult)
            if layer.relu:
                activated = np.clip(activated, 0, 127)
            else:
                activated = np.clip(activated, -128, 127)
            current = activated.astype(np.int8)
        raise RuntimeError("model has no layers")

    def score(self, features: np.ndarray) -> np.ndarray:
        quantized = self.quantize_features(features)
        prob, _ = self.forward_acc(quantized)
        return prob


def _layer_from_float(weight, bias, scale_in: float, calib_x: np.ndarray, relu: bool, last: bool):
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    peak = float(np.max(np.abs(weight)))
    scale_w = max(peak / 127.0, 1e-8)
    w_q = np.clip(qround(weight / scale_w), -127, 127).astype(np.int8)
    scale_acc = float(scale_in) * scale_w
    b_q = np.clip(qround(bias / scale_acc), np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
    acc = calib_x.astype(np.int32) @ w_q.astype(np.int32).T + b_q
    if last:
        return QuantLayer(w_q, b_q, 0, scale_acc, False), None, scale_acc
    pre = acc.astype(np.float64) * scale_acc
    span = float(np.max(np.abs(pre)))
    scale_out = max(span * 1.05, 1e-4) / 127.0
    mult = int(qround(np.float64(scale_acc / scale_out * (1 << Q_SHIFT))))
    mult = int(np.clip(mult, 1, np.iinfo(np.int32).max))
    activated = requant(acc, mult)
    if relu:
        activated = np.clip(activated, 0, 127)
    else:
        activated = np.clip(activated, -128, 127)
    return QuantLayer(w_q, b_q, mult, scale_acc, relu), activated.astype(np.int8), scale_out


def quantize_network(state: dict, feature_scale: np.ndarray, calib: np.ndarray) -> QuantModel:
    """Quantize a SpoofNet state_dict. `calib` is float windows, shape (N, 8, 16)."""
    scale = np.asarray(feature_scale, dtype=np.float32).astype(np.float64)
    flat = np.asarray(calib, dtype=np.float32).reshape(len(calib), -1)
    ratio = (flat / np.tile(scale.astype(np.float32), WINDOW)).astype(np.float32)
    quantized = np.clip(qround_f32(ratio), -127, 127).astype(np.int8)

    def tensor(key):
        value = state[key]
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    w1 = tensor("net.0.weight")
    b1 = tensor("net.0.bias")
    folded = w1 * np.tile(scale, WINDOW)[None, :]
    layer1, hidden, scale_h1 = _layer_from_float(folded, b1, 1.0, quantized, True, False)

    layer2, hidden2, scale_h2 = _layer_from_float(
        tensor("net.2.weight"), tensor("net.2.bias"), scale_h1, hidden, True, False
    )
    layer3, _, _ = _layer_from_float(
        tensor("net.4.weight"), tensor("net.4.bias"), scale_h2, hidden2, False, True
    )
    return QuantModel([layer1, layer2, layer3], scale.astype(np.float32), threshold=0.5)


def feature_scales(features: np.ndarray) -> np.ndarray:
    """Per-feature int8 scale from the 99.5th percentile of |feature|."""
    flat = np.asarray(features, dtype=np.float64).reshape(-1, N_FEATURES)
    peak = np.percentile(np.abs(flat), 99.5, axis=0)
    peak = np.maximum(peak, 0.25)
    return (peak / 127.0).astype(np.float32)
