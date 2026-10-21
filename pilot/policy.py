"""A two-layer MLP policy with hand-written backprop and Adam, in NumPy.

The policy is factorised: one categorical over N_ACTIONS per broken branch,
conditioned on that branch's neighbours and whether tests cover it.
"""

import numpy as np

from .env import N_ACTIONS, OBS_DIM


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class MLP:
    def __init__(self, rng: np.random.Generator, hidden: int = 128,
                 in_dim: int = OBS_DIM, out_dim: int = N_ACTIONS):
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 0.01, (hidden, out_dim)),
            "b2": np.zeros(out_dim),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def copy(self) -> "MLP":
        clone = MLP.__new__(MLP)
        clone.params = {k: v.copy() for k, v in self.params.items()}
        clone.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        clone.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        clone.t = 0
        return clone

    def forward(self, x: np.ndarray):
        """x: (N, OBS_DIM) -> logits (N, A), cache for backward."""
        p = self.params
        pre = x @ p["W1"] + p["b1"]
        h = np.maximum(pre, 0)
        logits = h @ p["W2"] + p["b2"]
        return logits, (x, pre, h)

    def backward(self, dlogits: np.ndarray, cache) -> dict:
        x, pre, h = cache
        p = self.params
        grads = {"W2": h.T @ dlogits, "b2": dlogits.sum(0)}
        dh = dlogits @ p["W2"].T
        dh[pre <= 0] = 0
        grads["W1"] = x.T @ dh
        grads["b1"] = dh.sum(0)
        return grads

    def adam_step(self, grads: dict, lr: float, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
