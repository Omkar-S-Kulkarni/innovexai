import numpy as np
from collections import defaultdict, deque


class BehaviorFingerprintStore:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.store = defaultdict(lambda: deque(maxlen=history_size))

    def compute_fingerprint(self, probs):
        probs = np.asarray(probs)

        if probs.ndim != 2 or probs.shape[0] == 0:
            return None

        probs = np.clip(probs, 1e-9, 1.0)

        # Confidence + margin
        sorted_probs = np.sort(probs, axis=1)
        max_conf = sorted_probs[:, -1]

        if probs.shape[1] >= 2:
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            margin = np.zeros_like(max_conf)

        # Entropy
        entropy_vals = -np.sum(probs * np.log(probs), axis=1)

        fingerprint = {
            "mean_confidence": float(np.mean(max_conf)),
            "confidence_variance": float(np.var(max_conf)),
            "mean_entropy": float(np.mean(entropy_vals)),
            "entropy_variance": float(np.var(entropy_vals)),
            "mean_margin": float(np.mean(margin)),
            "margin_variance": float(np.var(margin)),
            "class_balance": probs.mean(axis=0).tolist(),
            "prediction_volatility": float(np.var(max_conf))
        }

        return fingerprint

    def update(self, slice_name, fingerprint):
        if fingerprint is not None:
            self.store[slice_name].append(fingerprint)

    def get_history(self, slice_name):
        return list(self.store[slice_name])

    def has_enough_history(self, slice_name, min_len=3):
        return len(self.store[slice_name]) >= min_len
