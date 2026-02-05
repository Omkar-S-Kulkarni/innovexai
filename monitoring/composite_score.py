# composite_score.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CompositeDriftScore:
    """
    Compute a composite drift score using multiple signals:
    - output drift (KL, PSI, Wasserstein)
    - confidence
    - entropy
    - stability
    """

    def __init__(self, output_weight=0.4, confidence_weight=0.2,
                 entropy_weight=0.2, stability_weight=0.2):
        self.weights = {
            "output": output_weight,
            "confidence": confidence_weight,
            "entropy": entropy_weight,
            "stability": stability_weight
        }
        self.scaler = MinMaxScaler()

    def normalize_signal(self, signal):
        """Normalize a single signal array to 0-1"""
        signal = np.array(signal).reshape(-1, 1)
        normalized = self.scaler.fit_transform(signal).flatten()
        return normalized

    def compute_global_score(self, output_drift, confidence, entropy, stability):
        """Compute single global composite score"""
        # Normalize each signal
        norm_output = self.normalize_signal(output_drift)
        norm_conf = self.normalize_signal(confidence)
        norm_entropy = self.normalize_signal(entropy)
        norm_stability = self.normalize_signal(stability)

        # Weighted sum
        score = (
            self.weights["output"] * norm_output +
            self.weights["confidence"] * norm_conf +
            self.weights["entropy"] * norm_entropy +
            self.weights["stability"] * norm_stability
        )
        return score

    def compute_slice_scores(self, slice_signals):
        """
        Compute composite scores per slice.
        slice_signals: dict of slice_name -> dict of signals
        Example:
        slice_signals = {
            "slice_0": {"output": [...], "confidence": [...], "entropy": [...], "stability": [...]},
            "slice_1": {...},
        }
        Returns: dict of slice_name -> composite score array
        """
        slice_scores = {}
        for slice_name, signals in slice_signals.items():
            slice_scores[slice_name] = self.compute_global_score(
                signals["output"], signals["confidence"],
                signals["entropy"], signals["stability"]
            )
        return slice_scores
