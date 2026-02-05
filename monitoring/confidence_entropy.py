import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import entropy
from dataclasses import dataclass

def _valid_probs(probs):
    return (
        probs is not None
        and isinstance(probs, np.ndarray)
        and probs.ndim == 2
        and probs.shape[0] > 0
        and probs.shape[1] > 1
        and np.all(np.isfinite(probs))
    )

@dataclass
class ConfidenceEntropySignals:
    """
    Phase 4 — Silent degradation detection
    Monitors model cognition, not accuracy.
    """

    # =========================================================
    # -------- CONFIDENCE SIGNALS ------------------------------
    # =========================================================

    @staticmethod
    def mean_top1_confidence(probs):
        if not _valid_probs(probs):
            return 0.0

        return float(np.mean(np.max(probs, axis=1)))

    @staticmethod
    def confidence_variance(probs):
        if not _valid_probs(probs):
            return 0.0

        return float(np.var(np.max(probs, axis=1)))
    
    @staticmethod
    def confidence_decay_rate(
        confidence_series: np.ndarray,
        time_index: np.ndarray
    ) -> float:
        if len(confidence_series) < 2:
            return 0.0
        slope = np.polyfit(time_index, confidence_series, 1)[0]
        return slope

    @staticmethod
    def probability_margin(probs):
        if not _valid_probs(probs):
            return 0.0

        sorted_probs = np.sort(probs, axis=1)
        return float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2]))

    @staticmethod
    def low_confidence_mass(probs, threshold=0.5):
        if not _valid_probs(probs):
            return 0.0

        top1 = np.max(probs, axis=1)
        return float(np.mean(top1 < threshold))


    # =========================================================
    # -------- ENTROPY SIGNALS --------------------------------
    # =========================================================

    @staticmethod
    def sample_entropy(probs, eps=1e-8):
        if not _valid_probs(probs):
            return np.array([])

        p = np.clip(probs, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)


    @staticmethod
    def mean_entropy(entropies):
        if entropies is None or len(entropies) == 0:
            return 0.0

        return float(np.mean(entropies))


    @staticmethod
    def entropy_trend(entropies, time_idx):
        if entropies is None or len(entropies) < 2:
            return 0.0

        return float(np.polyfit(time_idx, entropies, 1)[0])

    @staticmethod
    def entropy_confidence_divergence(probs: np.ndarray) -> float:
        if not _valid_probs(probs):
            return 0.0

        ent = ConfidenceEntropySignals.sample_entropy(probs)
        conf = np.max(probs, axis=1)

        if np.std(ent) < 1e-6 or np.std(conf) < 1e-6:
            return 0.0

        return float(np.corrcoef(ent, conf)[0, 1])

    # =========================================================
    # -------- ADVANCED DETECTORS ------------------------------
    # =========================================================

    @staticmethod
    def entropy_shock(entropies, z_thresh=3.0):
        if entropies is None or len(entropies) < 5:
            return False

        z = (entropies - np.mean(entropies)) / (np.std(entropies) + 1e-8)
        return bool(np.any(z > z_thresh))

    @staticmethod
    def class_conditional_entropy(
        probs: np.ndarray,
        preds: np.ndarray
    ) -> Dict[int, float]:
        if not _valid_probs(probs):
            return {}

        ent = ConfidenceEntropySignals.sample_entropy(probs)
        result = {}

        for cls in np.unique(preds):
            mask = preds == cls
            if np.sum(mask) < 2:
                result[int(cls)] = 0.0
            else:
                result[int(cls)] = float(np.mean(ent[mask]))

        return result


    @staticmethod
    def confidence_shape_collapse(probs, var_thresh=1e-3):
        if not _valid_probs(probs):
            return False

        return np.var(probs) < var_thresh

    @staticmethod
    def silent_failure_typing(
        mean_conf: float,
        mean_ent: float,
        margin: float,
        low_conf_mass: float,
        conf_low: float = 0.6,
        ent_high: float = 1.0,
        margin_low: float = 0.15,
        low_conf_high: float = 0.3
    ) -> str:
        """
        Classifies silent failure modes using confidence & entropy signals.
        Thresholds are heuristics and can be tuned.
        """

        if mean_conf < conf_low and mean_ent > ent_high:
            return "Domain Shift"

        if margin < margin_low and mean_conf >= conf_low:
            return "Boundary Confusion"

        if low_conf_mass > low_conf_high:
            return "Data Noise Accumulation"

        if mean_ent > ent_high and mean_conf >= conf_low:
            return "Class Overlap"

        return "Healthy / Stable"
