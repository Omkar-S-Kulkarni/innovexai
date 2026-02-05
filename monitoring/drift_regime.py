# drift_regime.py

import numpy as np

class DriftRegimeDetector:
    """
    Detect drift regime types:
    - gradual
    - sudden
    - localized
    - oscillatory
    """

    def __init__(self, gradual_thresh=0.01, sudden_thresh=0.05, localized_thresh=0.2, osc_thresh=0.05):
        self.gradual_thresh = gradual_thresh
        self.sudden_thresh = sudden_thresh
        self.localized_thresh = localized_thresh
        self.osc_thresh = osc_thresh

    def detect_gradual(self, score_series):
        """Detect gradual drift based on slope"""
        slope = np.gradient(score_series)
        return np.any(slope > self.gradual_thresh)

    def detect_sudden(self, score_series):
        """Detect sudden drift based on delta between consecutive scores"""
        delta = np.diff(score_series)
        return np.any(delta > self.sudden_thresh)

    def detect_localized(self, slice_scores, global_score):
        """
        Detect localized drift if any slice's score deviates strongly from global
        slice_scores: dict of slice_name -> array of scores
        global_score: array of global scores
        """
        localized_flags = {}
        for slice_name, scores in slice_scores.items():
            deviation = np.mean(scores) - np.mean(global_score)
            localized_flags[slice_name] = deviation > self.localized_thresh
        return localized_flags

    def detect_oscillatory(self, score_series):
        """Detect oscillatory drift by checking sign changes in slope"""
        second_diff_sign = np.sign(np.diff(np.sign(np.diff(score_series))))
        return np.std(second_diff_sign) > self.osc_thresh

    def classify(self, global_score_series, slice_scores=None):
        """
        Returns dict of detected drift regimes
        """
        regimes = {
            "gradual": self.detect_gradual(global_score_series),
            "sudden": self.detect_sudden(global_score_series),
            "oscillatory": self.detect_oscillatory(global_score_series),
            "localized": {}
        }
        if slice_scores is not None:
            regimes["localized"] = self.detect_localized(slice_scores, global_score_series)
        return regimes
