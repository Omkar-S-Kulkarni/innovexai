import numpy as np
from scipy.stats import entropy, wasserstein_distance


class StabilityEvaluator:
    """
    Determines whether a window is stable enough
    to be used as a rolling reference.
    """

    def __init__(
        self,
        kl_threshold=0.15,
        psi_threshold=0.2,
        entropy_slope_threshold=0.002,
        low_conf_mass_threshold=0.25,
        tau=0.4,
    ):
        self.kl_threshold = kl_threshold
        self.psi_threshold = psi_threshold
        self.entropy_slope_threshold = entropy_slope_threshold
        self.low_conf_mass_threshold = low_conf_mass_threshold
        self.tau = tau

    @staticmethod
    def _hist(p, bins=20):
        h, _ = np.histogram(p, bins=bins, density=True)
        return h + 1e-8

    def kl(self, ref, cur):
        return entropy(self._hist(ref), self._hist(cur))

    def psi(self, ref, cur, bins=10):
        ref_h, _ = np.histogram(ref, bins=bins)
        cur_h, _ = np.histogram(cur, bins=bins)
        ref_p = ref_h / len(ref)
        cur_p = cur_h / len(cur)
        return np.sum((ref_p - cur_p) * np.log((ref_p + 1e-8) / (cur_p + 1e-8)))

    def low_conf_mass(self, probs):
        return (probs < self.tau).mean()

    def is_stable(
        self,
        reference_probs,
        current_probs,
        entropy_slope=0.0,
    ):
        kl = self.kl(reference_probs, current_probs)
        psi = self.psi(reference_probs, current_probs)
        low_conf = self.low_conf_mass(current_probs)

        stable = (
            kl < self.kl_threshold
            and psi < self.psi_threshold
            and abs(entropy_slope) < self.entropy_slope_threshold
            and low_conf < self.low_conf_mass_threshold
        )

        return stable, {
            "kl": kl,
            "psi": psi,
            "low_conf_mass": low_conf,
            "entropy_slope": entropy_slope,
        }
