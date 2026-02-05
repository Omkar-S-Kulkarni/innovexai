# cluster_risk_integration.py

from monitoring.slice_drift_metrics import compute_slice_metrics
from monitoring.slice_risk_ranker import rank_slices, explain_slice
from monitoring.behavior_fingerprint import BehaviorFingerprintStore
from monitoring.trend_acceleration import compute_trends
from monitoring.early_warning_state import EarlyWarningEngine


class ClusterRiskEngine:
    def __init__(self, history_size=10):
        self.fp_store = BehaviorFingerprintStore(history_size)
        self.early_warning = EarlyWarningEngine()

    def process(
        self,
        cluster_probs,
        global_probs,
        window_id
    ):
        """
        cluster_probs: dict {cluster_name: np.array probs}
        global_probs: np.array probs
        """

        # ---- Phase 8: fingerprints ----
        global_fp = self.fp_store.compute_fingerprint(global_probs)
        self.fp_store.update("global", global_fp)

        cluster_metrics = {}
        explanations = {}

        for cluster_name, probs in cluster_probs.items():
            fp = self.fp_store.compute_fingerprint(probs)
            self.fp_store.update(cluster_name, fp)

            history = self.fp_store.get_history(cluster_name)
            trend = compute_trends(history, "mean_entropy")
            ew_state = self.early_warning.update(cluster_name, trend)

            cluster_metrics[cluster_name] = {
                "mean_confidence": fp["mean_confidence"],
                "entropy": fp["mean_entropy"],
                "kl_divergence": 0.0,   # placeholder; filled by Phase 6
                "psi": 0.0,
                "wasserstein": 0.0
            }

            explanations[cluster_name] = {
                "explanation": f"Auto cluster in {ew_state} state"
            }

        # ---- Phase 6: ranking ----
        global_metrics = {
            "mean_confidence": global_fp["mean_confidence"],
            "entropy": global_fp["mean_entropy"],
            "kl_divergence": 0.0,
            "psi": 0.0,
            "wasserstein": 0.0
        }

        rankings = rank_slices(cluster_metrics, global_metrics)

        return {
            "rankings": rankings,
            "explanations": explanations
        }
    
