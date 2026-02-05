# monitoring/audit_report_generator.py

import json
from datetime import datetime
import numpy as np


class AuditReportGenerator:
    def __init__(
        self,
        model_metadata: dict,
        window_manager,
        alert_engine,
        blind_spot_matrix: dict,
    ):
        self.model_metadata = model_metadata
        self.window_manager = window_manager
        self.alert_engine = alert_engine
        self.blind_spot_matrix = blind_spot_matrix

    # -----------------------------
    # Core Report Builder
    # -----------------------------
    def generate(
        self,
        composite_score_series,
        drift_regimes,
        slice_scores,
        signal_deltas,
        confidence_metrics,
        entropy_metrics,
        stability_metrics,
    ):
        now = datetime.utcnow().isoformat()

        report = {
            "generated_at": now,
            "model_metadata": self.model_metadata,
            "monitoring_window": self.window_manager.describe(),
            "overall_drift_risk": float(np.max(composite_score_series)),
            "current_drift_regime": drift_regimes[-1],
            "affected_slices": self._top_slices(slice_scores),
            "what_changed": self._what_changed(signal_deltas),
            "when": self._temporal_narrative(composite_score_series, drift_regimes),
            "where": self._slice_localization(slice_scores),
            "confidence_level": self._audit_confidence(
                composite_score_series, slice_scores
            ),
            "alert_history": self.alert_engine.history(),
            "known_blind_spots": self.blind_spot_matrix,
        }

        return report

    # -----------------------------
    # Helpers
    # -----------------------------
    def _what_changed(self, deltas):
        explanations = []

        for k, v in deltas.items():
            if abs(v) > 0.05:
                explanations.append(
                    f"{k} changed by {round(v*100,1)}%"
                )

        return explanations

    def _temporal_narrative(self, scores, regimes):
        first_alert = np.argmax(scores > 0.7) if np.any(scores > 0.7) else None

        return {
            "first_warning_index": int(first_alert) if first_alert is not None else None,
            "windows_monitored": len(scores),
            "regime_timeline": regimes,
        }

    def _slice_localization(self, slice_scores):
        localized = {}

        global_mean = np.mean(list(slice_scores.values()))

        for name, score in slice_scores.items():
            if score > 1.5 * global_mean:
                localized[name] = {
                    "risk": round(score, 3),
                    "relative_to_global": round(score / global_mean, 2),
                }

        return localized

    def _audit_confidence(self, scores, slice_scores):
        agreement = np.std(scores)
        slice_variance = np.std(list(slice_scores.values()))

        confidence = 1 - min(1.0, agreement + slice_variance)

        return {
            "level": "High" if confidence > 0.75 else "Medium" if confidence > 0.5 else "Low",
            "score": round(confidence, 2),
        }

    def _top_slices(self, slice_scores, k=3):
        return dict(
            sorted(slice_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        )

    # -----------------------------
    # Export
    # -----------------------------
    def export_json(self, report, path):
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
