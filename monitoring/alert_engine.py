# alert_engine.py
"""
Alert Engine for Phase 7 — Non-blocking slice-level drift alerts.
Integrates slice risk rankings and explanations with persistence
and cooldown logic.
"""

import uuid
from collections import defaultdict
from monitoring.severity_scoring import compute_severity, should_escalate
from monitoring.cooldown_manager import CooldownManager


class AlertEngine:
    def __init__(
        self,
        persistence_windows: int = 2,
        cooldown_windows: int = 3,
        top_k_slices: int = 3
    ):
        """
        persistence_windows: Number of consecutive windows a slice must exceed
                             threshold to trigger alert
        cooldown_windows: Number of windows to suppress repeated alerts
        top_k_slices: Number of highest-risk slices to evaluate per window
        """
        self.persistence_windows = persistence_windows
        self.cooldown = CooldownManager(cooldown_windows)
        self.top_k_slices = top_k_slices

        self.active_alerts = {}
        self.persistence_counter = defaultdict(int)

    def process(
        self,
        slice_rankings: list,
        slice_explanations: dict,
        window_id: int
    ) -> list:
        """Evaluate top slices and trigger alerts if needed."""
        alerts_fired = []

        # Tick cooldowns at start of window
        self.cooldown.tick()

        for entry in slice_rankings[: self.top_k_slices]:
            slice_name = entry["slice"]
            risk_score = entry["risk_score"]
            severity = entry["severity"]

            if severity == "LOW":
                continue

            alert_key = slice_name

            # Require persistence over multiple windows
            self.persistence_counter[alert_key] += 1
            if self.persistence_counter[alert_key] < self.persistence_windows:
                continue

            # Skip if slice is in cooldown
            if self.cooldown.is_in_cooldown(alert_key):
                continue

            explanation = slice_explanations.get(slice_name, {}).get(
                "explanation", "Risk threshold exceeded"
            )

            alert = self._create_or_update_alert(
                slice_name, severity, risk_score, explanation, window_id
            )

            alerts_fired.append(alert)
            self.cooldown.trigger_cooldown(alert_key)

        return alerts_fired

    def _create_or_update_alert(
        self,
        slice_name: str,
        severity: str,
        risk_score: float,
        explanation: str,
        window_id: int
    ) -> dict:
        """Create a new alert or update an existing one."""
        existing = self.active_alerts.get(slice_name)

        if existing:
            # Escalate severity if needed
            if should_escalate(existing["severity"], severity):
                existing["severity"] = severity
                existing["risk_score"] = risk_score
                existing["last_updated"] = window_id
                existing["state"] = "ESCALATED"
            else:
                existing["last_updated"] = window_id
            return existing

        alert = {
            "alert_id": str(uuid.uuid4()),
            "slice": slice_name,
            "severity": severity,
            "risk_score": risk_score,
            "state": "ACTIVE",
            "first_seen": window_id,
            "last_updated": window_id,
            "explanation": explanation
        }

        self.active_alerts[slice_name] = alert
        return alert

    def resolve_alert(self, slice_name: str, window_id: int):
        """Resolve an alert for a given slice."""
        if slice_name in self.active_alerts:
            self.active_alerts[slice_name]["state"] = "RESOLVED"
            self.active_alerts[slice_name]["last_updated"] = window_id
            del self.active_alerts[slice_name]

    def get_active_alerts(self) -> list:
        """Return a list of active alerts."""
        return list(self.active_alerts.values())
