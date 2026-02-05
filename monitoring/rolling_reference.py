import numpy as np
from collections import deque
from datetime import datetime

from monitoring.stability import StabilityEvaluator
from monitoring.audit import ReferenceAuditLogger


class RollingReferenceManager:
    """
    Manages rolling, stability-gated references.
    """

    def __init__(
        self,
        reference_size=200,
        strategy="freeze",  # freeze | hard | canary
    ):
        self.reference_size = reference_size
        self.strategy = strategy

        self.reference_stable = None
        self.reference_recent = deque(maxlen=reference_size)
        self.reference_training = None  # read-only hook

        self.frozen = False
        self.last_update_time = None
        self.last_update_reason = "cold_start"

        self.stability = StabilityEvaluator()
        self.audit = ReferenceAuditLogger()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def set_training_reference(self, probs: np.ndarray):
        self.reference_training = probs.copy()

    def manual_reset(self):
        old_id = id(self.reference_stable)
        self.reference_stable = None
        self.reference_recent.clear()
        self.frozen = False

        self._audit(
            reason="manual_reset",
            old_ref=old_id,
            new_ref=None,
        )

    def update(self, current_probs, entropy_slope=0.0):
        """
        Called once per window.
        """
        if self.frozen:
            return

        self.reference_recent.extend(current_probs)

        if len(self.reference_recent) < self.reference_size:
            return

        candidate = np.array(self.reference_recent)

        if self.reference_stable is None:
            self.reference_stable = candidate.copy()
            self._commit("cold_start", None, candidate)
            return

        stable, metrics = self.stability.is_stable(
            self.reference_stable,
            candidate,
            entropy_slope,
        )

        if stable:
            self._commit("stable_update", self.reference_stable, candidate)
            self.reference_stable = candidate.copy()
        else:
            self._handle_instability(metrics)

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------
    def _handle_instability(self, metrics):
        if self.strategy == "freeze":
            self.frozen = True
            reason = "freeze_on_drift"

        elif self.strategy == "hard":
            self.reference_stable = None
            self.reference_recent.clear()
            reason = "hard_reset"

        elif self.strategy == "canary":
            reason = "canary_reference_building"

        self._audit(reason, self.reference_stable, None, metrics)

    def _commit(self, reason, old, new):
        self.last_update_time = datetime.utcnow().isoformat()
        self.last_update_reason = reason

        self._audit(reason, old, new)

    def _audit(self, reason, old_ref, new_ref, metrics=None):
        self.audit.log(
            {
                "reason": reason,
                "old_reference_id": id(old_ref) if old_ref is not None else None,
                "new_reference_id": id(new_ref) if new_ref is not None else None,
                "metrics": metrics,
                "strategy": self.strategy,
            }
        )
