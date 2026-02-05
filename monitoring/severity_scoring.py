# severity_scoring.py
"""
Compute severity levels for slice-level drift signals.
Thresholds can be adjusted as per system sensitivity.
"""

SEVERITY_THRESHOLDS = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.8
}


def compute_severity(risk_score: float) -> str:
    """Map numeric risk score to severity label."""
    if risk_score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif risk_score >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    elif risk_score >= SEVERITY_THRESHOLDS["LOW"]:
        return "LOW"
    return "NONE"


def severity_rank(severity: str) -> int:
    """Rank severity to allow escalation logic."""
    ranks = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    return ranks.get(severity, 0)


def should_escalate(prev_severity: str, new_severity: str) -> bool:
    """Determine if severity should escalate."""
    return severity_rank(new_severity) > severity_rank(prev_severity)
