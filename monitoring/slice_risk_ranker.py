import numpy as np

DEFAULT_WEIGHTS = {
    "kl_divergence": 0.25,
    "psi": 0.2,
    "wasserstein": 0.2,
    "entropy": 0.2,
    "confidence_drop": 0.15
}


def normalize(val, eps=1e-6):
    """Normalize delta to [-1,1]"""
    return val / (abs(val) + eps)


def severity_label(score):
    if score > 0.6:
        return "HIGH"
    elif score > 0.3:
        return "MEDIUM"
    return "LOW"


def compute_risk_score(slice_metrics, global_metrics, weights=DEFAULT_WEIGHTS):
    score = 0.0
    components = {}
    for k, w in weights.items():
        if k == "confidence_drop":
            delta = global_metrics["mean_confidence"] - slice_metrics["mean_confidence"]
        else:
            delta = slice_metrics[k] - global_metrics[k]

        norm_delta = normalize(delta)
        components[k] = norm_delta
        score += w * norm_delta

    return score, components


def rank_slices(slice_metrics_dict, global_metrics):
    rankings = []
    for slice_name, metrics in slice_metrics_dict.items():
        score, components = compute_risk_score(metrics, global_metrics)
        rankings.append({
            "slice": slice_name,
            "risk_score": score,
            "components": components,
            "severity": severity_label(score)
        })

    rankings.sort(key=lambda x: x["risk_score"], reverse=True)
    return rankings


def explain_slice(slice_name, metrics, global_metrics):
    reasons = []
    if metrics["entropy"] > global_metrics["entropy"]:
        reasons.append("Entropy increased relative to global")
    if metrics["mean_confidence"] < global_metrics["mean_confidence"]:
        reasons.append("Confidence collapse detected")
    if metrics["kl_divergence"] > global_metrics["kl_divergence"]:
        reasons.append("Output distribution drifted")

    return {
        "slice": slice_name,
        "explanation": "; ".join(reasons) if reasons else "No dominant risk factors"
    }
