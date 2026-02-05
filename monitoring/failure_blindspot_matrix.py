# monitoring/failure_blindspot_matrix.py

def build_failure_blindspot_matrix():
    return {
        "detectable": {
            "gradual_drift": "Composite slope",
            "sudden_drift": "Delta spikes",
            "localized_drift": "Slice divergence",
            "oscillatory_drift": "Rolling variance / FFT",
            "confidence_collapse": "Entropy + margin",
            "stability_failure": "Perturbation flip-rate",
        },
        "weakly_detectable": {
            "rare_event_drift": "Low sample mass",
        },
        "undetectable": {
            "label_shift": "No labels used",
            "semantic_drift": "Outputs unchanged",
            "perfectly_calibrated_concept_drift": "Confidence unchanged",
        },
        "explicit_non_actions": [
            "No retraining",
            "No labels",
            "No stopping inference",
            "No blocking predictions",
        ],
    }
