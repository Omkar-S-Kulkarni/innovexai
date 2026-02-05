import numpy as np
from scipy.stats import entropy, wasserstein_distance

def kl_divergence(p, q, eps=1e-6):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return entropy(p, q)


def psi(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)

    psi_val = np.sum(
        (expected_perc - actual_perc) *
        np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )
    return psi_val


def prediction_entropy(probs):
    """Mean entropy across all predictions"""
    return np.mean([entropy(p + 1e-12) for p in probs])


def confidence_metrics(probs):
    """Mean top-1 confidence & margin"""
    sorted_probs = np.sort(probs, axis=1)
    max_conf = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return max_conf.mean(), margin.mean()



from scipy.stats import entropy as scipy_entropy

def compute_slice_metrics(
    reference_probs,        # (N,)
    slice_confidences,      # (M,)
    slice_probs=None,       # (M, C)
    bins=20
):
    metrics = {}

    # ---- Distribution metrics (confidence-based) ----
    ref_hist, bin_edges = np.histogram(
        reference_probs, bins=bins, range=(0, 1), density=True
    )
    slice_hist, _ = np.histogram(
        slice_confidences, bins=bin_edges, density=True
    )

    ref_hist += 1e-8
    slice_hist += 1e-8

    ref_hist /= ref_hist.sum()
    slice_hist /= slice_hist.sum()

    metrics["kl_divergence"] = scipy_entropy(ref_hist, slice_hist)

    metrics["psi"] = np.sum(
        (ref_hist - slice_hist)
        * np.log(ref_hist / slice_hist)
    )

    metrics["wasserstein"] = wasserstein_distance(
        reference_probs, slice_confidences
    )

    metrics["mean_confidence"] = slice_confidences.mean()
    metrics["slice_size"] = len(slice_confidences)

    # ---- Entropy (REQUIRED FOR RANKING) ----
    if slice_probs is not None:
        sample_entropies = scipy_entropy(slice_probs.T)
        metrics["entropy"] = sample_entropies.mean()
    else:
        metrics["entropy"] = 0.0   # fallback (never crash)

    return metrics
