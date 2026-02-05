# drift_attribution.py

import numpy as np
import pandas as pd
from scipy.stats import entropy

# -----------------------------
# Feature Distribution Shift
# -----------------------------
def feature_kl_divergence(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list):
    """
    Compute KL divergence per feature between reference & current windows.
    Returns dict {feature: KL_divergence}.
    """
    kl_scores = {}
    for col in feature_cols:
        # Convert to numeric, coerce errors to NaN
        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        # Skip non-numeric columns that became empty after coercion
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            kl_scores[col] = 0.0
            continue

        # Bin values for discrete distribution
        bins = np.histogram_bin_edges(np.concatenate([ref_vals, curr_vals]), bins='auto')
        ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

        ref_hist += 1e-8  # avoid zeros
        curr_hist += 1e-8

        kl_scores[col] = float(entropy(ref_hist, curr_hist))

    return kl_scores


def feature_psi(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list, bins=10):
    """
    Compute PSI (Population Stability Index) per feature.
    Returns dict {feature: PSI}.
    """
    psi_scores = {}

    for col in feature_cols:
        # Convert to numeric, drop invalids
        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        # Skip non-numeric columns
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            psi_scores[col] = 0.0
            continue

        # Compute bins
        bin_edges = np.histogram_bin_edges(np.concatenate([ref_vals, curr_vals]), bins=bins)
        ref_hist, _ = np.histogram(ref_vals, bins=bin_edges)
        curr_hist, _ = np.histogram(curr_vals, bins=bin_edges)

        # Normalize
        ref_perc = ref_hist / ref_hist.sum()
        curr_perc = curr_hist / curr_hist.sum()

        # Avoid divide by zero or log(0)
        ref_perc += 1e-8
        curr_perc += 1e-8

        psi = np.sum((ref_perc - curr_perc) * np.log(ref_perc / curr_perc))
        psi_scores[col] = float(psi)

    return psi_scores


# -----------------------------
# Slice Comparison
# -----------------------------
def slice_feature_comparison(reference_df: pd.DataFrame, current_df: pd.DataFrame, slice_mask: pd.Series, feature_cols: list):
    """
    Compare feature distributions of a slice vs global.
    Returns dict {feature: KL_divergence}.
    """
    slice_df = current_df[slice_mask]
    if slice_df.empty:
        return {col: 0.0 for col in feature_cols}

    return feature_kl_divergence(reference_df, slice_df, feature_cols)
