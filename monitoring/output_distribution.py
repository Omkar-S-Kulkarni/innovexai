import numpy as np
import pandas as pd
from typing import Dict, Tuple


class OutputDistributionTracker:
    """
    Tracks output-level behavior:
    - Predicted class frequency
    - Prediction confidence distribution

    NO labels required.
    """

    # =========================================================
    # 2.1 PREDICTED CLASS FREQUENCY
    # =========================================================
    @staticmethod
    def class_frequency(
        df: pd.DataFrame,
        class_col: str = "y_pred",
        normalize: bool = True
    ) -> pd.Series:
        """
        Returns class frequency (count or percentage).
        Works for binary & multi-class.
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        counts = df[class_col].value_counts().sort_index()

        if normalize:
            return counts / counts.sum()

        return counts

    @staticmethod
    def align_class_distributions(
        current: pd.Series,
        reference: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Ensures same class index for fair comparison.
        Missing classes get 0.
        """
        all_classes = sorted(set(current.index).union(set(reference.index)))

        current_aligned = current.reindex(all_classes, fill_value=0)
        reference_aligned = reference.reindex(all_classes, fill_value=0)

        return current_aligned, reference_aligned

    # =========================================================
    # 2.2 CONFIDENCE / PROBABILITY DISTRIBUTION
    # =========================================================
    @staticmethod
    def confidence_histogram(
        df: pd.DataFrame,
        conf_col: str = "confidence",
        bins: int = 20,
        range_: Tuple[float, float] = (0.0, 1.0),
        density: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes histogram values ONLY (no plotting).
        Same bins must be reused for reference & current.
        """
        if df is None or df.empty:
            return np.zeros(bins), np.linspace(range_[0], range_[1], bins + 1)

        confidences = df[conf_col].clip(*range_)

        hist, bin_edges = np.histogram(
            confidences,
            bins=bins,
            range=range_,
            density=density
        )

        return hist, bin_edges

    @staticmethod
    def validate_confidence_range(df: pd.DataFrame, conf_col="confidence") -> bool:
        """
        Ensures confidence is within [0,1].
        """
        if df is None or df.empty:
            return True

        return df[conf_col].between(0.0, 1.0).all()
