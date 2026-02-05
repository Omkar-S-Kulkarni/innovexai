import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class Slice:
    def __init__(self, name, mask_fn, description=""):
        self.name = name
        self.mask_fn = mask_fn
        self.description = description

    def apply(self, df: pd.DataFrame):
        mask = self.mask_fn(df)
        return df[mask], mask.sum()


class SliceRegistry:
    def __init__(self, min_slice_size=30):
        self.slices = {}
        self.min_slice_size = min_slice_size

    def register(self, slice_obj: Slice):
        self.slices[slice_obj.name] = slice_obj

    def get_valid_slices(self, df: pd.DataFrame):
        valid = {}
        skipped = {}
        for name, slice_obj in self.slices.items():
            sliced_df, size = slice_obj.apply(df)
            if size >= self.min_slice_size:
                valid[name] = {
                    "data": sliced_df,
                    "size": size,
                    "description": slice_obj.description,
                    "mask_fn": slice_obj.mask_fn  # <--- add this
                }
            else:
                skipped[name] = {
                    "size": size,
                    "reason": "Slice too small"
                }
        return valid, skipped


# -------- Manual Feature Slices -------- #

def threshold_slice(feature, threshold, op=">"):
    ops = {
        ">": lambda x: x > threshold,
        "<": lambda x: x < threshold,
        ">=": lambda x: x >= threshold,
        "<=": lambda x: x <= threshold,
    }

    return Slice(
        name=f"{feature}{op}{threshold}",
        mask_fn=lambda df: ops[op](df[feature]),
        description=f"{feature} {op} {threshold}"
    )


def range_slice(feature, low, high):
    return Slice(
        name=f"{feature}_range_{low}_{high}",
        mask_fn=lambda df: (df[feature] >= low) & (df[feature] < high),
        description=f"{low} ≤ {feature} < {high}"
    )


# -------- Cluster-based Slices -------- #

def cluster_slices(df: pd.DataFrame, feature_cols, n_clusters=4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = kmeans.fit_predict(df[feature_cols])

    df = df.copy()
    df["_cluster_id"] = cluster_ids

    slices = []
    for cid in range(n_clusters):
        slices.append(
            Slice(
                name=f"cluster_{cid}",
                mask_fn=lambda d, cid=cid: d["_cluster_id"] == cid,
                description=f"Unsupervised cluster {cid}"
            )
        )
    return df, slices
