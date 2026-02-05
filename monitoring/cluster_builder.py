# cluster_builder.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusterBuilder:
    def __init__(
        self,
        method="kmeans",
        n_clusters=4,
        min_cluster_size=30,
        random_state=42
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit_predict(self, df, feature_cols):
        X = self.scaler.fit_transform(df[feature_cols])

        if self.method == "kmeans":
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
            cluster_ids = model.fit_predict(X)

        elif self.method == "hdbscan" and HDBSCAN_AVAILABLE:
            model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size
            )
            cluster_ids = model.fit_predict(X)

        else:
            raise ValueError("Unsupported clustering method")

        df = df.copy()
        df["_cluster_id"] = cluster_ids
        return df