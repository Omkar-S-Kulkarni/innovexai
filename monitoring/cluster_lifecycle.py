# cluster_lifecycle.py

from collections import defaultdict


class ClusterLifecycleTracker:
    def __init__(self, min_cluster_size=30):
        self.min_cluster_size = min_cluster_size
        self.history = defaultdict(list)

    def update(self, df):
        cluster_stats = {}

        for cid, group in df.groupby("_cluster_id"):
            size = len(group)
            cluster_stats[cid] = {
                "size": size,
                "active": size >= self.min_cluster_size
            }
            self.history[cid].append(size)

        return cluster_stats

    def cluster_stability(self, cluster_id):
        sizes = self.history.get(cluster_id, [])
        if len(sizes) < 2:
            return 0.0
        return 1.0 - (max(sizes) - min(sizes)) / max(sizes)