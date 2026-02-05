# cluster_slice_adapter.py

from monitoring.slice_definition import Slice


def build_cluster_slices(df):
    slices = []

    for cid in df["_cluster_id"].unique():
        slices.append(
            Slice(
                name=f"auto_cluster_{cid}",
                mask_fn=lambda d, cid=cid: d["_cluster_id"] == cid,
                description="Automatically discovered cluster"
            )
        )

    return slices


