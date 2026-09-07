
import pandas as pd
from sklearn.cluster import DBSCAN


def dbscan_association(
    observations,
    eps=1.5,
    min_samples=5
):
    """
    Cluster transformed cone observations using DBSCAN.

    Left and right cone detections are clustered separately
    so that observations of different cone types are not merged.
    """

    result = observations.copy()
    result["cluster_id"] = -1

    cone_tables = []
    next_cluster_id = 0

    for cone_type in ["left", "right"]:

        mask = result["cone_type"] == cone_type
        subset = result.loc[mask]

        if len(subset) == 0:
            continue

        points = subset[
            ["global_x", "global_y"]
        ].to_numpy()

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(points)

        labels = clustering.labels_

        for local_label in sorted(set(labels)):

            # DBSCAN label -1 means noise
            if local_label == -1:
                continue

            local_mask = labels == local_label
            indices = subset.index[local_mask]

            cluster_points = subset.loc[indices]

            cone_tables.append({
                "cluster_id": next_cluster_id,
                "cone_type": cone_type,
                "x": cluster_points["global_x"].mean(),
                "y": cluster_points["global_y"].mean(),
                "count": len(cluster_points)
            })

            result.loc[
                indices,
                "cluster_id"
            ] = next_cluster_id

            next_cluster_id += 1

    cones = pd.DataFrame(cone_tables)

    return result, cones
