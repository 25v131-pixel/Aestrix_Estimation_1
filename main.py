"""
main.py
-------
Wires the three independent modules together:

    transform.py             (180 deg fix + global-frame projection)
    clustering.py             (nearest-neighbor + gating, persistence-based
                                ghost flag)
    static_kalman_filter.py   (per-cluster noise fusion + uncertainty)

Only the "Phase 01" steps requested are performed here - no downstream
validation against the 'label' column, no track-boundary pairing, etc.
'label' is carried through the intermediate file purely so the user can
do that scoring themselves later.
"""

import os
import pandas as pd

from transform import load_and_transform
from clustering import cluster_detections
from static_kalman_filter import fuse_all_clusters
from plot_map import plot_cone_map


def run(
    perception_csv,
    telemetry_csv,
    output_dir,
    camera_offset=(0.0, 0.0),
    gating_radius=1.5,
    max_time_gap=2.0,
    persistence_threshold=5,
    base_sigma=0.3,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Backward-sensor correction + global-frame transform
    detections = load_and_transform(perception_csv, telemetry_csv, camera_offset=camera_offset)

    # 2. Nearest-neighbor + gating clustering (association only)
    detections_clustered, cluster_summary = cluster_detections(
        detections,
        gating_radius=gating_radius,
        max_time_gap=max_time_gap,
        persistence_threshold=persistence_threshold,
    )

    # 3. Static Kalman filter fusion, per cluster, ghosts included so far
    fused_all = fuse_all_clusters(detections_clustered, base_sigma=base_sigma)

    # Attach the persistence-based ghost flag computed in clustering.py
    fused_all = fused_all.merge(
        cluster_summary[["cluster_id", "is_ghost_candidate", "first_seen", "last_seen"]],
        on="cluster_id",
        how="left",
    )

    #----- plot cone maps ------------------------------------------
    plot_cone_map(fused_all, output_path=os.path.join(output_dir, "cone_map.png"))

    real_map = fused_all[~fused_all["is_ghost_candidate"]].reset_index(drop=True)
    ghost_candidates = fused_all[fused_all["is_ghost_candidate"]].reset_index(drop=True)

    
    # ---- outputs -------------------------------------------------
    detections_path = os.path.join(output_dir, "global_cone_observations.csv")
    detections_clustered.to_csv(detections_path, index=False)

    cone_map_path = os.path.join(output_dir, "cone_map.csv")
    real_map.drop(columns=["is_ghost_candidate"]).to_csv(cone_map_path, index=False)

    uncertainty_path = os.path.join(output_dir, "cone_uncertainty.csv")
    fused_all[[
        "cluster_id", "cone_type", "num_observations",
        "global_x", "global_y", "std_x", "std_y", "cov_xy",
        "is_ghost_candidate",
    ]].to_csv(uncertainty_path, index=False)

    print(f"Total detections processed:        {len(detections)}")
    print(f"Clusters found:                     {len(fused_all)}")
    print(f"  -> kept as real (persistent):      {len(real_map)}")
    print(f"  -> flagged as ghost candidates:    {len(ghost_candidates)}")
    print()
    print(f"Wrote: {detections_path}")
    print(f"Wrote: {cone_map_path}")
    print(f"Wrote: {uncertainty_path}")

    return detections_clustered, fused_all


if __name__ == "__main__":
    run(
        perception_csv = "D:/claude_abaja/perception_log.csv",
        telemetry_csv = "D:/claude_abaja/telemetry_log.csv",
        output_dir = "D:/claude_abaja",
        camera_offset=(0.0, 0.0),
        gating_radius=1.5,
        max_time_gap=2.0,
        persistence_threshold=5,
        base_sigma=0.3,
    )
