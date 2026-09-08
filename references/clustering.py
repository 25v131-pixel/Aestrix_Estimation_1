"""
clustering.py
-------------
Groups repeated global-frame detections of the same physical cone into
a single track, using sequential nearest-neighbor data association with
a distance gate.

This module does ONLY association / grouping. It does not fuse
positions (that's static_kalman_filter.py's job) and it does not
look at the 'label' column at all - ghost rejection here is done
purely from persistence (how many frames a track was observed in),
which is a property of the geometry, not of the ground-truth label.

Output: the same detections DataFrame with an added 'cluster_id'
column, plus a cluster summary table.
"""

import numpy as np
import pandas as pd


def cluster_detections(
    detections,
    gating_radius=1.5,
    max_time_gap=2.0,
    persistence_threshold=5,
    same_type_only=True,
):
    """
    Parameters
    ----------
    detections : pandas.DataFrame
        Output of transform.load_and_transform (must have timestamp,
        global_x, global_y, cone_type).
    gating_radius : float
        Max distance (m) from a track's running centroid for a new
        detection to be considered the same physical cone.
    max_time_gap : float
        If a track hasn't been matched for longer than this (seconds),
        it is "closed" - later detections cannot re-attach to it, even
        if they land in the same place. Prevents an old track silently
        absorbing an unrelated later ghost/cone.
    persistence_threshold : int
        Minimum number of observations a track needs to be treated as
        a real, persistent cone rather than a transient ghost blip.
    same_type_only : bool
        If True, a detection can only join a track whose recorded
        cone_type matches (left/right/unknown treated as separate).

    Returns
    -------
    (detections_with_cluster_id, cluster_summary) : tuple of DataFrames
        detections_with_cluster_id : original rows + 'cluster_id'
        cluster_summary : one row per cluster with count, cone_type,
                          a rough centroid (mean of raw detections -
                          NOT the Kalman fused estimate), first/last
                          seen timestamps, and 'is_ghost_candidate'
                          flag from persistence alone.
    """

    df = detections.sort_values("timestamp").reset_index(drop=False)  # keep orig index
    xy = df[["global_x", "global_y"]].to_numpy(dtype=float)
    times = df["timestamp"].to_numpy(dtype=float)
    types = df["cone_type"].astype(str).to_numpy()

    tracks = []  # each: dict with sum_x, sum_y, count, last_seen, cone_type, id
    cluster_id_for_row = np.empty(len(df), dtype=int)
    next_id = 0

    for i in range(len(df)):
        px, py = xy[i]
        t = times[i]
        ctype = types[i]

        best_track = None
        best_dist = np.inf

        for track in tracks:
            if (t - track["last_seen"]) > max_time_gap:
                continue  # track is closed, cannot rejoin
            if same_type_only and track["cone_type"] != ctype:
                continue
            cx = track["sum_x"] / track["count"]
            cy = track["sum_y"] / track["count"]
            dist = np.hypot(px - cx, py - cy)
            if dist <= gating_radius and dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is None:
            best_track = {
                "id": next_id,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "count": 0,
                "last_seen": t,
                "cone_type": ctype,
                "first_seen": t,
            }
            tracks.append(best_track)
            next_id += 1

        best_track["sum_x"] += px
        best_track["sum_y"] += py
        best_track["count"] += 1
        best_track["last_seen"] = t
        cluster_id_for_row[i] = best_track["id"]

    df["cluster_id"] = cluster_id_for_row
    # restore original row order / index
    df = df.sort_values("index").drop(columns="index").reset_index(drop=True)

    summary_rows = []
    for track in tracks:
        summary_rows.append({
            "cluster_id": track["id"],
            "cone_type": track["cone_type"],
            "num_observations": track["count"],
            "raw_mean_x": track["sum_x"] / track["count"],
            "raw_mean_y": track["sum_y"] / track["count"],
            "first_seen": track["first_seen"],
            "last_seen": track["last_seen"],
            "is_ghost_candidate": track["count"] < persistence_threshold,
        })

    cluster_summary = pd.DataFrame(summary_rows).sort_values("cluster_id").reset_index(drop=True)

    return df, cluster_summary
