"""
clustering.py
-------------
Two association strategies live in this file:

- cluster_detections(): OFFLINE/BATCH. Sequential NN with a 
fixed-radius Euclidean gate and a self-maintained raw-mean centroid.
Kept for Phase 1 regression comparison.

- associate_single_detection(): ONLINE/STREAMING. Chi-squared gating
on Mahalanobis distance, using each track's KF-PREDICTED position and
covariance as the single source of truth for track location (no separate
centroid is maintained here). Ghost status is NOT decided in this file 
--that is ghost_rejection.py's job; this module only reports 
hit_count/last_seen for it to consume.

"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def cluster_detections(
    detections,
    gating_radius=1.5,
    max_time_gap=2.0,
    persistence_threshold=5,
    same_type_only=True,
):
    """
    OFFLINE / BATCH USE ONLY. Retained for Phase 1 regression
    comparison against the online pipeline. See
    associate_single_detection() for the streaming equivalent.

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

# =============================================================
# ONLINE / STREAMING ASSOCIATION
# =============================================================

def mahalanobis_distance_squared(point_xy, predicted_xy, covariance):
    """
    Squared Mahalanobis distance between an incoming detection and
    a track's KF-predicted position, using the track's covariance.

    This is what lets the gate tighten for confident tracks and
    loosen for uncertain ones, instead of using one fixed radius
    for every track regardless of how well-established it is.
    """

    diff = np.array([
        point_xy[0] - predicted_xy[0],
        point_xy[1] - predicted_xy[1]
    ])

    try:
        inv_covariance = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        # Near-singular covariance (e.g. a brand-new track with
        # near-zero uncertainty in one axis) -- fall back to the
        # pseudo-inverse rather than crashing the pipeline.
        inv_covariance = np.linalg.pinv(covariance)

    return float(diff @ inv_covariance @ diff)


def chi_squared_gate_threshold(gating_confidence=0.95, dof=2):
    """
    Chi-squared critical value for the given confidence level and
    degrees of freedom (2 for a 2D position gate).
    """
    return chi2.ppf(gating_confidence, df=dof)


def associate_single_detection(
    detection,
    tracks,
    max_time_gap=2.0,
    same_type_only=True,
    gating_confidence=0.95,
):
    """
    Associate ONE incoming detection to the best matching active
    track using chi-squared gating on Mahalanobis distance. Online
    counterpart to cluster_detections().

    Parameters
    ----------
    detection : dict
        Must contain: 'timestamp', 'global_x', 'global_y', 'cone_type'.

    tracks : list of dict
        Each active track must expose the KF's PREDICTED state for
        this tick (i.e. after the KF's predict step has already run):
            'id', 'predicted_x', 'predicted_y',
            'covariance' (2x2 array), 'last_seen',
            'cone_type', 'hit_count'
        This function does not compute or store a track position
        itself -- the Kalman filter module owns that.

    max_time_gap : float
        Tracks unseen longer than this (seconds) cannot be matched.
        Must stay below the ~60s lap period to avoid cross-lap
        merging.

    same_type_only : bool
        Restrict matching to same-cone_type tracks. NOTE: relies on
        cone_type being reliable -- flagged elsewhere that 'left'
        detections have an inconsistent rel_y_sensor sign, so treat
        this as a temporary simplification pending that fix.

    gating_confidence : float
        Confidence level for the chi-squared gate (e.g. 0.95).

    Returns
    -------
    int or None
        Matched track id, or None if the detection should spawn a
        new candidate track.
    """

    threshold = chi_squared_gate_threshold(
        gating_confidence=gating_confidence,
        dof=2
    )

    point_xy = (detection["global_x"], detection["global_y"])
    t = detection["timestamp"]
    ctype = detection["cone_type"]

    best_id = None
    best_dist_sq = np.inf

    for track in tracks:

        if (t - track["last_seen"]) > max_time_gap:
            continue

        if same_type_only and track["cone_type"] != ctype:
            continue

        dist_sq = mahalanobis_distance_squared(
            point_xy,
            (track["predicted_x"], track["predicted_y"]),
            track["covariance"]
        )

        if dist_sq <= threshold and dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_id = track["id"]

    return best_id
