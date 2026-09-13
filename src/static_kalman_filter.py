"""
static_kalman_filter.py
------------------------
Fuses the (possibly many) noisy global-frame observations belonging to
one cluster (one physical cone) into a single position estimate with an
uncertainty (covariance), using a static-target Kalman filter.

State model: the cone does not move.
    state:      s_k = [x, y]^T,  s_k = s_{k-1}  (identity transition)
    process:    Q  (tiny, just keeps the filter from freezing/collapsing
                 numerically - it is NOT modelling real cone motion)
    measurement: z_k = s_k + noise,  noise ~ N(0, R_k)
    R_k is scaled by 1 / confidence_k, so a low-confidence detection
    pulls the estimate less than a high-confidence one.

This module only fuses positions it is handed - it has no idea whether
a cluster is "real" or "ghost" and does not read the label column at all.
That decision (persistence-based) is entirely clustering.py's job;
main.py decides which clusters to run through here.
"""

import numpy as np
import pandas as pd


def _measurement_covariance(confidence, base_sigma):
    """
    R for a single observation. Lower confidence -> larger noise.
    confidence is expected in (0, 1]; guarded against 0.
    """
    conf = max(float(confidence), 1e-3)
    sigma2 = (base_sigma ** 2) / conf
    return np.array([[sigma2, 0.0], [0.0, sigma2]])

def _kalman_predict_step(x, P, process_noise_rate, dt):
    """
    Static-target predict step: the mean doesn't move (identity
    transition), but the covariance grows to reflect elapsed
    uncertainty since the last predict.

    process_noise_rate : float
        Process noise variance PER SECOND. This differs from the
        original fuse_cluster's fixed per-observation `process_noise`
        -- online, predict() is called every tick regardless of
        whether a detection arrives that tick, so noise must scale
        with elapsed time (dt) rather than being a flat per-call
        constant, or an unseen track's uncertainty would never grow.
    dt : float
        Elapsed time (s) since this track's last predict step.
    """
    Q = np.eye(2) * process_noise_rate * dt
    x_pred = x
    P_pred = P + Q
    return x_pred, P_pred

def _kalman_update_step(x_pred, P_pred, z, confidence, base_sigma):
    """
    Standard KF measurement update. Identical math to the update
    half of fuse_cluster's loop body -- extracted so both the
    offline batch path and the online per-detection path call the
    exact same fusion logic.
    """
    H = np.eye(2)
    R = _measurement_covariance(confidence, base_sigma)

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_new = x_pred + K @ (z - H @ x_pred)
    P_new = (np.eye(2) - K @ H) @ P_pred

    return x_new, P_new

def fuse_cluster(observations_xy, confidences, base_sigma=0.3, process_noise=1e-6):
    """
    OFFLINE / BATCH USE ONLY. Takes a cluster's full observation
    history at once; retained for Phase 1 regression comparison.
    See init_track / predict_track / update_track below for the
    streaming equivalent.
    
    Run a static Kalman filter sequentially over one cluster's
    observations (already sorted by timestamp by the caller).

    Parameters
    ----------
    observations_xy : (N, 2) array-like
        Global x, y for each observation in the cluster, time-ordered.
    confidences : (N,) array-like
        Per-observation confidence score, same order.
    base_sigma : float
        Baseline measurement std-dev (m) at confidence == 1.0.
    process_noise : float
        Tiny per-step process variance added to both x and y each
        update, so the filter never becomes overconfident (P -> 0).

    Returns
    -------
    dict with:
        x, y                 : fused position estimate
        P                     : final 2x2 covariance matrix
        std_x, std_y          : sqrt of the diagonal of P
        cov_xy                : off-diagonal term of P
    """
    observations_xy = np.asarray(observations_xy, dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    if len(observations_xy) == 0:
        raise ValueError("fuse_cluster called with no observations.")

    Q = np.eye(2) * process_noise

    # Initialize with the first observation.
    x = observations_xy[0].copy()
    P = _measurement_covariance(confidences[0], base_sigma)

    H = np.eye(2)

    for k in range(1, len(observations_xy)):
        # Offline path treats each observation as one discrete step
        # (dt=1, process_noise as a flat per-step amount) -- this
        # preserves the original Phase 1/regression behavior exactly.
        x_pred, P_pred = _kalman_predict_step( 
            x, P, process_noise_rate=process_noise, dt=1.0)
        x, P = _kalman_update_step(
            x_pred, P_pred, observations_xy[k], confidences[k], base_sigma)

    return {
        "x": x[0],
        "y": x[1],
        "P": P,
        "std_x": np.sqrt(P[0, 0]),
        "std_y": np.sqrt(P[1, 1]),
        "cov_xy": P[0, 1],
    }

def fuse_all_clusters(detections_with_cluster_id, base_sigma=0.3, process_noise=1e-6):
    """
    OFFLINE / BATCH USE ONLY. Takes a cluster's full observation
    history at once; retained for Phase 1 regression comparison.
    See init_track / predict_track / update_track below for the
    streaming equivalent.
    
    Applies fuse_cluster to every cluster present in the detections
    DataFrame (as produced by clustering.cluster_detections).

    Parameters
    ----------
    detections_with_cluster_id : pandas.DataFrame
        Must contain: timestamp, global_x, global_y, confidence,
        cluster_id, cone_type.

    Returns
    -------
    pandas.DataFrame, one row per cluster_id, with the fused position
    and its uncertainty.
    """
    rows = []
    for cluster_id, group in detections_with_cluster_id.groupby("cluster_id"):
        group = group.sort_values("timestamp")
        result = fuse_cluster(
            group[["global_x", "global_y"]].to_numpy(),
            group["confidence"].to_numpy(),
            base_sigma=base_sigma,
            process_noise=process_noise,
        )
        rows.append({
            "cluster_id": cluster_id,
            "cone_type": group["cone_type"].iloc[0],
            "num_observations": len(group),
            "global_x": result["x"],
            "global_y": result["y"],
            "std_x": result["std_x"],
            "std_y": result["std_y"],
            "cov_xy": result["cov_xy"],
        })

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

# =============================================================
# ONLINE FUNCTIONS
#
# These operate on a single track's state dict, carried and
# mutated by main.py's loop across ticks. Field names match what
# clustering.associate_single_detection() and main.py expect:
#   predicted_x, predicted_y, covariance, last_predicted_time
# =============================================================

def init_track(x, y, confidence, base_sigma=0.3):
    """
    Initialize a new track's KF state from its first detection.

    Returns a dict with the fields main.py stores per track:
        predicted_x, predicted_y : the fused position estimate
        covariance                : 2x2 P matrix
        last_predicted_time       : timestamp of last predict()
                                     call, used to compute dt
    """
    P = _measurement_covariance(confidence, base_sigma)

    return {
        "predicted_x": x,
        "predicted_y": y,
        "covariance": P,
    }


def predict_track(track, timestamp, process_noise_rate=1e-6):
    """
    Advance one track's covariance to the current timestamp.
    Called once per tick for every ACTIVE track, before
    association, so clustering.py has an up-to-date covariance to
    gate against -- including for tracks with no new detection
    this tick.

    Mutates and returns `track`. Requires track to already have
    'last_predicted_time' set (init_track sets it to the track's
    creation timestamp; the caller must set it there since
    init_track doesn't take a timestamp argument itself).
    """
    last_time = track.get("last_predicted_time", timestamp)
    dt = max(timestamp - last_time, 0.0)

    x = np.array([track["predicted_x"], track["predicted_y"]])
    P = track["covariance"]

    x_pred, P_pred = _kalman_predict_step(
        x, P, process_noise_rate=process_noise_rate, dt=dt
    )

    track["predicted_x"] = x_pred[0]
    track["predicted_y"] = x_pred[1]
    track["covariance"] = P_pred
    track["last_predicted_time"] = timestamp

    return track


def update_track(track, global_x, global_y, confidence, base_sigma=0.3):
    """
    Incorporate one new matched detection into a track's KF state.
    Call this AFTER predict_track has already been called for this
    tick and AFTER clustering.associate_single_detection has
    matched this detection to this track.

    Mutates and returns `track`.
    """
    x_pred = np.array([track["predicted_x"], track["predicted_y"]])
    P_pred = track["covariance"]

    z = np.array([global_x, global_y])

    x_new, P_new = _kalman_update_step(
        x_pred, P_pred, z, confidence, base_sigma
    )

    track["predicted_x"] = x_new[0]
    track["predicted_y"] = x_new[1]
    track["covariance"] = P_new

    return track