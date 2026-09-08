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


def fuse_cluster(observations_xy, confidences, base_sigma=0.3, process_noise=1e-6):
    """
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
        # Predict (static model: state doesn't change, just add Q)
        x_pred = x
        P_pred = P + Q

        # Update
        z = observations_xy[k]
        R = _measurement_covariance(confidences[k], base_sigma)

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ (z - H @ x_pred)
        P = (np.eye(2) - K @ H) @ P_pred

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
