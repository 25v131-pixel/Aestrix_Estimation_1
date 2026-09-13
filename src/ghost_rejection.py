"""
ghost_rejection.py
-------------------
Three independent, parallel filters for separating real, persistent
cones from spurious/ghost detections, plus a track lifecycle state
machine (candidate -> confirmed -> ghost) with hysteresis on both
promotion and demotion.

Design notes:
- Confidence, persistence, and spatial consistency are INDEPENDENT
  checks targeting different failure modes. They are never merged
  into one score -- each filter's pass/fail is exposed separately so
  an ablation study can disable them one at a time.
- 'label' (ground truth) is NEVER read here. Thresholds must come
  from the ablation sweep, not from peeking at the answer key.
- Confidence happens to separate ghosts perfectly on the current
  dataset (max ghost 0.55, min real 0.60) -- NOT assumed to
  generalize, which is exactly why persistence and spatial
  consistency carry real weight rather than being decorative.
- Checked cheapest-first: confidence (O(1) running mean) ->
  persistence (O(1) counter) -> spatial consistency (O(1) running
  mean of stored Mahalanobis distances). A track that fails an
  earlier, cheaper check skips the rest.
- Track closure after max_time_gap is handled upstream in
  clustering.py; this module assumes any track handed to it is
  still "open".
"""

import numpy as np

from src.clustering import mahalanobis_distance_squared


# =============================================================
# Per-detection bookkeeping
# =============================================================

def record_detection(track, global_x, global_y, confidence, history_length=20):
    """
    Call once per detection matched to `track` -- for BOTH a newly
    spawned track and an existing matched track -- AFTER
    clustering.associate_single_detection has matched it and BEFORE
    static_kalman_filter.update_track mutates the track's predicted
    state. Spatial consistency needs the PRE-update predicted
    position/covariance to measure how far the raw detection landed
    from where the track expected it.

    Mutates track in place: appends to confidence_history and
    spatial_residual_history (both capped at history_length so
    memory stays bounded on a long-running track), and increments
    hit_count. This is now the single place hit_count is
    incremented -- main.py should not increment it separately.
    """
    dist_sq = mahalanobis_distance_squared(
        (global_x, global_y),
        (track["predicted_x"], track["predicted_y"]),
        track["covariance"],
    )

    confidence_history = track.setdefault("confidence_history", [])
    confidence_history.append(float(confidence))
    if len(confidence_history) > history_length:
        confidence_history.pop(0)

    residual_history = track.setdefault("spatial_residual_history", [])
    residual_history.append(dist_sq)
    if len(residual_history) > history_length:
        residual_history.pop(0)

    track["hit_count"] = track.get("hit_count", 0) + 1

    return track


# =============================================================
# Independent filters -- each returns True if the track PASSES
# (looks like a real cone), False if it fails that check.
# =============================================================

def confidence_filter(track, min_avg_confidence=0.6):
    """
    Cheapest check: running mean confidence of detections matched
    to this track must clear the threshold.
    """
    history = track.get("confidence_history", [])
    if len(history) == 0:
        return False
    return float(np.mean(history)) >= min_avg_confidence


def persistence_filter(track, persistence_threshold=5):
    """
    Track must have enough accumulated hits to be trusted as a
    real, repeatedly-observed cone rather than a transient blip.
    """
    return track.get("hit_count", 0) >= persistence_threshold


def spatial_consistency_filter(track, max_avg_mahalanobis_sq=9.0):
    """
    Checks whether detections matched to this track land close to
    where the track predicted them, on average, rather than
    repeatedly landing near the edge of the gate -- a sign the
    "track" is stitching together unrelated noise rather than
    observing one stationary cone.
    """
    history = track.get("spatial_residual_history", [])
    if len(history) == 0:
        return False
    return float(np.mean(history)) <= max_avg_mahalanobis_sq


# =============================================================
# Track lifecycle
# =============================================================

def evaluate_track(
    track,
    min_avg_confidence=0.6,
    persistence_threshold=5,
    max_avg_mahalanobis_sq=9.0,
    promotion_hysteresis_ticks=3,
    demotion_hysteresis_ticks=3,
    enable_confidence_filter=True,
    enable_persistence_filter=True,
    enable_spatial_consistency_filter=True,
):
    """
    Update one track's lifecycle status in place. Call once per
    tick for every active track (after record_detection, if a
    detection was matched this tick).

    Each filter is independently toggleable via enable_* flags --
    this is what lets the ablation study disable one filter at a
    time and measure its isolated contribution.

    Returns the updated status: "candidate", "confirmed", or "ghost".
    """
    track.setdefault("status", "candidate")
    track.setdefault("confirm_streak", 0)
    track.setdefault("demote_streak", 0)

    if enable_confidence_filter:
        if not confidence_filter(track, min_avg_confidence):
            _apply_demotion(track, demotion_hysteresis_ticks)
            return track["status"]

    if enable_persistence_filter:
        if not persistence_filter(track, persistence_threshold):
            # Not yet persistent enough -- only a "going bad" signal
            # for an already-confirmed track, not for a new candidate.
            if track["status"] == "confirmed":
                _apply_demotion(track, demotion_hysteresis_ticks)
            else:
                track["confirm_streak"] = 0
            return track["status"]

    if enable_spatial_consistency_filter:
        if not spatial_consistency_filter(track, max_avg_mahalanobis_sq):
            _apply_demotion(track, demotion_hysteresis_ticks)
            return track["status"]

    # All enabled checks passed this tick.
    track["demote_streak"] = 0
    track["confirm_streak"] += 1

    if track["status"] != "confirmed" and track["confirm_streak"] >= promotion_hysteresis_ticks:
        track["status"] = "confirmed"

    return track["status"]


def _apply_demotion(track, demotion_hysteresis_ticks):
    """
    A track only drops to 'ghost' after failing checks for
    demotion_hysteresis_ticks CONSECUTIVE evaluations, not on a
    single bad tick.
    """
    track["confirm_streak"] = 0
    track["demote_streak"] += 1

    if track["demote_streak"] >= demotion_hysteresis_ticks:
        track["status"] = "ghost"