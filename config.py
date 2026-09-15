"""
config.py
---------
Central configuration for the online cone-mapping pipeline.

All tunable parameters live here so they can be changed without
touching pipeline code (src/*) or orchestration entry points
(main.py, ablation_study.py). Every ablation condition in
ablation_study.py starts from DEFAULT_CONFIG and overrides only the
keys it needs to test -- see ablation_study.ABLATION_CONDITIONS.
"""

import os

# ---- File paths ----
PERCEPTION_FILE = "data/perception_log.csv"
TELEMETRY_FILE = "data/telemetry_log.csv"

OUTPUT_DIR = "outputs"
FINAL_TRACKS_CSV = os.path.join(OUTPUT_DIR, "final_tracks.csv")
FINAL_MAP_PNG = os.path.join(OUTPUT_DIR, "final_track_map.png")
DEMO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "demo_frames")

# ---- Live-demo snapshot cadence ----
SNAPSHOT_INTERVAL_TICKS = 10   # ticks between recorded map snapshots (deliverable #6)

# ---- Default pipeline parameters ----
# ablation_study.py builds each condition as {**DEFAULT_CONFIG, **overrides}
DEFAULT_CONFIG = {
    # Sensor / transform
    "camera_offset": (0.0, 0.0),        # (x, y) sensor mount offset in vehicle frame, meters
    "apply_sensor_correction": True,    # 180-degree backward-sensor correction; False only for ablation

    # Association (src/clustering.py)
    "max_time_gap": 2.0,                # seconds; stay well under the ~60s lap period to avoid cross-lap merging
    "gating_confidence": 0.95,          # chi-squared gate confidence level for Mahalanobis association

    # Kalman filter (src/static_kalman_filter.py)
    "process_noise_rate": 1e-6,         # per-second process noise; keeps covariance growing while a track is unseen
    "base_sigma": 0.3,                  # baseline measurement std-dev (m) at confidence == 1.0

    # Ghost rejection (src/ghost_rejection.py)
    "min_avg_confidence": 0.6,          # confidence_filter threshold
    "persistence_threshold": 5,         # min hits before a candidate can be promoted
    "max_avg_mahalanobis_sq": 9.0,      # spatial_consistency_filter threshold
    "promotion_hysteresis_ticks": 3,    # consecutive good ticks required: candidate -> confirmed
    "demotion_hysteresis_ticks": 3,     # consecutive bad ticks required: confirmed -> ghost

    "enable_confidence_filter": True,
    "enable_persistence_filter": True,
    "enable_spatial_consistency_filter": True,
}