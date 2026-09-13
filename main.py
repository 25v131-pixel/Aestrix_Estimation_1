import os
import pandas as pd

from src.coordinate_transform import unwrap_yaw_step, transform_single_detection
from src.clustering import associate_single_detection
from src.static_kalman_filter import init_track, predict_track, update_track
# from src.ghost_rejection import evaluate_track
from src.plot_map import plot_track_map, plot_track_map_sequence

PERCEPTION_FILE = "data/perception_log.csv"
TELEMETRY_FILE = "data/telemetry_log.csv"

OUTPUT_DIR = "outputs"
FINAL_TRACKS_CSV = os.path.join(OUTPUT_DIR, "final_tracks.csv")
FINAL_MAP_PNG = os.path.join(OUTPUT_DIR, "final_track_map.png")
DEMO_FRAMES_DIR = os.path.join(OUTPUT_DIR, "demo_frames")

# Config values -- move to a YAML/JSON config file in Stage 2;
# kept as constants here only until that lands.
MAX_TIME_GAP = 2.0
GATING_CONFIDENCE = 0.95
SNAPSHOT_INTERVAL_TICKS = 20   # for the live-demo frame sequence


def main():

    print("Aestrix Online Cone Mapping (Phase 2)")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    telemetry = pd.read_csv(TELEMETRY_FILE).sort_values("timestamp").reset_index(drop=True)
    perception = pd.read_csv(PERCEPTION_FILE).sort_values("timestamp").reset_index(drop=True)

    tracks = {}          # track_id -> track state dict
    next_track_id = 0
    previous_yaw = telemetry["yaw_rad"].iloc[0]   # seed unwrap with first raw yaw

    track_snapshots = []  # for the live-demo sequence

    # Group perception detections by timestamp so all detections
    # in the same tick are processed against the same vehicle pose.
    grouped_detections = perception.groupby("timestamp")

    for tick_index, (timestamp, telemetry_row) in enumerate(
        telemetry.set_index("timestamp").iterrows()
    ):

        # ---- Step 1: incremental yaw unwrap + pose for this tick ----
        previous_yaw = unwrap_yaw_step(previous_yaw, telemetry_row["yaw_rad"])
        vehicle_x = telemetry_row["x"]
        vehicle_y = telemetry_row["y"]

        # ---- Step 2: KF predict step for all active tracks ----
        # (static Kalman filter is time-invariant for stationary
        # cones, but this is where process-noise inflation for
        # unseen tracks would be applied, if used)
        for track in tracks.values():
            predict_track(track, timestamp, process_noise_rate=1e-6)

        # ---- Step 3: process detections arriving at this tick ----
        if timestamp in grouped_detections.groups:
            detections_this_tick = grouped_detections.get_group(timestamp)

            for _, detection_row in detections_this_tick.iterrows():

                global_x, global_y = transform_single_detection(
                    rel_x_sensor=detection_row["rel_x_sensor"],
                    rel_y_sensor=detection_row["rel_y_sensor"],
                    vehicle_x=vehicle_x,
                    vehicle_y=vehicle_y,
                    vehicle_yaw_unwrapped=previous_yaw,
                    camera_offset=(0.0, 0.0),
                )

                detection = {
                    "timestamp": timestamp,
                    "global_x": global_x,
                    "global_y": global_y,
                    "cone_type": detection_row["cone_type"],
                    "confidence": detection_row["confidence"],
                }

                matched_id = associate_single_detection(
                    detection,
                    list(tracks.values()),
                    max_time_gap=MAX_TIME_GAP,
                    same_type_only=True,
                    gating_confidence=GATING_CONFIDENCE,
                )

                if matched_id is None:
                    # Spawn new candidate track
                    matched_id = next_track_id
                    next_track_id += 1

                    new_track = init_track(
                        global_x, global_y, detection["confidence"]
                    )
                    new_track["id"] = matched_id
                    new_track["cone_type"] = detection["cone_type"]
                    new_track["last_seen"] = timestamp
                    new_track["last_predicted_time"] = timestamp
                    new_track["hit_count"] = 1
                    new_track["status"] = "candidate"

                    tracks[matched_id] = new_track
                else:
                    track = tracks[matched_id]
                    update_track(
                        track, global_x, global_y, detection["confidence"]
                    )
                    track["last_seen"] = timestamp
                    track["hit_count"] += 1

        # ---- Step 4: ghost rejection / lifecycle update ----
        for track in tracks.values():
            # track["status"] = evaluate_track(track, timestamp)
            pass  # placeholder pending ghost_rejection.py

        # ---- Step 5: periodic snapshot for the live demo ----
        if tick_index % SNAPSHOT_INTERVAL_TICKS == 0:
            snapshot_df = _tracks_to_dataframe(tracks)
            track_snapshots.append((timestamp, snapshot_df))

    # --------------------------------------------------
    # Final outputs
    # --------------------------------------------------

    final_tracks_df = _tracks_to_dataframe(tracks)
    final_tracks_df.to_csv(FINAL_TRACKS_CSV, index=False)
    print(f"Final tracks written to {FINAL_TRACKS_CSV}")

    plot_track_map(FINAL_TRACKS_CSV, TELEMETRY_FILE, FINAL_MAP_PNG)
    print(f"Final map written to {FINAL_MAP_PNG}")

    plot_track_map_sequence(track_snapshots, TELEMETRY_FILE, DEMO_FRAMES_DIR)
    print(f"Demo frames written to {DEMO_FRAMES_DIR}")

    print()
    print("Online pipeline run complete.")


def _tracks_to_dataframe(tracks):
    """
    Convert the in-memory tracks dict into the DataFrame schema
    expected by plot_track_map (id, cone_type, x, y, status,
    cov_xx, cov_xy, cov_yy).
    """
    rows = []
    for track in tracks.values():
        cov = track["covariance"]
        rows.append({
            "track_id": track["id"],
            "cone_type": track["cone_type"],
            "x": track["predicted_x"],
            "y": track["predicted_y"],
            "status": track["status"] if "status" in track else "candidate",
            "cov_xx": cov[0, 0] if cov is not None else 0.0,
            "cov_xy": cov[0, 1] if cov is not None else 0.0,
            "cov_yy": cov[1, 1] if cov is not None else 0.0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()