# NEW:
import os
import pandas as pd
import numpy as np
import time
import tracemalloc

from src.coordinate_transform import unwrap_yaw_step, transform_single_detection
from src.clustering import associate_single_detection
from src.static_kalman_filter import init_track, predict_track, update_track
from src.ghost_rejection import record_detection, evaluate_track
from src.plot_map import plot_track_map, plot_track_map_sequence
from src.data_io import load_telemetry, load_perception, ensure_output_dir, save_tracks_csv
from config import (
    PERCEPTION_FILE, TELEMETRY_FILE, OUTPUT_DIR,
    FINAL_TRACKS_CSV, FINAL_MAP_PNG, DEMO_FRAMES_DIR,
    SNAPSHOT_INTERVAL_TICKS, DEFAULT_CONFIG,
)


def run_pipeline_online(
    perception_csv=PERCEPTION_FILE,
    telemetry_csv=TELEMETRY_FILE,
    config=None,
    collect_snapshots=False,
    snapshot_interval_ticks=SNAPSHOT_INTERVAL_TICKS,
):
    """
    Core streaming loop, extracted so both main() and
    ablation_study.py can drive it. `config` lets a caller (like the
    ablation study) override any parameter without touching this
    function's body -- merged on top of DEFAULT_CONFIG (config.py),
    so calling run_pipeline_online() with config=None reproduces
    main()'s exact default behavior, and a partial override dict
    (e.g. {"apply_sensor_correction": False}) only changes that key.
    """
    merged_config = {**DEFAULT_CONFIG, **(config or {})}

    camera_offset = merged_config["camera_offset"]
    apply_sensor_correction = merged_config["apply_sensor_correction"]
    max_time_gap = merged_config["max_time_gap"]
    gating_confidence = merged_config["gating_confidence"]
    process_noise_rate = merged_config["process_noise_rate"]
    base_sigma = merged_config["base_sigma"]

    ghost_kwargs = dict(
        min_avg_confidence=merged_config["min_avg_confidence"],
        persistence_threshold=merged_config["persistence_threshold"],
        max_avg_mahalanobis_sq=merged_config["max_avg_mahalanobis_sq"],
        promotion_hysteresis_ticks=merged_config["promotion_hysteresis_ticks"],
        demotion_hysteresis_ticks=merged_config["demotion_hysteresis_ticks"],
        enable_confidence_filter=merged_config["enable_confidence_filter"],
        enable_persistence_filter=merged_config["enable_persistence_filter"],
        enable_spatial_consistency_filter=merged_config["enable_spatial_consistency_filter"],
    )

    telemetry = load_telemetry(telemetry_csv)
    perception, grouped_detections = load_perception(perception_csv)

    tracks = {}
    next_track_id = 0
    previous_yaw = telemetry["yaw_rad"].iloc[0]
    archived_tracks = []
    track_snapshots = []
    detection_records = []
    tick_latencies_ms = []

    tracemalloc.start()

    for tick_index, telemetry_row in telemetry.iterrows():
        tick_start = time.perf_counter()
        timestamp = telemetry_row["timestamp"]

        previous_yaw = unwrap_yaw_step(previous_yaw, telemetry_row["yaw_rad"])
        vehicle_x, vehicle_y = telemetry_row["x"], telemetry_row["y"]

        for track in tracks.values():
            predict_track(track, timestamp, process_noise_rate=process_noise_rate)

        if timestamp in grouped_detections.groups:
            for _, detection_row in grouped_detections.get_group(timestamp).iterrows():

                global_x, global_y = transform_single_detection(
                    rel_x_sensor=detection_row["rel_x_sensor"],
                    rel_y_sensor=detection_row["rel_y_sensor"],
                    vehicle_x=vehicle_x, vehicle_y=vehicle_y,
                    vehicle_yaw_unwrapped=previous_yaw,
                    camera_offset=camera_offset,
                    apply_sensor_correction=apply_sensor_correction,
                )

                detection = {
                    "timestamp": timestamp, "global_x": global_x, "global_y": global_y,
                    "cone_type": detection_row["cone_type"],
                }
                confidence = detection_row["confidence"]

                matched_id = associate_single_detection(
                    detection, list(tracks.values()),
                    max_time_gap=max_time_gap, same_type_only=True,
                    gating_confidence=gating_confidence,
                )

                if matched_id is None:
                    matched_id = next_track_id
                    next_track_id += 1
                    new_track = init_track(global_x, global_y, confidence, base_sigma=base_sigma)
                    new_track["id"] = matched_id
                    new_track["cone_type"] = detection["cone_type"]
                    new_track["last_seen"] = timestamp
                    new_track["last_predicted_time"] = timestamp
                    new_track["status"] = "candidate"
                    record_detection(new_track, global_x, global_y, confidence)
                    tracks[matched_id] = new_track
                else:
                    track = tracks[matched_id]
                    record_detection(track, global_x, global_y, confidence)
                    update_track(track, global_x, global_y, confidence, base_sigma=base_sigma)
                    track["last_seen"] = timestamp

                detection_records.append({"track_id": matched_id, "label": detection_row.get("label", None)})

        for track in tracks.values():
            evaluate_track(track, **ghost_kwargs)

        # Prune stale non-confirmed tracks (see prior discussion:
        # keeps the live scan bounded; confirmed tracks are never
        # pruned; pruned tracks are archived, not dropped).
        stale_ids = [
            tid for tid, track in tracks.items()
            if track.get("status") != "confirmed"
            and (timestamp - track["last_seen"]) > max_time_gap
        ]
        for tid in stale_ids:
            pruned_track = tracks.pop(tid)
            cov = pruned_track["covariance"]
            archived_tracks.append({
                "track_id": pruned_track["id"], "cone_type": pruned_track["cone_type"],
                "x": pruned_track["predicted_x"], "y": pruned_track["predicted_y"],
                "status": pruned_track.get("status", "candidate"),
                "hit_count": pruned_track.get("hit_count", 0),
                "cov_xx": cov[0, 0], "cov_xy": cov[0, 1], "cov_yy": cov[1, 1],
            })

        if collect_snapshots and tick_index % snapshot_interval_ticks == 0:
            track_snapshots.append((timestamp, tracks_to_dataframe(tracks)))

        tick_latencies_ms.append((time.perf_counter() - tick_start) * 1000.0)

    peak_memory_kb = tracemalloc.get_traced_memory()[1] / 1024.0
    tracemalloc.stop()

    timing = {
        "mean_tick_ms": float(np.mean(tick_latencies_ms)),
        "median_tick_ms": float(np.median(tick_latencies_ms)),
        "max_tick_ms": float(np.max(tick_latencies_ms)),
        "peak_memory_kb": peak_memory_kb,
        "pruned_candidate_count": sum(1 for t in archived_tracks if t["status"] != "ghost"),
        "pruned_ghost_count": sum(1 for t in archived_tracks if t["status"] == "ghost"),
        "live_track_count_at_end": len(tracks),
    }

    return {
        "tracks": tracks,
        "archived_tracks": archived_tracks,
        "detection_records": detection_records,
        "track_snapshots": track_snapshots,
        "timing": timing,
    }

def full_tracks_dataframe(result):
    """
    Combine a run_pipeline_online() result's still-live tracks with
    its archived (pruned) tracks into one DataFrame -- the complete
    picture, since pruning only removes a track from the live scan
    for performance, not from the run's history. Use this (not
    tracks_to_dataframe alone) for anything final: the saved CSV,
    the map plot, or metrics/validation.
    """
    live_df = tracks_to_dataframe(result["tracks"])
    archived_df = pd.DataFrame(result["archived_tracks"])
    if len(archived_df) == 0:
        return live_df
    return pd.concat([live_df, archived_df], ignore_index=True)

def main():
    print("Aestrix Online Cone Mapping (Phase 2)")
    print()

    ensure_output_dir(OUTPUT_DIR)

    result = run_pipeline_online(
        PERCEPTION_FILE, TELEMETRY_FILE,
        config=None,                 # None -> DEFAULT_CONFIG from config.py
        collect_snapshots=True,      # main.py still wants demo frames
    )

    track_snapshots = result["track_snapshots"]

    final_tracks_df = full_tracks_dataframe(result)
    save_tracks_csv(final_tracks_df, FINAL_TRACKS_CSV)
    print(f"Final tracks written to {FINAL_TRACKS_CSV}")

    plot_track_map(FINAL_TRACKS_CSV, TELEMETRY_FILE, FINAL_MAP_PNG)
    print(f"Final map written to {FINAL_MAP_PNG}")

    plot_track_map_sequence(track_snapshots, TELEMETRY_FILE, DEMO_FRAMES_DIR)
    print(f"Demo frames written to {DEMO_FRAMES_DIR}")

    print()
    print("Online pipeline run complete.")

def tracks_to_dataframe(tracks):
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