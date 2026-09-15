"""
data_io.py
----------
All disk I/O for the online pipeline lives here: reading the
telemetry/perception logs (and getting them into the shape
run_pipeline_online expects), creating the output directory, and
writing the final tracks CSV.

Pulled out of main.py so main.py stays pure orchestration logic and
the file-reading/writing details can change (e.g. a different log
format, a different output layout) without touching the streaming
loop itself. ablation_study.py does NOT use this module directly --
it goes through main.run_pipeline_online, which now calls into this
module internally.
"""

import os
import pandas as pd


def load_telemetry(telemetry_csv):
    """
    Load the vehicle telemetry log, sorted by timestamp.

    Parameters
    ----------
    telemetry_csv : str
        Path to the telemetry CSV (timestamp, x, y, yaw_rad, ...).

    Returns
    -------
    pandas.DataFrame
        Sorted by timestamp, index reset.
    """
    return pd.read_csv(telemetry_csv).sort_values("timestamp").reset_index(drop=True)


def load_perception(perception_csv):
    """
    Load the raw cone-detection log, sorted by timestamp, and return
    both the flat DataFrame and a timestamp-grouped view (what the
    online loop iterates over tick-by-tick).

    Parameters
    ----------
    perception_csv : str
        Path to the perception CSV (timestamp, rel_x_sensor,
        rel_y_sensor, confidence, cone_type, label).

    Returns
    -------
    (perception, grouped_detections) : (pandas.DataFrame, DataFrameGroupBy)
        `perception` is sorted by timestamp with index reset.
        `grouped_detections` is `perception.groupby("timestamp")`,
        used with `.groups` / `.get_group(timestamp)` in the tick loop.
    """
    perception = pd.read_csv(perception_csv).sort_values("timestamp").reset_index(drop=True)
    grouped_detections = perception.groupby("timestamp")
    return perception, grouped_detections


def ensure_output_dir(output_dir):
    """
    Create the output directory if it doesn't already exist.
    Thin wrapper kept here so main.py doesn't need its own `os`
    import just for this one call.
    """
    os.makedirs(output_dir, exist_ok=True)


def save_tracks_csv(tracks_df, output_path):
    """
    Write a final tracks DataFrame (as produced by
    main.full_tracks_dataframe) to CSV.

    Parameters
    ----------
    tracks_df : pandas.DataFrame
    output_path : str
        Destination path, e.g. config.FINAL_TRACKS_CSV.
    """
    tracks_df.to_csv(output_path, index=False)
