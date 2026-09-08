"""
transform.py
------------
Converts raw cone detections (backward-facing sensor frame) into the
global map frame.

Pipeline for each detection:
    sensor frame  --[180 deg rotation]-->  vehicle frame
    vehicle frame --[+ vehicle pose (x, y, yaw)]--> global frame

The perception log's 'label' column (real/ghost) is NEVER used here.
It is carried through the output purely so it can be used later, by
the user, to score the pipeline against ground truth. It plays no
role in the transform, clustering, or filtering math.
"""

import numpy as np
import pandas as pd


REQUIRED_PERCEPTION_COLUMNS = ["timestamp", "rel_x_sensor", "rel_y_sensor"]
REQUIRED_TELEMETRY_COLUMNS = ["timestamp", "x", "y", "yaw_rad"]


def load_and_transform(perception_csv, telemetry_csv, camera_offset=(0.0, 0.0)):
    """
    Parameters
    ----------
    perception_csv : str
        Path to the perception log CSV.
    telemetry_csv : str
        Path to the telemetry log CSV.
    camera_offset : (float, float)
        Sensor mounting offset in the vehicle frame. Defaults to (0, 0)
        because no confirmed mounting offset was given in the spec.

    Returns
    -------
    pandas.DataFrame
        One row per detection, with:
            timestamp, cone_type, confidence,           (from perception log)
            global_x, global_y,                         (computed)
            vehicle_x, vehicle_y, vehicle_yaw,           (pose used)
            label                                        (validation only)
    """

    perception = pd.read_csv(perception_csv).sort_values("timestamp").reset_index(drop=True)
    telemetry = pd.read_csv(telemetry_csv).sort_values("timestamp").reset_index(drop=True)

    for col in REQUIRED_PERCEPTION_COLUMNS:
        if col not in perception.columns:
            raise ValueError(f"Missing column '{col}' in perception log.")
    for col in REQUIRED_TELEMETRY_COLUMNS:
        if col not in telemetry.columns:
            raise ValueError(f"Missing column '{col}' in telemetry log.")

    # ---------------------------------------------------------------
    # Time alignment
    #
    # The spec for this dataset guarantees both logs share the same
    # 0.1s grid, so we do an exact merge on timestamp rather than
    # interpolating. If that assumption ever breaks, fail loudly
    # instead of silently interpolating bad data.
    # ---------------------------------------------------------------
    telemetry = telemetry.copy()
    telemetry["yaw_rad"] = np.unwrap(telemetry["yaw_rad"].to_numpy(dtype=float))

    merged = perception.merge(
        telemetry[["timestamp", "x", "y", "yaw_rad"]],
        on="timestamp",
        how="left",
        validate="many_to_one",
    )

    missing = merged["x"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} perception rows have no exact telemetry timestamp match. "
            "Interpolation would be required but was not expected for this dataset."
        )

    # ---------------------------------------------------------------
    # 180-degree backward-sensor correction: sensor -> vehicle frame
    # ---------------------------------------------------------------
    camera_offset = np.asarray(camera_offset, dtype=float)
    if camera_offset.shape != (2,):
        raise ValueError("camera_offset must be (x_offset, y_offset).")

    sensor_xy = merged[["rel_x_sensor", "rel_y_sensor"]].to_numpy(dtype=float)
    vehicle_relative_xy = -sensor_xy + camera_offset  # 180 deg rotation

    # ---------------------------------------------------------------
    # vehicle frame -> global frame
    # ---------------------------------------------------------------
    yaw = merged["yaw_rad"].to_numpy(dtype=float)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    rel_x, rel_y = vehicle_relative_xy[:, 0], vehicle_relative_xy[:, 1]
    veh_x, veh_y = merged["x"].to_numpy(dtype=float), merged["y"].to_numpy(dtype=float)

    global_x = veh_x + cos_yaw * rel_x - sin_yaw * rel_y
    global_y = veh_y + sin_yaw * rel_x + cos_yaw * rel_y

    out = pd.DataFrame({
        "timestamp": merged["timestamp"],
        "cone_type": merged["cone_type"],
        "confidence": merged["confidence"],
        "global_x": global_x,
        "global_y": global_y,
        "vehicle_x": veh_x,
        "vehicle_y": veh_y,
        "vehicle_yaw": yaw,
    })

    # Kept ONLY for later validation by the user - not used upstream.
    if "label" in merged.columns:
        out["label"] = merged["label"]

    return out
