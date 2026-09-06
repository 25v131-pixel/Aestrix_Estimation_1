import numpy as np
import pandas as pd


def map_cones_to_global(perception_csv, telemetry_csv, camera_offset=(0.0, 0.0)):
    """
    Convert cone detections from a backward-facing sensor frame
    into global map coordinates.

    Parameters
    ----------
    perception_csv : str
        Path to the perception log CSV.

    telemetry_csv : str
        Path to the vehicle telemetry CSV.

    camera_offset : tuple/list of float
        Physical position of the sensor relative to the vehicle
        reference point, expressed in the vehicle frame.

        Default is (0.0, 0.0).
        Change this only if the challenge specification gives
        a confirmed sensor mounting offset.

    Returns
    -------
    pandas.DataFrame
        Original perception data with additional columns:
        - global_x
        - global_y
        - vehicle_x_at_detection
        - vehicle_y_at_detection
        - vehicle_yaw_at_detection
    """

    # ---------------------------------------------------------
    # 1. Load the two data streams
    # ---------------------------------------------------------

    perception = pd.read_csv(perception_csv)
    telemetry = pd.read_csv(telemetry_csv)

    # Sort by timestamp
    perception = (
        perception
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    telemetry = (
        telemetry
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2. Check that the required columns exist
    # ---------------------------------------------------------

    required_perception_columns = [
        "timestamp",
        "rel_x_sensor",
        "rel_y_sensor"
    ]

    required_telemetry_columns = [
        "timestamp",
        "x",
        "y",
        "yaw_rad"
    ]

    for column in required_perception_columns:
        if column not in perception.columns:
            raise ValueError(
                f"Missing column '{column}' in perception log."
            )

    for column in required_telemetry_columns:
        if column not in telemetry.columns:
            raise ValueError(
                f"Missing column '{column}' in telemetry log."
            )

    # ---------------------------------------------------------
    # 3. Extract timestamps and telemetry data
    # ---------------------------------------------------------

    sensor_time = perception["timestamp"].to_numpy(dtype=float)

    telemetry_time = telemetry["timestamp"].to_numpy(dtype=float)

    telemetry_xy = telemetry[["x", "y"]].to_numpy(dtype=float)

    telemetry_yaw = telemetry["yaw_rad"].to_numpy(dtype=float)

    # ---------------------------------------------------------
    # 4. Unwrap yaw
    #
    # Prevents problems when yaw crosses from +pi to -pi
    # or from -pi to +pi.
    # ---------------------------------------------------------

    telemetry_yaw_unwrapped = np.unwrap(telemetry_yaw)

    # ---------------------------------------------------------
    # 5. Time synchronization
    #
    # First check whether every sensor timestamp has an
    # exact matching telemetry timestamp.
    #
    # For the supplied dataset, this should be true because
    # both logs use the same 0.1 s timestamp grid.
    #
    # If the timestamps do not match, interpolation is used.
    # ---------------------------------------------------------

    telemetry_lookup = {
        timestamp: index
        for index, timestamp in enumerate(telemetry_time)
    }

    timestamps_match_exactly = all(
        timestamp in telemetry_lookup
        for timestamp in sensor_time
    )

    if timestamps_match_exactly:

        # -----------------------------------------------------
        # Exact timestamp match
        # -----------------------------------------------------

        telemetry_indices = np.array(
            [telemetry_lookup[t] for t in sensor_time],
            dtype=int
        )

        vehicle_xy = telemetry_xy[telemetry_indices]

        vehicle_yaw = telemetry_yaw_unwrapped[telemetry_indices]

        print("Time synchronization:")
        print("Exact timestamp matches found.")
        print("No interpolation required.")

    else:

        # -----------------------------------------------------
        # Interpolation
        #
        # Used if the perception and telemetry timestamps do
        # not exactly match.
        # -----------------------------------------------------

        if sensor_time.min() < telemetry_time.min():
            raise ValueError(
                "Some perception timestamps occur before "
                "the telemetry log begins."
            )

        if sensor_time.max() > telemetry_time.max():
            raise ValueError(
                "Some perception timestamps occur after "
                "the telemetry log ends."
            )

        # Interpolate vehicle X and Y
        vehicle_x = np.interp(
            sensor_time,
            telemetry_time,
            telemetry_xy[:, 0]
        )

        vehicle_y = np.interp(
            sensor_time,
            telemetry_time,
            telemetry_xy[:, 1]
        )

        vehicle_xy = np.column_stack(
            (vehicle_x, vehicle_y)
        )

        # Interpolate unwrapped yaw
        vehicle_yaw = np.interp(
            sensor_time,
            telemetry_time,
            telemetry_yaw_unwrapped
        )

        print("Time synchronization:")
        print("Exact timestamp matches not found.")
        print("Telemetry pose interpolated to sensor timestamps.")

    # ---------------------------------------------------------
    # 6. Extract sensor-frame cone coordinates
    # ---------------------------------------------------------

    sensor_xy = perception[
        ["rel_x_sensor", "rel_y_sensor"]
    ].to_numpy(dtype=float)

    # ---------------------------------------------------------
    # 7. Convert backward-facing sensor coordinates
    #    to vehicle coordinates
    #
    # A backward-facing sensor is rotated by 180 degrees.
    #
    #       [ -1   0 ]
    # R =   [  0  -1 ]
    #
    # Therefore:
    #
    #       x_vehicle = -x_sensor
    #       y_vehicle = -y_sensor
    #
    # Then add the physical sensor mounting offset.
    # ---------------------------------------------------------

    camera_offset = np.asarray(
        camera_offset,
        dtype=float
    )

    if camera_offset.shape != (2,):
        raise ValueError(
            "camera_offset must contain exactly two values: "
            "(x_offset, y_offset)"
        )

    vehicle_relative_xy = -sensor_xy + camera_offset

    # ---------------------------------------------------------
    # 8. Transform vehicle-frame coordinates to global frame
    #
    # p_global = p_vehicle + R(yaw) * p_relative
    #
    # where
    #
    #       [ cos(yaw)  -sin(yaw) ]
    # R =   [ sin(yaw)   cos(yaw) ]
    # ---------------------------------------------------------

    cos_yaw = np.cos(vehicle_yaw)
    sin_yaw = np.sin(vehicle_yaw)

    relative_x = vehicle_relative_xy[:, 0]
    relative_y = vehicle_relative_xy[:, 1]

    vehicle_x = vehicle_xy[:, 0]
    vehicle_y = vehicle_xy[:, 1]

    global_x = (
        vehicle_x
        + cos_yaw * relative_x
        - sin_yaw * relative_y
    )

    global_y = (
        vehicle_y
        + sin_yaw * relative_x
        + cos_yaw * relative_y
    )

    # ---------------------------------------------------------
    # 9. Create output DataFrame
    #
    # Keep all original perception columns because confidence,
    # cone_type, and label will be useful later.
    #
    # IMPORTANT:
    # 'label' is NOT used in the transformation.
    # It will only be used later for validation.
    # ---------------------------------------------------------

    output = perception.copy()

    output["global_x"] = global_x
    output["global_y"] = global_y

    output["vehicle_x_at_detection"] = vehicle_x
    output["vehicle_y_at_detection"] = vehicle_y
    output["vehicle_yaw_at_detection"] = vehicle_yaw

    return output


# =============================================================
# Example execution
# =============================================================

if __name__ == "__main__":

    perception_file = "data/perception_log.csv"
    telemetry_file = "data/telemetry_log(1).csv"

    global_cone_data = map_cones_to_global(
        perception_file,
        telemetry_file,
        camera_offset=(0.0, 0.0)
    )

    # Save transformed observations
    output_file = "outputs/global_cone_observations.csv"

    global_cone_data.to_csv(
        output_file,
        index=False
    )

    print()
    print("Coordinate transformation complete.")
    print(f"Observations processed: {len(global_cone_data)}")
    print(f"Output saved to: {output_file}")

    print()
    print("First five transformed observations:")
    print(
        global_cone_data[
            [
                "timestamp",
                "rel_x_sensor",
                "rel_y_sensor",
                "global_x",
                "global_y",
                "confidence",
                "label"
            ]
        ].head()
    )
