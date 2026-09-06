import pandas as pd
import matplotlib.pyplot as plt


def plot_global_map(global_csv, telemetry_csv, output_file):
    # Load transformed cone observations
    cones = pd.read_csv(global_csv)

    # Load vehicle trajectory
    telemetry = pd.read_csv(telemetry_csv)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot left cones
    left = cones[cones["cone_type"] == "left"]

    plt.scatter(
        left["global_x"],
        left["global_y"],
        s=10,
        label="Left cones"
    )

    # Plot right cones
    right = cones[cones["cone_type"] == "right"]

    plt.scatter(
        right["global_x"],
        right["global_y"],
        s=10,
        label="Right cones"
    )

    # Plot unknown detections, if any
    unknown = cones[cones["cone_type"] == "unknown"]

    if len(unknown) > 0:
        plt.scatter(
            unknown["global_x"],
            unknown["global_y"],
            s=10,
            label="Unknown"
        )

    # Plot vehicle trajectory
    plt.plot(
        telemetry["x"],
        telemetry["y"],
        linewidth=2,
        label="Vehicle trajectory"
    )

    # Mark starting position
    plt.scatter(
        telemetry["x"].iloc[0],
        telemetry["y"].iloc[0],
        s=80,
        marker="o",
        label="Start"
    )

    # Labels and formatting
    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.title("Global Cone Observations and Vehicle Trajectory")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # Save figure
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    global_csv = "outputs/global_cone_observations.csv"
    telemetry_csv = "data/telemetry_log(1).csv"
    output_file = "outputs/global_map.png"

    plot_global_map(
        global_csv,
        telemetry_csv,
        output_file
    )

    print()
    print(f"Map saved to: {output_file}")
