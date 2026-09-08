import os
import pandas as pd
import matplotlib.pyplot as plt

from src.coordinate_transform import map_cones_to_global
from src.association import dbscan_association


PERCEPTION_FILE = "data/perception_log.csv"
TELEMETRY_FILE = "data/telemetry_log.csv"

GLOBAL_CSV = "outputs/global_cone_observations.csv"
FINAL_CSV = "outputs/final_cone_map.csv"
MAP_PNG = "outputs/final_reconstructed_map.png"


def main():

    print("Aestrix State Estimation and Autonomous Mapping")
    print()

    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------
    # Stage 1: Time alignment + coordinate transformation
    # --------------------------------------------------

    print("Stage 1: coordinate transformation")

    global_cone_data = map_cones_to_global(
        PERCEPTION_FILE,
        TELEMETRY_FILE,
        camera_offset=(0.0, 0.0),
    )

    global_cone_data.to_csv(
        GLOBAL_CSV,
        index=False
    )

    print(
        f"  {len(global_cone_data)} observations written to "
        f"{GLOBAL_CSV}"
    )

    print()

    # --------------------------------------------------
    # Stage 2: DBSCAN association + noise rejection
    # --------------------------------------------------

    print("Stage 2: DBSCAN association")

    dbscan_result, dbscan_cones = dbscan_association(
        global_cone_data,
        eps=1.5,
        min_samples=5
    )

    print(f"  Estimated cones: {len(dbscan_cones)}")

    print(
        f"  Left cones: "
        f"{(dbscan_cones['cone_type'] == 'left').sum()}"
    )

    print(
        f"  Right cones: "
        f"{(dbscan_cones['cone_type'] == 'right').sum()}"
    )

    print(
        f"  Noise observations: "
        f"{(dbscan_result['cluster_id'] == -1).sum()}"
    )

    print()

    # --------------------------------------------------
    # Stage 3: Save final cone map
    # --------------------------------------------------

    print("Stage 3: saving final cone map")

    final_cones = dbscan_cones[
        ["cluster_id", "cone_type", "x", "y", "count"]
    ].copy()

    final_cones = final_cones.rename(
        columns={"count": "observations"}
    )

    final_cones = final_cones.sort_values(
        ["cone_type", "cluster_id"]
    ).reset_index(drop=True)

    final_cones.to_csv(
        FINAL_CSV,
        index=False
    )

    print(
        f"  Final cone list written to {FINAL_CSV}"
    )

    print()

    # --------------------------------------------------
    # Stage 4: Generate final reconstructed map
    # --------------------------------------------------

    print("Stage 4: generating final reconstructed map")

    left_cones = final_cones[
        final_cones["cone_type"] == "left"
    ]

    right_cones = final_cones[
        final_cones["cone_type"] == "right"
    ]

    telemetry = pd.read_csv(TELEMETRY_FILE)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        left_cones["x"],
        left_cones["y"],
        s=80,
        marker="o",
        label="Reconstructed left cones"
    )

    plt.scatter(
        right_cones["x"],
        right_cones["y"],
        s=80,
        marker="o",
        label="Reconstructed right cones"
    )

    plt.plot(
        telemetry["x"],
        telemetry["y"],
        linewidth=2,
        label="Vehicle trajectory"
    )

    plt.scatter(
        telemetry["x"].iloc[0],
        telemetry["y"].iloc[0],
        s=100,
        marker="*",
        label="Start"
    )

    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.title("Final Reconstructed Cone Map")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        MAP_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Final map written to {MAP_PNG}")
    print()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
