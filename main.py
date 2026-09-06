import os

from src.coordinate_transform import map_cones_to_global
from src.plot_map import plot_global_map


PERCEPTION_FILE = "data/perception_log.csv"
TELEMETRY_FILE = "data/telemetry_log(1).csv"

GLOBAL_CSV = "outputs/global_cone_observations.csv"
MAP_PNG = "outputs/global_map.png"


def main():
    print("Aestrix State Estimation and Autonomous Mapping")
    print()

    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------------------------
    # Stage 1: time alignment + coordinate transform
    # ---------------------------------------------------------
    print("Stage 1: coordinate transformation")
    global_cone_data = map_cones_to_global(
        PERCEPTION_FILE,
        TELEMETRY_FILE,
        camera_offset=(0.0, 0.0),
    )
    global_cone_data.to_csv(GLOBAL_CSV, index=False)
    print(f"  {len(global_cone_data)} observations written to {GLOBAL_CSV}")
    print()

    # ---------------------------------------------------------
    # Stage 2: plot
    # ---------------------------------------------------------
    print("Stage 2: plotting global map")
    plot_global_map(GLOBAL_CSV, TELEMETRY_FILE, MAP_PNG)
    print(f"  map saved to {MAP_PNG}")
    print()

    # ---------------------------------------------------------
    # Stage 3: ghost filtering            [not yet implemented]
    # Stage 4: data association/clustering [not yet implemented]
    # Stage 5: validation                 [not yet implemented]
    # ---------------------------------------------------------


if __name__ == "__main__":
    main()
