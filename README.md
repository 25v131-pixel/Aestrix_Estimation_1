# Aestrix State Estimation

## Problem

State Estimation and Autonomous Mapping — Sensor Fusion and Track Reconstruction.

This project reconstructs a clean 2D cone map from perception detections and vehicle telemetry.

## Pipeline

1. Time synchronization of perception and telemetry data.
2. Coordinate transformation from the backward-facing sensor frame to the vehicle frame and then to the global frame.
3. DBSCAN-based spatial association of cone detections.
4. Estimation of final cone positions from clustered observations.
5. Generation of the final reconstructed cone map.

## Project Structure

```text
Aestrix_Estimation_1/
├── data/
│   ├── perception_log.csv
│   └── telemetry_log.csv
├── src/
│   ├── association.py
│   ├── coordinate_transform.py
│   └── plot_map.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
