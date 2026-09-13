import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def covariance_to_ellipse_params(covariance, n_std=2.0):
    """
    Convert a 2x2 position covariance matrix into ellipse
    parameters for plotting.

    Parameters
    ----------
    covariance : array-like, shape (2, 2)
        Position covariance of a track, as maintained by the
        static Kalman filter.

    n_std : float
        Number of standard deviations the ellipse should span.
        2.0 corresponds to roughly a 95% confidence region for
        a 2D Gaussian.

    Returns
    -------
    width, height, angle_deg : float
        Full width and height of the ellipse (not semi-axes),
        and its rotation angle in degrees, ready to pass to
        matplotlib.patches.Ellipse.
    """

    covariance = np.asarray(covariance, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # eigh returns eigenvalues in ascending order; take the
    # largest-eigenvalue eigenvector to define the major axis.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Guard against tiny negative eigenvalues from floating
    # point noise on a near-singular covariance.
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    angle_deg = np.degrees(
        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )

    return width, height, angle_deg

def plot_track_map(tracks_csv, telemetry_csv, output_file, n_std=2.0):
    """
    Plot the online pipeline's track-based cone map, including
    per-track covariance ellipses and lifecycle status.

    Expects tracks_csv to contain one row per track with columns:
        track_id, cone_type, x, y, status,
        cov_xx, cov_xy, cov_yy

    where status is one of "confirmed", "candidate", or "ghost",
    and cov_xx/cov_xy/cov_yy are the entries of the track's 2x2
    position covariance from the Kalman filter.
    """

    tracks = pd.read_csv(tracks_csv)
    telemetry = pd.read_csv(telemetry_csv)

    fig, ax = plt.subplots(figsize=(12, 8))

    status_styles = {
        "confirmed": {"color": "tab:green", "alpha_ellipse": 0.25},
        "candidate": {"color": "tab:orange", "alpha_ellipse": 0.20},
        "ghost": {"color": "tab:red", "alpha_ellipse": 0.15},
    }

    for status, style in status_styles.items():

        subset = tracks[tracks["status"] == status]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset["x"],
            subset["y"],
            s=40,
            marker="o",
            color=style["color"],
            label=f"{status.capitalize()} tracks"
        )

        # Draw a covariance ellipse for each track individually,
        # since each track has its own uncertainty.
        for _, track in subset.iterrows():

            covariance = np.array([
                [track["cov_xx"], track["cov_xy"]],
                [track["cov_xy"], track["cov_yy"]]
            ])

            width, height, angle_deg = covariance_to_ellipse_params(
                covariance,
                n_std=n_std
            )

            ellipse = Ellipse(
                xy=(track["x"], track["y"]),
                width=width,
                height=height,
                angle=angle_deg,
                facecolor=style["color"],
                alpha=style["alpha_ellipse"],
                edgecolor=style["color"],
                linewidth=0.5
            )

            ax.add_patch(ellipse)

    # Plot vehicle trajectory
    ax.plot(
        telemetry["x"],
        telemetry["y"],
        linewidth=2,
        color="tab:blue",
        label="Vehicle trajectory"
    )

    ax.scatter(
        telemetry["x"].iloc[0],
        telemetry["y"].iloc[0],
        s=80,
        marker="*",
        color="black",
        label="Start"
    )

    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.set_title("Online Cone Map with Track Uncertainty")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return fig

def plot_global_map(global_csv, telemetry_csv, output_file):
    """
    OFFLINE / BATCH USE ONLY (Phase 1)

    Plots raw per-observation cone detections colored by
    cone_type, with no uncertainty information. Expects
    global_csv to be the output of coordinate_transform's
    map_cones_to_global (or the DBSCAN association step),
    not the online pipeline's track output.

    Retained for regression comparison against the online
    pipeline's plot_track_map(). Use plot_track_map() for
    Phase 2 / streaming output.
    """

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

def plot_track_map_sequence(track_snapshots, telemetry_csv, output_dir, n_std=2.0):
    """
    Render a sequence of track-state snapshots as numbered PNG
    frames, for building the live-demo video (deliverable #6).

    Parameters
    ----------
    track_snapshots : list of (timestamp, DataFrame)
        Each DataFrame has the same schema as plot_track_map's
        tracks_csv, representing the track state at that
        timestamp during the streaming run.

    telemetry_csv : str
        Full telemetry log, used to draw the trajectory up to
        each snapshot's timestamp (not the whole trajectory,
        so the vehicle path also builds up incrementally).

    output_dir : str
        Directory to write frame_0000.png, frame_0001.png, etc.
    """

    import os
    os.makedirs(output_dir, exist_ok=True)

    telemetry = pd.read_csv(telemetry_csv)

    for frame_index, (timestamp, tracks_snapshot) in enumerate(track_snapshots):

        telemetry_so_far = telemetry[telemetry["timestamp"] <= timestamp]

        frame_tracks_csv = os.path.join(
            output_dir, f"_tmp_tracks_{frame_index}.csv"
        )
        tracks_snapshot.to_csv(frame_tracks_csv, index=False)

        frame_telemetry_csv = os.path.join(
            output_dir, f"_tmp_telemetry_{frame_index}.csv"
        )
        telemetry_so_far.to_csv(frame_telemetry_csv, index=False)

        frame_output = os.path.join(
            output_dir, f"frame_{frame_index:04d}.png"
        )

        plot_track_map(
            frame_tracks_csv,
            frame_telemetry_csv,
            frame_output,
            n_std=n_std
        )

        os.remove(frame_tracks_csv)
        os.remove(frame_telemetry_csv)

if __name__ == "__main__":

    global_csv = "outputs/global_cone_observations.csv"
    telemetry_csv = "data/telemetry_log.csv"
    output_file = "outputs/global_map.png"

    plot_global_map(
        global_csv,
        telemetry_csv,
        output_file
    )

    print()
    print("Offline map saved to: outputs/global_map.png")

    # Online / Phase 2 track-based plot with uncertainty
    plot_track_map(
        tracks_csv="outputs/final_tracks.csv",
        telemetry_csv="data/telemetry_log.csv",
        output_file="outputs/final_track_map.png"
    )
    print("Online track map saved to: outputs/final_track_map.png")
