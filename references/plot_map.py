"""
plot_map.py
-----------
Visualizes the finalized cone map: real (kept) cones vs. ghost
candidates (rejected by persistence), using their fused Kalman
positions and uncertainty.

This module only plots what it's handed - it does not recompute
anything, does not touch the 'label' column, and does not decide
what counts as a ghost (that decision already lives in cone_uncertainty
/ the 'is_ghost_candidate' column produced upstream).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_cone_map(
    fused_clusters,
    output_path=None,
    show_uncertainty_ellipses=True,
    ellipse_n_std=2.0,
    title="Finalized Cone Map",
):
    """
    Parameters
    ----------
    fused_clusters : pandas.DataFrame
        Must contain: global_x, global_y, cone_type, is_ghost_candidate.
        Optionally std_x, std_y, cov_xy (used for uncertainty ellipses).
        This is exactly the DataFrame produced by
        static_kalman_filter.fuse_all_clusters (after the
        is_ghost_candidate column has been merged in), i.e. the
        `fused_all` variable in main.py, or a re-read of
        cone_uncertainty.csv.
    output_path : str or None
        If given, saves the figure to this path (e.g. "cone_map.png").
    show_uncertainty_ellipses : bool
        If True, draws a covariance ellipse around each point using
        std_x / std_y / cov_xy (skipped if those columns are absent).
    ellipse_n_std : float
        Number of standard deviations the ellipse radius represents.
    title : str
        Plot title.

    Returns
    -------
    (fig, ax) : the matplotlib Figure and Axes, so the caller can
        further customize or show/save it themselves.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    real = fused_clusters[~fused_clusters["is_ghost_candidate"]]
    ghosts = fused_clusters[fused_clusters["is_ghost_candidate"]]

    # Real cones, split by side so left/right boundaries are visible.
    for cone_type, color, marker in [
        ("left", "tab:blue", "o"),
        ("right", "tab:orange", "o"),
        ("unknown", "tab:green", "o"),
    ]:
        subset = real[real["cone_type"] == cone_type]
        if len(subset):
            ax.scatter(
                subset["global_x"], subset["global_y"],
                c=color, marker=marker, s=40,
                label=f"Real cone ({cone_type})", zorder=3,
            )

    # Ghost candidates - flagged, rejected from the final map.
    if len(ghosts):
        ax.scatter(
            ghosts["global_x"], ghosts["global_y"],
            c="red", marker="x", s=60, linewidths=2,
            label="Ghost candidate (rejected)", zorder=4,
        )

    # Uncertainty ellipses for the kept real cones.
    has_cov_cols = {"std_x", "std_y", "cov_xy"}.issubset(fused_clusters.columns)
    if show_uncertainty_ellipses and has_cov_cols:
        for _, row in real.iterrows():
            _draw_cov_ellipse(
                ax, row["global_x"], row["global_y"],
                row["std_x"], row["std_y"], row["cov_xy"],
                n_std=ellipse_n_std,
            )

    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax


def _draw_cov_ellipse(ax, x, y, std_x, std_y, cov_xy, n_std=2.0):
    """
    Draws a covariance ellipse from std_x, std_y, cov_xy (the 2x2
    covariance matrix's components) around (x, y).
    """
    import numpy as np

    cov = np.array([[std_x ** 2, cov_xy], [cov_xy, std_y ** 2]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=0, a_max=None)  # guard tiny negatives

    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        (x, y), width=width, height=height, angle=angle,
        edgecolor="gray", facecolor="none", linestyle="-",
        linewidth=0.8, alpha=0.6, zorder=2,
    )
    ax.add_patch(ellipse)
