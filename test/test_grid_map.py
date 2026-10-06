"""
Test entry point for GridMap, using real field data.

Runs GridMap.update() over every (RGB, depth, pose) triplet produced
by FramePosePairer, i.e. real frames from testset/images +
testset/depths, each fused with the closest pose logged in
testset/pose/odometry_log.csv -- no synthetic trajectory involved
anymore.

CAMERA POSE: FramePosePairer is given the real per-camera
calibration (config/camera_intrinsics.yaml, via
CameraCalibrationConfig), so R, t returned for each frame are the
TRUE camera pose in the odom frame (odom_tform_camera), not the body
pose. This matters because points_xyz from PointCloudBuilder are in
the camera OPTICAL frame (x right, y down, z forward): applying the
body pose directly would map depth onto the world's vertical axis
(producing upward "cones").

NOISE MODEL: GridMap weights each point by the inverse of a
depth-dependent variance, sigma^2 = sigma0^2 + alpha * z^p, read
from the 'noise' section of grid_map_config.yaml. z is the depth
along the optical axis (points_xyz[:, 2]), the quantity the model
was calibrated on (fit_depth_noise.py) and the same one the 4 m
point cloud pre-filter cuts on. If PointCloudBuilder ever starts
returning points in a frame other than the camera optical frame,
the weights become wrong silently -- the sanity check below prints
depth and per-point weights so this is easy to spot. The 3D
distance is printed too, for information only: with the wide-FOV
cameras it exceeds the depth off-axis and can go beyond 4 m.

KEYFRAMES: a frame is fused only if the camera moved at least
KEYFRAME_MIN_TRANSLATION or rotated at least KEYFRAME_MIN_ROTATION_DEG
since the last FUSED frame. Without this, frames recorded while the
robot stands still (e.g. 20260914_145610 .. 145628 in this testset)
multiply the weight of the same area without adding information:
they repeat the same correlated stereo errors, inflate the cells'
confidence and dominate the no-forgetting fusion. Skipped frames
are not even run through PointCloudBuilder, saving inference time.

A/B COMPARISON: when COMPARE_WITH_UNWEIGHTED is True, a second
GridMap with noise_alpha = 0 (every point weight 1, i.e. the old
plain-mean behavior) is fused from the same point clouds, and the
two maps are compared at the end. Small differences mean point
density was already doing most of the near/far weighting.

IMAGE ORIENTATION: the testset files are used exactly as
spot_rgb_depth_log.py saves them (AUTO_ROTATE = False), i.e. in each
camera's NATIVE raster -- the one the calibration refers to. Nothing
is rotated here. PointCloudBuilder turns only the RGB upright for
SegFormer (UPRIGHT_ROT90_K: right 180 deg, front cameras ~90 deg)
and maps the probabilities back to native; depth and backprojection
stay native. Do NOT feed files rotated for display: build() rejects
them when the shape no longer matches, but a 180 deg rotation keeps
the shape and would silently flip the geometry.

MULTI-CAMERA FOLDERS: testset/images and testset/depths may hold
every camera of each shot (same timestamp, camera in the filename);
FramePosePairer selects camera_name's files only.

This script is intended for offline field-data testing only. It is
not part of the live Spot pipeline.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from perception.perception_config_loader import PerceptionConfig
from perception.class_reducer import ClassReducer
from fusion.camera_calibration_loader import CameraCalibrationConfig
from fusion.pointcloud_builder import PointCloudBuilder
from fusion.frame_pose_pairing import FramePosePairer
from mapping.grid_map_config_loader import GridMapConfig
from mapping.grid_map import GridMap
from mapping.grid_map_visualizer import GridMapVisualizer


# Fuse a second, unweighted map (noise_alpha = 0) from the same
# point clouds and compare it with the weighted one.
COMPARE_WITH_UNWEIGHTED = True

# Elevation difference [m] above which a cell is counted as
# "changed" by the weighting in the A/B report.
AB_DIFF_THRESHOLD = 0.02

# Keyframe selection: fuse a frame only if the camera moved or
# rotated at least this much since the last fused frame.
USE_KEYFRAMES = True
KEYFRAME_MIN_TRANSLATION = 0.10    # [m]
KEYFRAME_MIN_ROTATION_DEG = 5.0    # [deg]

def load_frame(rgb_path, depth_path):
    """
    Load RGB and depth exactly as Spot saved them (NATIVE raster).

    No rotation here -- see the IMAGE ORIENTATION note at the top of
    this file. Image size vs calibration and RGB/depth alignment are
    checked inside PointCloudBuilder.build().
    """

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))

    return rgb, depth


def get_field(pc, key):
    """
    Fetch a key from PointCloudBuilder.build()'s output with a clear
    error message if the key doesn't exist, instead of a bare
    KeyError -- the exact key names are assumed, not confirmed.
    """

    if key not in pc:
        raise KeyError(
            f"PointCloudBuilder.build() output has no '{key}' key. "
            f"Available keys: {list(pc.keys())}"
        )

    return pc[key]


def make_unweighted_config(config):
    """
    Return a shallow copy of the grid map config with noise_alpha
    forced to 0, i.e. every point weight 1 (the old plain mean).
    """

    unweighted = copy.copy(config)
    setattr(unweighted, "noise_alpha", 0.0)

    return unweighted


def rotation_angle_deg(R_a, R_b):
    """Angle [deg] of the relative rotation between R_a and R_b."""

    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0

    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def is_keyframe(R, t, last_R, last_t):
    """
    Whether a frame with pose (R, t) should be fused, given the pose
    of the last fused frame (None for the first frame).

    Returns
    -------
    keep : bool
    moved : float
        Translation [m] since the last fused frame (0 if first).
    rotated : float
        Rotation [deg] since the last fused frame (0 if first).
    """

    if last_t is None:
        return True, 0.0, 0.0

    moved = float(np.linalg.norm(t - last_t))
    rotated = rotation_angle_deg(last_R, R)

    keep = (
        moved >= KEYFRAME_MIN_TRANSLATION
        or rotated >= KEYFRAME_MIN_ROTATION_DEG
    )

    return keep, moved, rotated


def finite_percentiles(values, q):
    """Percentiles over the finite entries only; None if there are none."""

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return np.percentile(values, q)


def plot_layer(ax, grid_map, layer, bounds, title, cbar_label,
               cmap="viridis", symmetric=False):
    """
    Top-down heatmap of a (H, W) layer, cropped to the observed
    bounding box, in world coordinates. NaN cells are left blank.
    Rows map to world y and columns to world x, as in
    GridMap.compute_cell_indices().
    """

    row_min, row_max, col_min, col_max = bounds
    crop = layer[row_min:row_max + 1, col_min:col_max + 1]

    x0, y0 = grid_map.cell_to_world(row_min, col_min)
    x1, y1 = grid_map.cell_to_world(row_max, col_max)
    h = grid_map.resolution / 2

    kwargs = {}

    if symmetric:
        vmax = finite_percentiles(np.abs(crop), 99)
        vmax = float(vmax) if vmax is not None and vmax > 0 else 1.0
        kwargs = {"vmin": -vmax, "vmax": vmax}

    im = ax.imshow(
        np.ma.masked_invalid(crop),
        origin="lower",
        extent=[x0 - h, x1 + h, y0 - h, y1 + h],
        cmap=cmap,
        interpolation="nearest",
        **kwargs
    )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)


if __name__ == "__main__":

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    perception_config = PerceptionConfig(
        PROJECT_ROOT / "config" / "perception_config.yaml"
    )

    grid_map_config = GridMapConfig(
        PROJECT_ROOT / "config" / "grid_map_config.yaml"
    )

    camera_calibration = CameraCalibrationConfig(
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    # Which camera this dataset's testset/images + testset/depths
    # came from -- change this if you test a different camera.
    camera_name = "frontleft"

    # -------------------------------------------------------------
    # Pair real frames with real poses (true camera pose)
    # -------------------------------------------------------------

    pairer = FramePosePairer(
        rgb_dir=PROJECT_ROOT / "testset" / "images",
        depth_dir=PROJECT_ROOT / "testset" / "depths",
        pose_csv_path=PROJECT_ROOT / "testset" / "pose" / "odometry_log.csv",
        max_pose_dt=1.0,
        camera_calibration=camera_calibration,
        camera_name=camera_name
    )

    paired_frames = pairer.pair()

    if not paired_frames:
        sys.exit("No frames could be paired with a pose -- nothing to fuse.")

    # -------------------------------------------------------------
    # Build the pipeline
    # -------------------------------------------------------------

    builder = PointCloudBuilder(perception_config, camera_calibration)
    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)

    print("\n" + "-" * 60)
    print("Noise model")
    print("-" * 60)
    print(
        f"sigma^2 = {grid_map.noise_sigma0}^2 + "
        f"{grid_map.noise_alpha} * z^{grid_map.noise_exponent}, "
        f"z_ref = {grid_map.noise_reference_distance} m "
        "(z = depth along the optical axis)"
    )

    if grid_map.noise_alpha == 0:
        print(
            "noise_alpha = 0: every point has weight 1 (plain mean). "
            "Set noise.alpha in grid_map_config.yaml to enable the "
            "depth weighting."
        )

    grid_map_unweighted = None

    if COMPARE_WITH_UNWEIGHTED:
        if grid_map.noise_alpha == 0:
            print(
                "A/B comparison skipped: the main map is already "
                "unweighted."
            )
        else:
            grid_map_unweighted = GridMap(
                make_unweighted_config(grid_map_config), class_reducer
            )

            # If GridMap reads alpha from a nested 'noise' section
            # instead of config.noise_alpha, the setattr above does
            # nothing and the A/B would silently compare two
            # identical maps.
            if grid_map_unweighted.noise_alpha != 0:
                raise RuntimeError(
                    "make_unweighted_config() did not reach GridMap: "
                    f"unweighted map still has noise_alpha = "
                    f"{grid_map_unweighted.noise_alpha}. Check where "
                    "GridMap reads noise_alpha from the config."
                )

    if USE_KEYFRAMES:
        print(
            f"Keyframes: fuse only after >= {KEYFRAME_MIN_TRANSLATION} m "
            f"or >= {KEYFRAME_MIN_ROTATION_DEG} deg since the last "
            "fused frame."
        )
    else:
        print("Keyframes: disabled, every paired frame is fused.")

    print("-" * 60)

    # -------------------------------------------------------------
    # Sanity check on the first frame, before fusing everything
    # -------------------------------------------------------------

    first = paired_frames[0]
    rgb, depth = load_frame(first["rgb_path"], first["depth_path"])
    pc = builder.build(rgb, depth, camera_name)

    pts_cam = get_field(pc, "points_xyz")

    if pts_cam.shape[0] > 0:
        pts_world = pts_cam @ first["R"].T + first["t"]
        distances = np.linalg.norm(pts_cam, axis=1)
        weights = grid_map.compute_point_weights(pts_cam)

        print("\n" + "-" * 60)
        print("Sanity check (first frame)")
        print("-" * 60)
        print("Camera position (world):", first["t"])
        print(
            "Depth z [m] p1/p50/p99:",
            np.percentile(pts_cam[:, 2], [1, 50, 99])
        )
        print(
            "3D distance [m] p1/p50/p99 (info only):",
            np.percentile(distances, [1, 50, 99])
        )
        print(
            "Point weight p1/p50/p99:",
            np.percentile(weights, [1, 50, 99])
        )
        print(
            "World z [m] p1/p50/p99:",
            np.percentile(pts_world[:, 2], [1, 50, 99])
        )
        print(
            "Expected: world z mostly at or below the camera height, "
            "no values growing with depth; depth z within the 4 m "
            "cutoff (3D distance may exceed it off-axis); weights "
            "decreasing with depth (all 1 if noise_alpha = 0)."
        )
        print("-" * 60)

    print(f"\nProcessing {len(paired_frames)} paired frame(s)...\n")

    # -------------------------------------------------------------
    # Fuse every keyframe
    # -------------------------------------------------------------

    last_R, last_t = None, None
    n_fused, n_skipped = 0, 0

    for frame in paired_frames:
        R = frame["R"]
        t = frame["t"]

        if USE_KEYFRAMES:
            keep, moved, rotated = is_keyframe(R, t, last_R, last_t)

            if not keep:
                n_skipped += 1
                print(
                    f"{frame['rgb_path'].name}: SKIPPED (static: "
                    f"moved {moved * 100:.1f} cm, rotated "
                    f"{rotated:.1f} deg)"
                )
                continue

        rgb, depth = load_frame(frame["rgb_path"], frame["depth_path"])

        pc = builder.build(rgb, depth, camera_name)

        points_xyz = get_field(pc, "points_xyz").astype(
            np.float32, copy=False
        )

        semantic_colors = get_field(pc, "semantic_colors").astype(
            np.uint8, copy=False
        )

        semantic_probs = get_field(pc, "semantic_probs").astype(
            np.float32, copy=False
        )

        grid_map.update(points_xyz, semantic_colors, semantic_probs, R, t)

        if grid_map_unweighted is not None:
            grid_map_unweighted.update(
                points_xyz, semantic_colors, semantic_probs, R, t
            )

        last_R, last_t = R, t
        n_fused += 1

        print(
            f"{frame['rgb_path'].name}: "
            f"{points_xyz.shape[0]} points, pose t={t}, "
            f"pose_dt={frame['pose_dt']:.3f}s"
        )

    print(
        f"\nFused {n_fused} frame(s), skipped {n_skipped} "
        f"(out of {len(paired_frames)}).\n"
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    elevation = grid_map.get_elevation_layer()
    count = grid_map.get_count_layer()
    weight = grid_map.get_weight_layer()
    elevation_std = np.sqrt(grid_map.get_elevation_variance_layer())
    roughness_std = np.sqrt(grid_map.get_roughness_layer())

    print("Grid shape:", elevation.shape)
    print("Origin:", grid_map.get_origin())

    datum = grid_map.get_height_datum()

    if datum is None:
        print("Height datum: none (elevation in raw odom heights)")
    else:
        print(
            f"Height datum: {datum:+.3f} m odom z -> elevation 0 = "
            "ground under the starting pose"
        )
    print(
        "Observed cells:",
        np.isfinite(elevation).sum(), "/", elevation.size
    )
    print("Max observations in a single cell:", np.nanmax(count))
    print(
        "Elevation [m] min/max:",
        np.nanmin(elevation), np.nanmax(elevation)
    )

    observed = count > 0

    if np.any(observed):
        print(
            "Weight per cell p5/p50/p95:",
            np.percentile(weight[observed], [5, 50, 95])
        )

    std_pct = finite_percentiles(elevation_std, [5, 50, 95])

    if std_pct is not None:
        print(
            "Posterior elevation std [cm] p5/p50/p95:",
            std_pct * 100,
            "(optimistic: relative confidence only)"
        )

    rough_pct = finite_percentiles(roughness_std, [5, 50, 95])

    if rough_pct is not None:
        print(
            "Roughness (height spread in cell) std [cm] p5/p50/p95:",
            rough_pct * 100
        )

    bounds = grid_map.get_observed_bounds()

    if bounds is not None:
        row_min, row_max, col_min, col_max = bounds

        x_min, y_min = grid_map.cell_to_world(row_min, col_min)
        x_max, y_max = grid_map.cell_to_world(row_max, col_max)

        print(
            f"Observed footprint: x=[{x_min:.2f}, {x_max:.2f}] m, "
            f"y=[{y_min:.2f}, {y_max:.2f}] m "
            f"(grid half-extent: {grid_map.grid_map_config.map_length / 2} m)"
        )

    # -------------------------------------------------------------
    # A/B comparison: weighted vs unweighted
    # -------------------------------------------------------------

    elevation_diff = None

    if grid_map_unweighted is not None:
        elevation_unweighted = grid_map_unweighted.get_elevation_layer()
        elevation_diff = elevation - elevation_unweighted

        abs_diff = np.abs(elevation_diff[np.isfinite(elevation_diff)])

        print("\n" + "-" * 60)
        print("A/B: weighted (current noise model) vs unweighted (alpha = 0)")
        print("-" * 60)

        if abs_diff.size > 0:
            print(
                "|elevation diff| [cm] p50/p95/max:",
                np.percentile(abs_diff, 50) * 100,
                np.percentile(abs_diff, 95) * 100,
                abs_diff.max() * 100
            )
            print(
                f"Cells changed by more than "
                f"{AB_DIFF_THRESHOLD * 100:.0f} cm: "
                f"{np.count_nonzero(abs_diff > AB_DIFF_THRESHOLD)} / "
                f"{abs_diff.size} "
                f"({100 * np.mean(abs_diff > AB_DIFF_THRESHOLD):.1f}%)"
            )

        probs_w = grid_map.get_semantic_probs_layer()[observed]
        probs_u = grid_map_unweighted.get_semantic_probs_layer()[observed]

        if probs_w.size > 0:
            changed_labels = np.count_nonzero(
                probs_w.argmax(axis=-1) != probs_u.argmax(axis=-1)
            )

            print(
                "Cells whose most likely class changed: "
                f"{changed_labels} / {probs_w.shape[0]} "
                f"({100 * changed_labels / probs_w.shape[0]:.1f}%)"
            )

        print(
            "Small differences mean point density was already "
            "favoring near observations; large ones concentrate "
            "where cells were seen both from far and from close."
        )
        print("-" * 60)

    # -------------------------------------------------------------
    # Visualize: 2D elevation heatmap + 3D semantic scatter
    # -------------------------------------------------------------

    visualizer = GridMapVisualizer(grid_map)

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    visualizer.plot_elevation_heatmap(ax=ax1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    visualizer.plot_semantic_3d(ax=ax2)

    # Equal-ish scale on the 3D axes, so z isn't visually stretched.
    xs = np.array(ax2.get_xlim3d())
    ys = np.array(ax2.get_ylim3d())
    zs = np.array(ax2.get_zlim3d())
    ax2.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Visualize: uncertainty, roughness and A/B difference
    # -------------------------------------------------------------

    if bounds is not None:
        n_panels = 3 if elevation_diff is not None else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

        plot_layer(
            axes[0], grid_map, elevation_std * 100, bounds,
            "Posterior elevation std", "std [cm]"
        )

        plot_layer(
            axes[1], grid_map, roughness_std * 100, bounds,
            "Roughness (height spread in cell)", "std [cm]",
            cmap="magma"
        )

        if elevation_diff is not None:
            plot_layer(
                axes[2], grid_map, elevation_diff * 100, bounds,
                "Elevation: weighted - unweighted", "diff [cm]",
                cmap="RdBu_r", symmetric=True
            )

        plt.tight_layout()
        plt.show()

    fig, ax = visualizer.plot_semantic_topdown()
    plt.show()

    fig3d = visualizer.plot_semantic_3d_plotly()
    fig3d.write_html(PROJECT_ROOT / "grid_map_semantic_3d.html")
    fig3d.show()