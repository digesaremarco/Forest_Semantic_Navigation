"""
Build a GridMap from real field data, save it and plot it.

Exercises the full fusion pipeline end to end (pairing -> point cloud
-> GridMap) on a recorded run, writes the result to disk and saves
the map plots. Intended for offline field-data testing only, not part
of the live Spot pipeline.

INPUT: <DATA_ROOT>/images_rgb/<camera>/, <DATA_ROOT>/depth_data/<camera>/
and <DATA_ROOT>/odometry/odometry_log.csv, as written by
spot_rgb_depth_log.py (AUTO_ROTATE = False).

OUTPUT (in OUTPUT_DIR):
    grid_map.npz     GridMap.save() state: raw accumulators, origin,
                     height datum, noise model, class order. Reload
                     with GridMap(same config, ClassReducer).load().
    trajectory.npz   One entry per paired frame, in fusion order:
                     timestamp, camera_name, R, t (camera pose in
                     odom), fused (bool, False = skipped keyframe).
                     For overlaying the path on the map later.
    elevation.png         Elevation heatmap (top-down).
    semantic_topdown.png  Winning class per cell, top-down, legend.
    semantic_3d.png       Matplotlib 3D scatter, winning class, legend
                          (subsampled above PLOT_3D_MAX_POINTS cells).
    semantic_3d.html      Interactive Plotly 3D scatter, legend, full
                          map; open it in a browser.

MULTI-CAMERA: every camera in CAMERA_NAMES is fused into the SAME
grid. One FramePosePairer per camera (own folders, own extrinsics);
frames of all cameras are merged in chronological order, so the grid
origin and height datum anchor on the earliest frame of the run, and
each frame is fused with ITS OWN camera_name (intrinsics, depth_scale,
upright rotation for SegFormer).

CAMERA POSE: FramePosePairer composes odom_tform_body with the static
body_tform_camera, so R, t are the TRUE camera pose in odom.
points_xyz from PointCloudBuilder are in the camera OPTICAL frame
(x right, y down, z forward).

NOISE MODEL: GridMap weights each point by the inverse of
sigma^2 = sigma0^2 + alpha * z^p, z = depth along the optical axis
(grid_map_config.yaml, 'noise'). The sanity check prints depth and
weights per camera, so a point cloud returned in the wrong frame is
easy to spot.

KEYFRAMES: a frame is fused only if ITS camera moved >=
KEYFRAME_MIN_TRANSLATION or rotated >= KEYFRAME_MIN_ROTATION_DEG since
the last frame fused from that same camera. Static frames would repeat
the same correlated stereo errors and inflate the cells' confidence.
Skipped frames never reach PointCloudBuilder.

IMAGE ORIENTATION: files are used exactly as Spot saved them (NATIVE
raster). PointCloudBuilder turns only the RGB upright for SegFormer
and maps the probabilities back. Do NOT feed files rotated for
display: a 180 deg rotation keeps the shape and would silently flip
the geometry.
"""

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


# Cameras fused into the same grid. 'back' is left out on purpose.
CAMERA_NAMES = ["frontleft", "frontright", "left", "right"]

# Dataset layout: <DATA_ROOT>/images_rgb/<camera>/, depth_data/<camera>/
# and odometry/odometry_log.csv.
DATA_ROOT = PROJECT_ROOT / "oggi2"

# Where grid_map.npz and trajectory.npz are written.
OUTPUT_DIR = DATA_ROOT / "output"

# Keyframe selection: fuse a frame only if its camera moved or
# rotated at least this much since the last frame fused from it.
USE_KEYFRAMES = True
KEYFRAME_MIN_TRANSLATION = 0.10    # [m]
KEYFRAME_MIN_ROTATION_DEG = 5.0    # [deg]

MAX_POSE_DT = 1.0                  # [s] pairing tolerance

# Plots: resolution of the saved PNGs, whether to also open them
# (Matplotlib windows + Plotly in the browser), and the cell budget of
# the Matplotlib 3D scatter (it gets slow and muddy beyond that).
FIGURE_DPI = 200
SHOW_PLOTS = True
PLOT_3D_MAX_POINTS = 150000


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def load_frame(rgb_path, depth_path):
    """Load RGB and depth exactly as Spot saved them (NATIVE raster)."""

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))

    return rgb, depth


def get_field(pc, key):
    """PointCloudBuilder.build() output field, with a clear error."""

    if key not in pc:
        raise KeyError(
            f"PointCloudBuilder.build() output has no '{key}' key. "
            f"Available keys: {list(pc.keys())}"
        )

    return pc[key]


def rotation_angle_deg(R_a, R_b):
    """Angle [deg] of the relative rotation between R_a and R_b."""

    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0

    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def is_keyframe(R, t, last_R, last_t):
    """
    Whether a frame with pose (R, t) should be fused, given the pose
    of the last fused frame of the same camera (None if none yet).

    Returns (keep, moved [m], rotated [deg]).
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


def section(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# -----------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------

def pair_all_cameras(camera_calibration):
    """
    One FramePosePairer per camera; returns (paired_by_camera,
    paired_frames) with paired_frames merged chronologically and
    tagged with their camera_name.
    """

    paired_by_camera = {}

    for camera_name in CAMERA_NAMES:
        pairer = FramePosePairer(
            rgb_dir=DATA_ROOT / "images_rgb" / camera_name,
            depth_dir=DATA_ROOT / "depth_data" / camera_name,
            pose_csv_path=DATA_ROOT / "odometry" / "odometry_log.csv",
            max_pose_dt=MAX_POSE_DT,
            camera_calibration=camera_calibration,
            camera_name=camera_name
        )

        frames = pairer.pair()

        if not frames:
            print(f"WARNING: no frames paired for '{camera_name}', skipping it.")
            continue

        for frame in frames:
            frame["camera_name"] = camera_name

        paired_by_camera[camera_name] = frames

    if not paired_by_camera:
        sys.exit("No frames could be paired with a pose -- nothing to fuse.")

    # Chronological across cameras, so origin and height datum anchor
    # on the earliest frame of the run. Ties (same shot) broken by
    # CAMERA_NAMES order, i.e. front cameras first: they see more
    # ground ahead, which steadies the datum estimate.
    order = {name: i for i, name in enumerate(CAMERA_NAMES)}

    paired_frames = sorted(
        (frame for frames in paired_by_camera.values() for frame in frames),
        key=lambda f: (f["timestamp"], order[f["camera_name"]])
    )

    print("\nPaired frames per camera:")

    for camera_name, frames in paired_by_camera.items():
        print(f"  {camera_name:<12} {len(frames)}")

    return paired_by_camera, paired_frames


def sanity_check(paired_by_camera, builder, grid_map):
    """
    Point cloud statistics on the first frame of EACH camera. A camera
    with a wrong orientation/extrinsic shows up as world z off from
    the others (far above the camera, or growing with depth).
    """

    for camera_name, frames in paired_by_camera.items():
        first = frames[0]
        rgb, depth = load_frame(first["rgb_path"], first["depth_path"])
        pts_cam = get_field(builder.build(rgb, depth, camera_name), "points_xyz")

        section(f"Sanity check ({camera_name}, first frame)")

        if pts_cam.shape[0] == 0:
            print("No valid points in this frame.")
            continue

        pts_world = pts_cam @ first["R"].T + first["t"]

        print("Camera position (world):", first["t"])
        print("Depth z [m] p1/p50/p99:",
              np.percentile(pts_cam[:, 2], [1, 50, 99]))
        print("Point weight p1/p50/p99:",
              np.percentile(grid_map.compute_point_weights(pts_cam), [1, 50, 99]))
        print("World z [m] p1/p50/p99:",
              np.percentile(pts_world[:, 2], [1, 50, 99]))

    print(
        "\nExpected for every camera: world z mostly at or below the "
        "camera height and consistent ACROSS cameras (same ground), "
        "depth z within the 4 m cutoff, weights decreasing with depth."
    )


def fuse(paired_frames, builder, grid_map):
    """
    Fuse every keyframe into grid_map (keyframe state kept per
    camera). Returns the per-frame trajectory records.
    """

    cameras = {f["camera_name"] for f in paired_frames}
    last_pose = {name: (None, None) for name in cameras}
    fused_count = {name: 0 for name in cameras}
    skipped_count = {name: 0 for name in cameras}

    trajectory = []

    print(f"\nProcessing {len(paired_frames)} paired frame(s)...\n")

    for frame in paired_frames:
        camera_name = frame["camera_name"]
        R, t = frame["R"], frame["t"]

        keep = True

        if USE_KEYFRAMES:
            last_R, last_t = last_pose[camera_name]
            keep, moved, rotated = is_keyframe(R, t, last_R, last_t)

        trajectory.append({
            "timestamp": frame["timestamp"],
            "camera_name": camera_name,
            "R": R,
            "t": t,
            "fused": keep,
        })

        if not keep:
            skipped_count[camera_name] += 1
            print(
                f"{frame['rgb_path'].name}: SKIPPED (static: moved "
                f"{moved * 100:.1f} cm, rotated {rotated:.1f} deg)"
            )
            continue

        rgb, depth = load_frame(frame["rgb_path"], frame["depth_path"])
        pc = builder.build(rgb, depth, camera_name)

        points_xyz = get_field(pc, "points_xyz").astype(np.float32, copy=False)
        semantic_colors = get_field(pc, "semantic_colors").astype(np.uint8, copy=False)
        semantic_probs = get_field(pc, "semantic_probs").astype(np.float32, copy=False)

        grid_map.update(points_xyz, semantic_colors, semantic_probs, R, t)

        last_pose[camera_name] = (R, t)
        fused_count[camera_name] += 1

        print(
            f"{frame['rgb_path'].name}: {points_xyz.shape[0]} points, "
            f"pose t={t}, pose_dt={frame['pose_dt']:.3f}s"
        )

    print("\nFused / skipped per camera:")

    for camera_name in [c for c in CAMERA_NAMES if c in cameras]:
        print(
            f"  {camera_name:<12} fused {fused_count[camera_name]:>4}, "
            f"skipped {skipped_count[camera_name]:>4}"
        )

    print(
        f"Total: fused {sum(fused_count.values())}, "
        f"skipped {sum(skipped_count.values())} "
        f"(out of {len(paired_frames)})."
    )

    return trajectory


def print_summary(grid_map):
    """Text summary of the fused map."""

    section("Map summary")

    elevation = grid_map.get_elevation_layer()
    count = grid_map.get_count_layer()
    observed = count > 0

    print("Grid shape:", elevation.shape)
    print("Origin:", grid_map.get_origin())

    datum = grid_map.get_height_datum()

    if datum is None:
        print("Height datum: none (elevation in raw odom heights)")
    else:
        print(f"Height datum: {datum:+.3f} m odom z "
              "(elevation 0 = ground under the starting pose)")

    print("Observed cells:", int(observed.sum()), "/", elevation.size)

    if not np.any(observed):
        print("No observed cells.")
        return

    print("Max observations in a single cell:", float(count.max()))
    print("Elevation [m] min/max:", np.nanmin(elevation), np.nanmax(elevation))
    print("Weight per cell p5/p50/p95:",
          np.percentile(grid_map.get_weight_layer()[observed], [5, 50, 95]))

    std_pct = finite_percentiles(
        np.sqrt(grid_map.get_elevation_variance_layer()), [5, 50, 95]
    )

    if std_pct is not None:
        print("Posterior elevation std [cm] p5/p50/p95:", std_pct * 100,
              "(optimistic: relative confidence only)")

    rough_pct = finite_percentiles(
        np.sqrt(grid_map.get_roughness_layer()), [5, 50, 95]
    )

    if rough_pct is not None:
        print("Roughness std [cm] p5/p50/p95:", rough_pct * 100)

    row_min, row_max, col_min, col_max = grid_map.get_observed_bounds()
    x_min, y_min = grid_map.cell_to_world(row_min, col_min)
    x_max, y_max = grid_map.cell_to_world(row_max, col_max)
    half = grid_map.grid_map_config.map_length / 2

    print(f"Observed footprint: x=[{x_min:.2f}, {x_max:.2f}] m, "
          f"y=[{y_min:.2f}, {y_max:.2f}] m (grid half-extent: {half} m)")


def save_outputs(grid_map, trajectory, grid_map_config, class_reducer):
    """
    Save the map and the trajectory, then reload the map into a fresh
    GridMap and check every layer matches, so the file is known to
    be usable by the visualization script.
    """

    section("Saving")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    map_path = OUTPUT_DIR / "grid_map.npz"
    trajectory_path = OUTPUT_DIR / "trajectory.npz"

    grid_map.save(map_path)

    np.savez_compressed(
        trajectory_path,
        timestamp=np.array([r["timestamp"] for r in trajectory], dtype=np.float64),
        camera_name=np.array([r["camera_name"] for r in trajectory]),
        R=np.stack([np.asarray(r["R"], dtype=np.float64) for r in trajectory]),
        t=np.stack([np.asarray(r["t"], dtype=np.float64) for r in trajectory]),
        fused=np.array([r["fused"] for r in trajectory], dtype=bool),
    )

    print(f"Trajectory saved to {trajectory_path} ({len(trajectory)} frame(s)).")

    # Round-trip check.
    reloaded = GridMap(grid_map_config, class_reducer)
    reloaded.load(map_path)

    layers = {
        "elevation": GridMap.get_elevation_layer,
        "elevation_variance": GridMap.get_elevation_variance_layer,
        "roughness": GridMap.get_roughness_layer,
        "rgb": GridMap.get_rgb_layer,
        "semantic_probs": GridMap.get_semantic_probs_layer,
        "count": GridMap.get_count_layer,
    }

    mismatched = [
        name for name, getter in layers.items()
        if not np.allclose(getter(grid_map), getter(reloaded), equal_nan=True)
    ]

    if reloaded.get_height_datum() != grid_map.get_height_datum():
        mismatched.append("height_datum")

    if mismatched:
        raise RuntimeError(
            f"Reloaded map differs from the fused one in: {mismatched}"
        )

    print("Round-trip check OK: reloaded map matches the fused one.")


def plot_outputs(grid_map):
    """
    Plot the fused map and save every figure in OUTPUT_DIR: elevation,
    top-down semantics, Matplotlib 3D and Plotly 3D (HTML).
    """

    section("Plots")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visualizer = GridMapVisualizer(grid_map)
    figures = {}

    fig, _ = visualizer.plot_elevation_heatmap()
    figures["elevation.png"] = fig

    fig, _ = visualizer.plot_semantic_topdown()
    figures["semantic_topdown.png"] = fig

    fig, _ = visualizer.plot_semantic_3d(max_points=PLOT_3D_MAX_POINTS)
    figures["semantic_3d.png"] = fig

    # Save before show(): closing the windows discards the figures.
    for filename, fig in figures.items():
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved {path}")

    fig_plotly = visualizer.plot_semantic_3d_plotly()
    html_path = OUTPUT_DIR / "semantic_3d.html"
    fig_plotly.write_html(html_path)
    print(f"Saved {html_path}")

    if SHOW_PLOTS:
        fig_plotly.show()
        plt.show()
    else:
        plt.close("all")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    perception_config = PerceptionConfig(
        PROJECT_ROOT / "config" / "perception_config.yaml"
    )
    grid_map_config = GridMapConfig(
        PROJECT_ROOT / "config" / "grid_map_config.yaml"
    )
    camera_calibration = CameraCalibrationConfig(
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    paired_by_camera, paired_frames = pair_all_cameras(camera_calibration)

    builder = PointCloudBuilder(perception_config, camera_calibration)
    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)

    section("Settings")
    print(
        f"Noise model: sigma^2 = {grid_map.noise_sigma0}^2 + "
        f"{grid_map.noise_alpha} * z^{grid_map.noise_exponent} "
        f"(z_ref = {grid_map.noise_reference_distance} m)"
    )
    print(f"Height datum mode: {grid_map.height_datum_mode}")

    if USE_KEYFRAMES:
        print(
            f"Keyframes (per camera): >= {KEYFRAME_MIN_TRANSLATION} m "
            f"or >= {KEYFRAME_MIN_ROTATION_DEG} deg"
        )
    else:
        print("Keyframes: disabled, every paired frame is fused.")

    sanity_check(paired_by_camera, builder, grid_map)

    trajectory = fuse(paired_frames, builder, grid_map)

    print_summary(grid_map)

    save_outputs(grid_map, trajectory, grid_map_config, class_reducer)

    plot_outputs(grid_map)


if __name__ == "__main__":
    main()