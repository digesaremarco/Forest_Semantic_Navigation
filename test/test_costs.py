"""
Throwaway diagnostic script: fuse the real field dataset (paired
RGB/depth/pose frames) into GridMap, compute GeometricCost and
SemanticCost, fuse them with CostFusion, and visualize everything.

Purpose: eyeball whether the current thresholds (costmap_config.yaml)
and per-class costs (class_costs.yaml) look reasonable on real
terrain, and how much CostFusion's per-cell fallback actually kicks
in in practice. Not meant to be kept around -- delete once you've
looked at the plots.

Geometry: same setup as test_grid_map.py. FramePosePairer is given
the real per-camera calibration (config/camera_intrinsics.yaml, via
CameraCalibrationConfig), so R, t are the camera pose in the odom
frame.

IMAGE ORIENTATION: both RGB and depth for THIS testset are rotated
180 degrees in load_frame() -- see the same note in test_grid_map.py.
They currently look upright (display orientation) rather than native,
and PointCloudBuilder.build() pairs each depth pixel's semantic label
using the SAME (row, col) index on both arrays with no internal
rotation, so RGB and depth must stay pixel-aligned with each other
AND match the native orientation the extrinsics assume.
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
from costmap.costmap_config_loader import CostmapConfig
from costmap.geometric_cost import GeometricCost
from costmap.semantic_cost import SemanticCost
from costmap.cost_fusion import CostFusion


def load_frame(rgb_path, depth_path):
    """
    Undo the 180-degree rotation on BOTH RGB and depth for THIS
    testset (see the IMAGE ORIENTATION note at the top of this file).
    """

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))
    rgb = np.rot90(rgb, k=2)
    depth = np.rot90(depth, k=2)

    return rgb, depth


def get_field(pc, key):
    if key not in pc:
        raise KeyError(
            f"PointCloudBuilder.build() output has no '{key}' key. "
            f"Available keys: {list(pc.keys())}"
        )

    return pc[key]


def plot_layer(visualizer, layer, ax, title, cmap="viridis", vmin=None, vmax=None):
    """
    Crop `layer` to the grid's observed bounding box (reusing
    GridMapVisualizer's own crop/extent logic, so this matches
    plot_elevation_heatmap exactly) and draw it as a heatmap.
    """

    row_slice, col_slice = visualizer.resolve_crop(crop_to_observed=True)
    view = layer[row_slice, col_slice]
    extent = visualizer.compute_extent(row_slice, col_slice)

    im = ax.imshow(
        np.ma.masked_invalid(view),
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.colorbar(im, ax=ax, fraction=0.046)


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

    costmap_config = CostmapConfig(
        PROJECT_ROOT / "config" / "costmap_config.yaml",
        PROJECT_ROOT / "config" / "class_costs.yaml"
    )

    camera_calibration = CameraCalibrationConfig(
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    # Which camera this dataset's testset/images + testset/depths
    # came from -- change this if you test a different camera.
    camera_name = "right"

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

    # -------------------------------------------------------------
    # Build the pipeline and fuse every frame into the grid
    # -------------------------------------------------------------

    builder = PointCloudBuilder(perception_config, camera_calibration)
    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)

    print(f"\nFusing {len(paired_frames)} real frame(s) into the grid map...\n")

    for frame in paired_frames:
        rgb, depth = load_frame(frame["rgb_path"], frame["depth_path"])

        pc = builder.build(rgb, depth, camera_name)

        points_xyz = get_field(pc, "points_xyz").astype(np.float32, copy=False)
        semantic_colors = get_field(pc, "semantic_colors").astype(np.uint8, copy=False)
        semantic_probs = get_field(pc, "semantic_probs").astype(np.float32, copy=False)

        grid_map.update(
            points_xyz, semantic_colors, semantic_probs,
            frame["R"], frame["t"]
        )

        print(f"{frame['rgb_path'].name}: {points_xyz.shape[0]} points")

    print()

    # -------------------------------------------------------------
    # Compute both costs, then fuse them
    # -------------------------------------------------------------

    elevation = grid_map.get_elevation_layer()
    semantic_probs_layer = grid_map.get_semantic_probs_layer()
    count = grid_map.get_count_layer()

    geometric_cost = GeometricCost(costmap_config)
    geo_result = geometric_cost.compute(elevation, resolution=grid_map.resolution)

    semantic_cost = SemanticCost(costmap_config, class_reducer)
    sem_cost = semantic_cost.compute(semantic_probs_layer)

    cost_fusion = CostFusion(costmap_config)
    fused_cost = cost_fusion.compute(geo_result["cost"], sem_cost)

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    geo_valid = np.isfinite(geo_result["cost"])
    sem_valid = np.isfinite(sem_cost)

    both = int(np.sum(geo_valid & sem_valid))
    only_geo = int(np.sum(geo_valid & ~sem_valid))
    only_sem = int(np.sum(sem_valid & ~geo_valid))
    neither = int(np.sum(~geo_valid & ~sem_valid))

    print("Elevation observed cells:", np.isfinite(elevation).sum())
    print("C_geo observed cells:    ", int(geo_valid.sum()))
    print("C_sem observed cells:    ", int(sem_valid.sum()))
    print()
    print("Fusion breakdown:")
    print(f"  both available (weighted avg): {both}")
    print(f"  only C_geo (fallback):         {only_geo}")
    print(f"  only C_sem (fallback):         {only_sem}")
    print(f"  neither (NaN):                 {neither}")
    print()
    print("C_geo range:  ", np.nanmin(geo_result["cost"]), "-", np.nanmax(geo_result["cost"]))
    print("C_sem range:  ", np.nanmin(sem_cost), "-", np.nanmax(sem_cost))
    print("C_total range:", np.nanmin(fused_cost), "-", np.nanmax(fused_cost))

    # -------------------------------------------------------------
    # Visualize: elevation, C_geo, C_sem, C_total / slope, roughness,
    # step, observation count
    # -------------------------------------------------------------

    visualizer = GridMapVisualizer(grid_map)

    fig, axes = plt.subplots(2, 4, figsize=(25, 11))

    visualizer.plot_elevation_heatmap(ax=axes[0, 0])

    plot_layer(
        visualizer, geo_result["cost"], axes[0, 1],
        "Geometric cost (C_geo)", cmap="inferno", vmin=0, vmax=1
    )

    plot_layer(
        visualizer, sem_cost, axes[0, 2],
        "Semantic cost (C_sem)", cmap="inferno", vmin=0, vmax=1
    )

    plot_layer(
        visualizer, fused_cost, axes[0, 3],
        "Fused cost (C_total)", cmap="inferno", vmin=0, vmax=1
    )

    plot_layer(
        visualizer, geo_result["slope_cost"], axes[1, 0],
        "Slope cost", cmap="inferno", vmin=0, vmax=1
    )

    plot_layer(
        visualizer, geo_result["roughness_cost"], axes[1, 1],
        "Roughness cost", cmap="inferno", vmin=0, vmax=1
    )

    plot_layer(
        visualizer, geo_result["step_cost"], axes[1, 2],
        "Step cost", cmap="inferno", vmin=0, vmax=1
    )

    # Unobserved cells (count == 0) shown blank, like the other layers.
    plot_layer(
        visualizer, np.where(count > 0, count, np.nan), axes[1, 3],
        "Observation count", cmap="magma"
    )

    plt.tight_layout()
    plt.show()