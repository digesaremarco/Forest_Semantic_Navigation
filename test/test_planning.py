"""
Test entry point for PlanningPipeline, running on top of the real
field dataset (same ingestion as test_costs.py) instead of a
synthetic scenario.

CAVEAT -- READ BEFORE INTERPRETING THE RESULT: this dataset is small
(14 frames covering a modest area) and NOT representative of a real
exploration mission -- most of the grid is still NaN/unknown, and
there's very little already-explored area to launch a frontier
search from. A None result from PlanningPipeline.plan() is a
legitimate outcome here (e.g. the nearest frontier collapses too
close to the robot, see pipeline/planning_pipeline.py's "CLUSTER
TARGET POINT" docstring), not necessarily a failure -- treat this as
"does the mechanism run end-to-end on real data without crashing and
produce something sane", not as a meaningful exploration decision.

Geometry: same setup as test_grid_map.py / test_costs.py.
FramePosePairer is given the real per-camera calibration
(config/camera_intrinsics.yaml, via CameraCalibrationConfig), so R,
t are the camera pose in the odom frame.

IMAGE ORIENTATION: same as test_grid_map.py -- files are used exactly
as spot_rgb_depth_log.py saves them (NATIVE raster), nothing is
rotated here. PointCloudBuilder turns only the RGB upright for
SegFormer and maps the probabilities back to native. Do NOT rotate
the files: a 180 deg rotation keeps the shape, so build() cannot
detect it, and the geometry would come out flipped.

HEIGHT DATUM: irrelevant here -- planning uses the cost layer and
(x, y) positions only; the costs are built from height differences.

Robot position used for planning: the robot BODY position at the
LAST frame ("where the robot ended up"), recovered from the camera
pose by removing the static body -> camera offset.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from perception.perception_config_loader import PerceptionConfig
from fusion.camera_calibration_loader import CameraCalibrationConfig
from fusion.frame_pose_pairing import FramePosePairer
from mapping.grid_map_config_loader import GridMapConfig
from costmap.costmap_config_loader import CostmapConfig
from planning.planning_config_loader import PlanningConfig
from planning.planning_visualizer import PlanningVisualizer
from pipeline.traversability_pipeline import TraversabilityPipeline
from pipeline.planning_pipeline import PlanningPipeline


def load_frame(rgb_path, depth_path):
    """
    Load RGB and depth exactly as Spot saved them (NATIVE raster),
    see the IMAGE ORIENTATION note at the top of this file.
    """

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))

    return rgb, depth


def camera_pose_to_body_position(R_odom_camera, t_odom_camera, R_body_camera, t_body_camera):
    """
    Recover the robot body position in the odom frame from the
    camera pose, inverting:

        R_odom_camera = R_odom_body @ R_body_camera
        t_odom_camera = R_odom_body @ t_body_camera + t_odom_body
    """

    R_odom_body = R_odom_camera @ R_body_camera.T
    t_odom_body = t_odom_camera - R_odom_body @ t_body_camera

    return t_odom_body


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

    planning_config = PlanningConfig(
        PROJECT_ROOT / "config" / "planning_config.yaml"
    )

    camera_calibration = CameraCalibrationConfig(
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    # Which camera this dataset's testset/images + testset/depths
    # came from -- change this if you test a different camera.
    camera_name = "frontleft"

    # -------------------------------------------------------------
    # Pair real frames with real poses (true camera pose), fuse
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

    traversability_pipeline = TraversabilityPipeline(
        perception_config, camera_calibration, grid_map_config, costmap_config
    )

    print(f"\nFusing {len(paired_frames)} real frame(s) into the grid map...\n")

    for frame in paired_frames:
        rgb, depth = load_frame(frame["rgb_path"], frame["depth_path"])
        traversability_pipeline.update(rgb, depth, frame["R"], frame["t"], camera_name)
        print(frame["rgb_path"].name)

    print()

    # -------------------------------------------------------------
    # Plan from where the robot ended up (body position, not camera)
    # -------------------------------------------------------------

    grid_map = traversability_pipeline.grid_map

    cost_layer = traversability_pipeline.get_cost_layer("total")

    last_frame = paired_frames[-1]

    R_body_camera, t_body_camera = camera_calibration.get_extrinsics(camera_name)

    last_body_t = camera_pose_to_body_position(
        last_frame["R"], last_frame["t"], R_body_camera, t_body_camera
    )

    robot_position_world = (float(last_body_t[0]), float(last_body_t[1]))

    print(
        "Robot position (world, body at the last frame):",
        robot_position_world
    )

    planning_pipeline = PlanningPipeline(planning_config)

    result = planning_pipeline.plan(
        cost_layer, robot_position_world,
        grid_map.resolution, grid_map.get_origin()[:2], grid_map.cell_n
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    detection = planning_pipeline.last_detection

    print()
    print("Frontier clusters found:", len(detection["clusters"]))

    for cluster in detection["clusters"]:
        print(f"  size={cluster['size']}, centroid={cluster['centroid']}")

    if result is None:
        print()
        print(
            "plan() returned None -- see the frontier count above for "
            "whether any frontier was found at all, and this script's "
            "module docstring for why a found frontier can still lead "
            "to no plan (this is a small, non-representative dataset)."
        )
    else:
        frontier = result["frontier"]

        print()
        print("Selected frontier:")
        print("  size:", frontier["size"])
        print("  information_gain:", frontier["information_gain"])
        print("  navigation_cost:", frontier["navigation_cost"])
        print("  utility:", frontier["utility"])
        print("  refined_centroid (cell):", frontier["refined_centroid"])
        print()
        print("Path length (cells, after smoothing):", len(result["path_cells"]))
        print("Number of waypoints:", len(result["waypoints"]))

        for wp in result["waypoints"]:
            print(f"  position={wp['position']}, heading={wp['heading']:.3f} rad")

    # -------------------------------------------------------------
    # Visualize -- just show it, nothing saved
    # -------------------------------------------------------------

    visualizer = PlanningVisualizer(grid_map)

    visualizer.plot(
        cost_layer, detection, result, robot_position_world,
        title="Planning test: C_total + frontiers + selected path",
        show=True
    )