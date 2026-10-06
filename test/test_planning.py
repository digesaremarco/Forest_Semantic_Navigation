"""
Run PlanningPipeline on a saved GridMap and plot the result.

Loads the map and trajectory written by test_grid_map.py (no pairing,
no SegFormer, no fusion), computes C_total exactly as
TraversabilityPipeline.get_cost_layer("total") does (GeometricCost +
SemanticCost + CostFusion), plans from where the robot ended up, and
saves the overlay (cost + frontiers + selected path + waypoints).

CAVEAT -- READ BEFORE INTERPRETING THE RESULT: a single recorded run
is NOT a real exploration mission -- most of the grid is unknown and
there is little explored area to launch a frontier search from. A
None result from PlanningPipeline.plan() is a legitimate outcome
(e.g. the nearest frontier collapses too close to the robot, see
pipeline/planning_pipeline.py's "CLUSTER TARGET POINT" docstring).
Treat this as "does the mechanism run end-to-end on real data and
produce something sane", not as a meaningful exploration decision.

ROBOT POSITION: the robot BODY position at the LAST frame of the run
("where the robot ended up"), recovered from that frame's camera pose
in trajectory.npz by removing the static body -> camera offset OF ITS
CAMERA. As a cross-check, the body position from the last frame of
each camera is printed too: with correct extrinsics they agree to
within how far the robot moved between those frames.

HEIGHT DATUM: irrelevant here -- planning uses the cost layer and
(x, y) positions only; the costs are built from height differences.

Because nothing is rebuilt, rerun it freely after changing
planning_config.yaml, costmap_config.yaml or class_costs.yaml. If
grid_map_config.yaml or the retained classes changed, GridMap.load()
refuses the file: rebuild the map with test_grid_map.py first.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from perception.perception_config_loader import PerceptionConfig
from perception.class_reducer import ClassReducer
from fusion.camera_calibration_loader import CameraCalibrationConfig
from mapping.grid_map_config_loader import GridMapConfig
from mapping.grid_map import GridMap
from costmap.costmap_config_loader import CostmapConfig
from costmap.geometric_cost import GeometricCost
from costmap.semantic_cost import SemanticCost
from costmap.cost_fusion import CostFusion
from planning.planning_config_loader import PlanningConfig
from planning.planning_visualizer import PlanningVisualizer
from pipeline.planning_pipeline import PlanningPipeline


# Map and trajectory written by test_grid_map.py; the plot is saved
# next to them.
OUTPUT_DIR = PROJECT_ROOT / "oggi2" / "output"
MAP_PATH = OUTPUT_DIR / "grid_map.npz"
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory.npz"

# Whether to also open the plot on screen (it is always saved).
SHOW_PLOTS = True


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def section(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def load_trajectory(path):
    """
    trajectory.npz as a list of dicts (timestamp, camera_name, R, t,
    fused), in fusion order.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory not found: {path}. Run test_grid_map.py first."
        )

    data = np.load(path, allow_pickle=False)

    return [
        {
            "timestamp": float(data["timestamp"][i]),
            "camera_name": str(data["camera_name"][i]),
            "R": data["R"][i],
            "t": data["t"][i],
            "fused": bool(data["fused"][i]),
        }
        for i in range(data["timestamp"].shape[0])
    ]


def body_position(frame, camera_calibration):
    """
    Robot body position in odom at a trajectory frame, inverting

        R_odom_camera = R_odom_body @ R_body_camera
        t_odom_camera = R_odom_body @ t_body_camera + t_odom_body

    with the extrinsics of the frame's OWN camera.
    """

    R_body_camera, t_body_camera = camera_calibration.get_extrinsics(
        frame["camera_name"]
    )

    R_odom_body = frame["R"] @ np.asarray(R_body_camera).T

    return frame["t"] - R_odom_body @ np.asarray(t_body_camera)


def compute_total_cost(grid_map, costmap_config, class_reducer):
    """C_total, same composition as TraversabilityPipeline."""

    geo_cost = GeometricCost(costmap_config).compute(
        grid_map.get_elevation_layer(), resolution=grid_map.resolution
    )["cost"]

    sem_cost = SemanticCost(costmap_config, class_reducer).compute(
        grid_map.get_semantic_probs_layer()
    )

    return CostFusion(costmap_config).compute(geo_cost, sem_cost)


def robot_position_from_trajectory(trajectory, camera_calibration):
    """
    (x, y) of the body at the last frame of the run, with the
    per-camera cross-check printed.
    """

    section("Robot position")

    last = trajectory[-1]
    last_body = body_position(last, camera_calibration)

    print(
        f"Body at the last frame (from '{last['camera_name']}'): "
        f"({last_body[0]:+.3f}, {last_body[1]:+.3f})"
    )

    # Cross-check: a camera far off from the others points at a wrong
    # extrinsic.
    print("Body at the last frame of each camera (cross-check):")

    last_per_camera = {}

    for frame in trajectory:
        last_per_camera[frame["camera_name"]] = frame

    for camera_name, frame in last_per_camera.items():
        body = body_position(frame, camera_calibration)
        offset = np.linalg.norm(body[:2] - last_body[:2])

        print(
            f"  {camera_name:<12} ({body[0]:+.3f}, {body[1]:+.3f}) "
            f"-> {offset * 100:.1f} cm from the planning position"
        )

    return float(last_body[0]), float(last_body[1])


def print_planning_result(detection, result):
    """Frontiers found and, if any, the selected one and its path."""

    section("Planning result")

    print("Frontier clusters found:", len(detection["clusters"]))

    for cluster in detection["clusters"]:
        print(f"  size={cluster['size']}, centroid={cluster['centroid']}")

    if result is None:
        print(
            "\nplan() returned None -- see the frontier count above for "
            "whether any frontier was found at all, and the module "
            "docstring for why a found frontier can still lead to no "
            "plan on a single recorded run."
        )
        return

    frontier = result["frontier"]

    print("\nSelected frontier:")
    print("  size:", frontier["size"])
    print("  information_gain:", frontier["information_gain"])
    print("  navigation_cost:", frontier["navigation_cost"])
    print("  utility:", frontier["utility"])
    print("  refined_centroid (cell):", frontier["refined_centroid"])

    print("\nPath length (cells, after smoothing):", len(result["path_cells"]))
    print("Number of waypoints:", len(result["waypoints"]))

    for wp in result["waypoints"]:
        print(f"  position={wp['position']}, heading={wp['heading']:.3f} rad")


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

    # -------------------------------------------------------------
    # Load map and trajectory
    # -------------------------------------------------------------

    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)
    grid_map.load(MAP_PATH)

    trajectory = load_trajectory(TRAJECTORY_PATH)

    cameras = sorted({frame["camera_name"] for frame in trajectory})
    print(f"Trajectory: {len(trajectory)} frame(s), cameras: {', '.join(cameras)}")

    # -------------------------------------------------------------
    # Cost, robot position, plan
    # -------------------------------------------------------------

    cost_layer = compute_total_cost(grid_map, costmap_config, class_reducer)

    robot_position_world = robot_position_from_trajectory(
        trajectory, camera_calibration
    )

    planning_pipeline = PlanningPipeline(planning_config)

    result = planning_pipeline.plan(
        cost_layer, robot_position_world,
        grid_map.resolution, grid_map.get_origin()[:2], grid_map.cell_n
    )

    detection = planning_pipeline.last_detection

    print_planning_result(detection, result)

    # -------------------------------------------------------------
    # Plot (saved, optionally shown)
    # -------------------------------------------------------------

    section("Plots")

    plot_path = OUTPUT_DIR / "planning.png"

    PlanningVisualizer(grid_map).plot(
        cost_layer, detection, result, robot_position_world,
        title=(
            "Planning: C_total + frontiers + selected path "
            f"({', '.join(cameras)})"
        ),
        save_path=plot_path,
        show=SHOW_PLOTS
    )

    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()