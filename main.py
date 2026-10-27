"""
Forest Semantic Navigation -- autonomous exploration mission on Spot.

Run on the CORE I/O (Jetson). Wires the real components and runs the
ExplorationMission state machine (mission/exploration_mission.py):
acquisition -> semantic mapping -> frontier planning -> navigation,
stop-and-go, until a stop criterion is met (config/mission_config.yaml),
then back to the start position.

BEFORE RUNNING
    - Operator with the E-stop on the tablet, robot in the area to
      explore (the grid is anchored on the start pose: grid_map_config
      map_length around it).
    - First run in a new place: PLAN_ONLY = True. The robot does not
      move (no lease): one acquisition + map + plan, plot and summary.
      Check plots/cycle_001.png, then set PLAN_ONLY = False.

OUTPUTS: one folder per run under acquisition_config.yaml
logging.output_root, named <run_prefix>_<date>_<time>:
    mission.log, mission_summary.json (timing, end reason, counters),
    cycles.csv (one row per cycle, with the time of each phase),
    path.csv + planned_paths.csv (walked path at ~10 Hz, A* plans),
    mission_path.png (walked path over the final map),
    grid_map.npz + trajectory.npz (loadable by test_costs.py /
    test_planning.py), plots/cycle_NNN.png, and the raw snapshots
    (images_rgb/, depth_data/, odometry/, poses/).

TIMING: mission_start is marked right before connecting to the robot,
after the models are loaded -- loading SegFormer/TensorRT is setup,
not part of the experiment. See mission/mission_timer.py for the
intervals reported.

Ctrl+C stops the robot, sits it down, powers it off and saves
everything.
"""

import sys
from pathlib import Path

import matplotlib

# Headless on the Jetson: render plots to files only.
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))

from spot.spot_config_loader import SpotConfig
from spot.spot_interface import SpotInterface
from acquisition.acquisition_config_loader import AcquisitionConfig
from acquisition.image_acquirer import ImageAcquirer
from acquisition.observation_logger import ObservationLogger
from perception.perception_config_loader import PerceptionConfig
from fusion.camera_calibration_loader import CameraCalibrationConfig
from mapping.grid_map_config_loader import GridMapConfig
from costmap.costmap_config_loader import CostmapConfig
from planning.planning_config_loader import PlanningConfig
from planning.planning_visualizer import PlanningVisualizer
from navigation.navigation_config_loader import NavigationConfig
from navigation.waypoint_follower import WaypointFollower
from pipeline.traversability_pipeline import TraversabilityPipeline
from pipeline.planning_pipeline import PlanningPipeline
from mission.mission_config_loader import MissionConfig
from mission.mission_report import MissionReport
from mission.mission_timer import MissionTimer
from mission.mission_plotter import MissionPlotter
from mission.exploration_mission import ExplorationMission


# True: no lease, the robot does not move -- one acquisition + map +
# plan cycle. Use it for the first run in a new place.
PLAN_ONLY = False

CONFIG_DIR = PROJECT_ROOT / "config"


def load_configs():
    acquisition_config = AcquisitionConfig(CONFIG_DIR / "acquisition_config.yaml")

    return {
        "spot": SpotConfig(CONFIG_DIR / "spot_config.yaml"),
        "acquisition": acquisition_config,
        "perception": PerceptionConfig(CONFIG_DIR / "perception_config.yaml"),
        "calibration": CameraCalibrationConfig(
            CONFIG_DIR / "camera_intrinsics.yaml",
            camera_names=acquisition_config.camera_names
        ),
        "grid_map": GridMapConfig(CONFIG_DIR / "grid_map_config.yaml"),
        "costmap": CostmapConfig(
            CONFIG_DIR / "costmap_config.yaml",
            CONFIG_DIR / "class_costs.yaml"
        ),
        "planning": PlanningConfig(CONFIG_DIR / "planning_config.yaml"),
        "navigation": NavigationConfig(CONFIG_DIR / "navigation_config.yaml"),
        "mission": MissionConfig(CONFIG_DIR / "mission_config.yaml"),
    }


def make_cycle_plotter(grid_map):
    visualizer = PlanningVisualizer(grid_map)

    def plot(save_path, cost_layer, detection, result, robot_xy, title):
        visualizer.plot(
            cost_layer, detection, result, robot_xy,
            title=title, save_path=save_path, show=False
        )

    return plot


def main():
    configs = load_configs()

    run_dir = ObservationLogger.create_run_dir(
        configs["acquisition"].output_root,
        prefix=configs["mission"].run_prefix + ("_planonly" if PLAN_ONLY else "")
    )

    report = MissionReport(run_dir)
    report.start_console_log()

    try:
        print(f"Run folder: {run_dir}")
        print(f"Mode: {'PLAN ONLY (robot will not move)' if PLAN_ONLY else 'FULL MISSION'}")

        # Models and pipelines first: loading is setup, not experiment.
        traversability = TraversabilityPipeline(
            configs["perception"], configs["calibration"],
            configs["grid_map"], configs["costmap"]
        )
        planner = PlanningPipeline(configs["planning"])

        timer = MissionTimer()
        timer.mark("mission_start")

        with SpotInterface(configs["spot"]) as spot:
            acquirer = ImageAcquirer(
                configs["acquisition"], configs["calibration"], spot.image_client
            )

            observation_logger = (
                ObservationLogger(configs["acquisition"], run_dir)
                if configs["acquisition"].save_to_disk else None
            )

            mission = ExplorationMission(
                mission_config=configs["mission"],
                planning_config=configs["planning"],
                spot=spot,
                acquirer=acquirer,
                traversability=traversability,
                planner=planner,
                follower=WaypointFollower(configs["navigation"], spot),
                report=report,
                timer=timer,
                observation_logger=observation_logger,
                cycle_plotter=make_cycle_plotter(traversability.grid_map),
                mission_plotter=MissionPlotter()
            )

            mission.run(plan_only=PLAN_ONLY)

    finally:
        report.stop_console_log()


if __name__ == "__main__":
    main()