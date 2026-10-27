"""
Everything a mission writes about itself, in its run folder (the same
folder ObservationLogger writes the snapshots to):

    mission.log            full console output of the run (stdout and
                           stderr are duplicated into it by Tee)
    cycles.csv             one row per exploration cycle, written as
                           the mission goes (a crash loses nothing)
    mission_summary.json   end reason, timing, counters, distances,
                           coverage, battery, poses, configuration
    trajectory.npz         every camera pose seen, in the format of
                           test_grid_map.py (timestamp, camera_name, R,
                           t, fused) -- test_planning.py / test_costs.py
                           can load it next to grid_map.npz
    path.csv               the path actually walked: body pose polled
                           by WaypointFollower (~10 Hz) plus a row at
                           every capture -- t_s (since mission start),
                           cycle, phase (exploration | return_home),
                           event (walk | stop_capture | moving_capture),
                           x, y, yaw_deg in the vision frame
    planned_paths.csv      every planned path (A* waypoints): cycle,
                           phase, attempt, index, x, y
    grid_map.npz           (written by the mission via GridMap.save())
    plots/cycle_NNN.png    (written by the mission, if enabled)
    mission_path.png       (MissionPlotter: walked path over the map)

path.csv and planned_paths.csv are appended at the end of every cycle
(flush_paths()), so a crash loses at most the current cycle.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


PATH_COLUMNS = ("t_s", "cycle", "phase", "event", "x", "y", "yaw_deg")

PLANNED_COLUMNS = ("cycle", "phase", "attempt", "index", "x", "y")

CYCLE_COLUMNS = (
    "cycle", "wall_time", "exploration_time_s",
    "robot_x", "robot_y", "robot_yaw_deg", "battery_percent",
    "moving_fused", "cameras_fused", "cameras_skipped", "coverage", "observed_cells",
    "plan_status", "frontier_x", "frontier_y", "waypoints",
    "nav_status", "nav_distance_to_goal_m", "nav_walked_m",
    "moving_captured", "acquisition_s", "moving_acquisition_s",
    "mapping_s", "planning_s", "navigation_s", "cycle_s",
    "note",
)


class Tee:
    """Write to several streams at once (console + mission.log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


class MissionReport:

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.cycles_path = self.run_dir / "cycles.csv"
        self.summary_path = self.run_dir / "mission_summary.json"
        self.trajectory_path = self.run_dir / "trajectory.npz"
        self.log_path = self.run_dir / "mission.log"
        self.plots_dir = self.run_dir / "plots"
        self.path_path = self.run_dir / "path.csv"
        self.planned_path = self.run_dir / "planned_paths.csv"
        self.mission_plot_path = self.run_dir / "mission_path.png"

        for path, columns in (
            (self.cycles_path, CYCLE_COLUMNS),
            (self.path_path, PATH_COLUMNS),
            (self.planned_path, PLANNED_COLUMNS),
        ):
            with open(path, "w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(columns)

        self.trajectory = []

        # Walked path and planned paths: kept in memory (for the final
        # plot) and appended to their CSV by flush_paths().
        self.path_rows = []
        self.planned_paths = []
        self.path_rows_written = 0
        self.planned_paths_written = 0

        self.log_file = None
        self.original_streams = None

    # -----------------------------------------------------------------
    # Console log
    # -----------------------------------------------------------------

    def start_console_log(self):
        """Duplicate stdout/stderr into mission.log until stop_console_log()."""

        if self.log_file is not None:
            return

        self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self.original_streams = (sys.stdout, sys.stderr)

        sys.stdout = Tee(sys.stdout, self.log_file)
        sys.stderr = Tee(sys.stderr, self.log_file)

    def stop_console_log(self):
        if self.log_file is None:
            return

        sys.stdout, sys.stderr = self.original_streams
        self.log_file.close()
        self.log_file = None

    # -----------------------------------------------------------------
    # Records
    # -----------------------------------------------------------------

    def write_cycle(self, record):
        row = []

        for column in CYCLE_COLUMNS:
            value = record.get(column)

            if value is None:
                row.append("")
            elif isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(value)

        with open(self.cycles_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)

    def add_path_pose(self, t_s, cycle, phase, event, pose):
        """One walked-path row; pose = (x, y, yaw [rad]) in vision."""

        self.path_rows.append({
            "t_s": float(t_s),
            "cycle": int(cycle),
            "phase": phase,
            "event": event,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "yaw_deg": float(np.degrees(pose[2])),
        })

    def add_planned_path(self, cycle, phase, attempt, waypoints):
        """The waypoints of one plan (PlanningPipeline output)."""

        self.planned_paths.append({
            "cycle": int(cycle),
            "phase": phase,
            "attempt": int(attempt),
            "points": [tuple(float(v) for v in waypoint["position"][:2]) for waypoint in waypoints],
        })

    def flush_paths(self):
        """Append the rows added since the last flush to the CSVs."""

        new_rows = self.path_rows[self.path_rows_written:]

        if new_rows:
            with open(self.path_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                for row in new_rows:
                    writer.writerow([
                        f"{row['t_s']:.3f}", row["cycle"], row["phase"], row["event"],
                        f"{row['x']:.4f}", f"{row['y']:.4f}", f"{row['yaw_deg']:.2f}",
                    ])

            self.path_rows_written = len(self.path_rows)

        new_plans = self.planned_paths[self.planned_paths_written:]

        if new_plans:
            with open(self.planned_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                for plan in new_plans:
                    for index, (x, y) in enumerate(plan["points"]):
                        writer.writerow([
                            plan["cycle"], plan["phase"], plan["attempt"], index,
                            f"{x:.4f}", f"{y:.4f}",
                        ])

            self.planned_paths_written = len(self.planned_paths)

    def add_camera_pose(self, timestamp, camera_name, R, t, fused):
        self.trajectory.append({
            "timestamp": float(timestamp),
            "camera_name": camera_name,
            "R": np.asarray(R, dtype=np.float64),
            "t": np.asarray(t, dtype=np.float64),
            "fused": bool(fused),
        })

    def save_trajectory(self):
        if not self.trajectory:
            return None

        np.savez_compressed(
            self.trajectory_path,
            timestamp=np.array([r["timestamp"] for r in self.trajectory], dtype=np.float64),
            camera_name=np.array([r["camera_name"] for r in self.trajectory]),
            R=np.stack([r["R"] for r in self.trajectory]),
            t=np.stack([r["t"] for r in self.trajectory]),
            fused=np.array([r["fused"] for r in self.trajectory], dtype=bool),
        )

        return self.trajectory_path

    def save_summary(self, summary):
        with open(self.summary_path, "w", encoding="utf-8") as file:
            json.dump(self.to_jsonable(summary), file, indent=2)

        return self.summary_path

    def cycle_plot_path(self, cycle):
        return self.plots_dir / f"cycle_{cycle:03d}.png"

    @classmethod
    def to_jsonable(cls, value):
        if isinstance(value, dict):
            return {str(key): cls.to_jsonable(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [cls.to_jsonable(item) for item in value]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, float) and not np.isfinite(value):
            return None

        if isinstance(value, Path):
            return str(value)

        return value