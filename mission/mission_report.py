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
    grid_map.npz           (written by the mission via GridMap.save())
    plots/cycle_NNN.png    (written by the mission, if enabled)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


CYCLE_COLUMNS = (
    "cycle", "wall_time", "exploration_time_s",
    "robot_x", "robot_y", "robot_yaw_deg", "battery_percent",
    "cameras_fused", "cameras_skipped", "coverage", "observed_cells",
    "plan_status", "frontier_x", "frontier_y", "waypoints",
    "nav_status", "nav_distance_to_goal_m", "nav_walked_m",
    "acquisition_s", "mapping_s", "planning_s", "navigation_s", "cycle_s",
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

        with open(self.cycles_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(CYCLE_COLUMNS)

        self.trajectory = []

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
