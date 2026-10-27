"""
Autonomous exploration mission: the state machine that ties
acquisition, mapping, planning and navigation together, stop-and-go.

    INIT ──> ACQUIRE ──> MAP_UPDATE ──> CHECK ──> PLAN ──> NAVIGATE ──┐
               ^                          │         │                  │
               └──────────────────────────┼─────────┼──────────────────┘
                                          v         v
                                   RETURN_HOME ──> SHUTDOWN ──> DONE

    INIT         E-stop check, lease (take), power on, stand; acquirer
                 setup + calibration check; home pose = body pose now.
    ACQUIRE      one synchronized capture of all cameras (+ saved to
                 disk by ObservationLogger).
    MAP_UPDATE   each camera fused into the GridMap if it is a keyframe
                 (moved/turned enough since its last fused frame);
                 coverage recomputed.
    CHECK        stop criteria (coverage OR time, battery, consecutive
                 navigation/acquisition failures): the first one met
                 ends the exploration -> RETURN_HOME.
    PLAN         next frontier on C_total. If A* fails on the tried
                 frontiers they are blacklisted and planning is
                 repeated (max MAX_REPLANS_PER_CYCLE). No frontier left
                 -> exploration complete; frontiers left but none
                 reachable -> no_reachable_frontier. Both -> RETURN_HOME.
    NAVIGATE     WaypointFollower walks the whole path, stopping only
                 at the frontier. With moving_capture enabled, images
                 are also captured on the way (see MOVING CAPTURE). Success -> ACQUIRE. Timeout / stalled
                 / blocked -> target blacklisted, failure counted,
                 ACQUIRE where the robot stopped. Fault (fall) or
                 command overridden (operator took control) -> the
                 mission is aborted: SHUTDOWN, no return home.
    RETURN_HOME  plan_to_goal(home) on the current map and follow it;
                 up to return_home.max_attempts, re-acquiring and
                 replanning from wherever the previous attempt stopped.
                 If it fails, the robot sits where it is.
    SHUTDOWN     sit, power off, return the lease.

MOVING CAPTURE (mission_config.yaml, moving_capture): during NAVIGATE
the follower calls on_waypoint_passed() each time Spot passes an
intermediate waypoint, right after the next waypoint's command is
out -- Spot keeps walking the same A* path. If the robot is at least
min_distance_m from the last capture, ONE capture attempt is made
(get_image + decoding only, no settle, no retry) and the snapshot is
buffered. The buffered snapshots are saved to disk and fused at the
next stop, in MAP_UPDATE, before the stop snapshot -- so planning
sees the whole stretch walked. Captures do not run during the return
home. Snapshots still buffered when the mission ends (fault, Ctrl+C)
are saved to disk, not fused.

WALKED PATH: the follower reports every body pose it polls (~10 Hz)
through on_pose(); each one is recorded with its cycle and phase
(exploration / return_home), plus a row at every capture. Together
with the planned paths they go to path.csv / planned_paths.csv
(MissionReport) and, at the end, to mission_path.png (MissionPlotter).

Ctrl+C at any time stops the robot, then the normal shutdown and
saving run. Any exception does the same and is re-raised afterwards.
Whatever happens, the run folder gets the map, the trajectory, the
cycle table and the summary with the timing (MissionReport).

PLAN_ONLY (run(plan_only=True)): no lease, the robot never moves --
INIT without power, one ACQUIRE / MAP_UPDATE / PLAN cycle, plot and
summary. The safe first test of the whole chain in the field.

All components are passed in (dependency injection): main.py wires the
real ones, test/test_mission_sim.py a simulated robot and world.
"""

import math
import time
import traceback

import numpy as np

from acquisition.image_acquirer import AcquisitionError
from spot.frame_utils import yaw_from_rotation
from mission.stop_criteria import ExplorationCoverage, StopCriteria
from navigation.nav_result import NavResult
from pipeline.planning_pipeline import PlanningPipeline


# Replanning within one cycle after A* failed on every tried frontier.
MAX_REPLANS_PER_CYCLE = 5


class MissionState:
    INIT = "INIT"
    ACQUIRE = "ACQUIRE"
    MAP_UPDATE = "MAP_UPDATE"
    CHECK = "CHECK"
    PLAN = "PLAN"
    NAVIGATE = "NAVIGATE"
    RETURN_HOME = "RETURN_HOME"
    SHUTDOWN = "SHUTDOWN"
    DONE = "DONE"


class EndReason:
    EXPLORATION_COMPLETE = "exploration_complete"
    NO_REACHABLE_FRONTIER = "no_reachable_frontier"
    ROBOT_FAULT = "robot_fault"
    OPERATOR_OVERRIDE = "operator_override"
    OPERATOR_INTERRUPT = "operator_interrupt"
    ERROR = "error"
    PLAN_ONLY = "plan_only_completed"
    # + the StopCriteria reasons (coverage_reached, max_exploration_time, ...)


class ExplorationMission:

    def __init__(
        self,
        mission_config,
        planning_config,
        spot,
        acquirer,
        traversability,
        planner,
        follower,
        report,
        timer,
        observation_logger=None,
        cycle_plotter=None,
        mission_plotter=None
    ):
        """
        Parameters
        ----------
        mission_config : MissionConfig
        planning_config : PlanningConfig (border_margin_cells, for coverage)
        spot : SpotInterface
        acquirer : ImageAcquirer
        traversability : TraversabilityPipeline
        planner : PlanningPipeline
        follower : WaypointFollower
        report : MissionReport
        timer : MissionTimer
        observation_logger : ObservationLogger or None
            None: snapshots are not saved to disk.
        cycle_plotter : callable or None
            cycle_plotter(save_path, cost_layer, detection, result,
            robot_xy, title) -- None: no per-cycle plots.
        mission_plotter : MissionPlotter or None
            End-of-mission overview (mission_path.png) -- None: none.
        """

        self.config = mission_config
        self.spot = spot
        self.acquirer = acquirer
        self.traversability = traversability
        self.planner = planner
        self.follower = follower
        self.report = report
        self.timer = timer
        self.observation_logger = observation_logger
        self.cycle_plotter = cycle_plotter if mission_config.save_cycle_plots else None
        self.mission_plotter = mission_plotter if mission_config.save_mission_plot else None

        self.coverage = ExplorationCoverage(planning_config.border_margin_cells)
        self.stop_criteria = StopCriteria(mission_config)

        # Moving capture: the follower reports every intermediate
        # waypoint it passes.
        self.moving_buffer = []
        self.last_capture_xy = None

        if mission_config.moving_capture_enabled and follower is not None:
            follower.on_waypoint_passed = self.on_waypoint_passed

        # Walked path: every pose the follower polls.
        if follower is not None:
            follower.on_pose = self.on_robot_pose

        self.handlers = {
            MissionState.INIT: self.state_init,
            MissionState.ACQUIRE: self.state_acquire,
            MissionState.MAP_UPDATE: self.state_map_update,
            MissionState.CHECK: self.state_check,
            MissionState.PLAN: self.state_plan,
            MissionState.NAVIGATE: self.state_navigate,
            MissionState.RETURN_HOME: self.state_return_home,
            MissionState.SHUTDOWN: self.state_shutdown,
        }

        self.plan_only = False
        self.state = None

        # Mission state
        self.home_pose = None
        self.cycle_index = 0
        self.cycle = None
        self.snapshot = None
        self.plan_result = None
        self.last_fused_pose = {}
        self.last_coverage = None

        self.end_reason = None
        self.end_detail = None
        self.error = None
        self.shutdown_done = False

        # Counters
        self.consecutive_nav_failures = 0
        self.consecutive_acquisition_failures = 0
        self.stats = {
            "snapshots": 0,
            "acquisition_failures": 0,
            "frames_fused": 0,
            "frames_skipped": 0,
            "moving_snapshots": 0,
            "moving_capture_failures": 0,
            "nav_succeeded": 0,
            "nav_failed": {},
            "walked_exploration_m": 0.0,
            "walked_return_m": 0.0,
            "battery_start": None,
            "battery_end": None,
        }
        self.return_home = {
            "attempted": False,
            "reached": False,
            "attempts": 0,
            "final_distance_m": None,
            "detail": None,
        }

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def run(self, plan_only=False):
        """
        Run the mission to completion.

        Returns
        -------
        dict, the mission summary (also saved as mission_summary.json).
        """

        self.plan_only = plan_only

        if not self.timer.has_mark("mission_start"):
            self.timer.mark("mission_start")

        state = MissionState.INIT
        pending_error = None

        try:
            while state != MissionState.DONE:
                if state != self.state:
                    self.log(f"-> {state}")

                self.state = state
                state = self.handlers[state]()

        except KeyboardInterrupt:
            self.log("Interrupted by the operator (Ctrl+C): stopping the robot.")
            self.set_end(EndReason.OPERATOR_INTERRUPT, f"in state {self.state}")
            self.stop_robot()

        except Exception as error:
            self.log(f"ERROR in state {self.state}: {error!r}")
            traceback.print_exc()
            self.set_end(EndReason.ERROR, f"in state {self.state}")
            self.error = repr(error)
            self.stop_robot()
            pending_error = error

        summary = self.finalize()

        if pending_error is not None:
            raise pending_error

        return summary

    # -----------------------------------------------------------------
    # States
    # -----------------------------------------------------------------

    def state_init(self):
        with self.timer.measure("init"):
            if not self.plan_only:
                self.spot.check_estop()
                self.spot.acquire_lease()
                self.spot.power_on()
                self.spot.stand()

            self.acquirer.initialize()
            self.acquirer.verify_calibration()

            self.home_pose = self.spot.get_body_pose_2d()
            self.stats["battery_start"] = self.spot.get_battery_percentage()

        x, y, yaw = self.home_pose

        self.log(
            f"Home pose (vision): x={x:+.2f}, y={y:+.2f}, yaw={math.degrees(yaw):+.0f} deg, "
            f"battery {self.format_battery(self.stats['battery_start'])}"
        )

        self.timer.mark("exploration_start")

        return MissionState.ACQUIRE

    def state_acquire(self):
        self.start_cycle()

        failure = None

        with self.timer.measure("acquisition") as span:
            try:
                self.snapshot = self.acquirer.capture(settle=True)
            except AcquisitionError as error:
                self.snapshot = None
                failure = error

        self.cycle["acquisition_s"] = span.duration

        if failure is not None:
            self.consecutive_acquisition_failures += 1
            self.stats["acquisition_failures"] += 1
            self.cycle["note"] = f"acquisition failed: {failure}"
            self.log(f"Acquisition failed ({self.consecutive_acquisition_failures} in a row): {failure}")

            return MissionState.CHECK

        self.consecutive_acquisition_failures = 0
        self.stats["snapshots"] += 1
        self.last_capture_xy = self.snapshot.body_position_xy()
        self.record_capture_pose(self.snapshot, "stop_capture")

        if self.observation_logger is not None:
            with self.timer.measure("logging"):
                self.observation_logger.log(self.snapshot)

        return MissionState.MAP_UPDATE

    def state_map_update(self):
        moving = self.take_moving_buffer()

        if moving and self.observation_logger is not None:
            with self.timer.measure("logging"):
                for snapshot in moving:
                    self.observation_logger.log(snapshot)

        with self.timer.measure("mapping") as span:
            fused, skipped = 0, 0

            # Chronological: the snapshots taken on the way, then the
            # one taken here.
            for snapshot in moving + [self.snapshot]:
                snapshot_fused, snapshot_skipped = self.fuse_snapshot(snapshot)
                fused += snapshot_fused
                skipped += snapshot_skipped

        self.update_coverage()

        self.cycle["mapping_s"] = span.duration
        self.cycle["cameras_fused"] = fused
        self.cycle["cameras_skipped"] = skipped
        self.cycle["moving_fused"] = len(moving)

        self.log(
            f"Fused {fused} camera(s) from {len(moving) + 1} snapshot(s) "
            f"({len(moving)} taken on the move), skipped {skipped} (static) in "
            f"{span.duration:.2f} s, coverage {self.last_coverage['coverage'] * 100:.1f} %"
        )

        if self.config.save_map_every_cycle:
            with self.timer.measure("map_saving"):
                self.save_map()

        return MissionState.CHECK

    def state_check(self):
        battery = self.spot.get_battery_percentage()
        exploration_time = self.timer.elapsed_since("exploration_start")

        self.cycle["battery_percent"] = battery
        self.cycle["exploration_time_s"] = exploration_time

        reason, detail = self.stop_criteria.check(
            coverage=None if self.last_coverage is None else self.last_coverage["coverage"],
            exploration_time_s=exploration_time,
            battery_percent=battery,
            consecutive_nav_failures=self.consecutive_nav_failures,
            consecutive_acquisition_failures=self.consecutive_acquisition_failures
        )

        # Plan-only: one cycle whatever the map looks like; only
        # repeated acquisition failures can stop it.
        if self.plan_only and reason != StopCriteria.ACQUISITION_FAILURES:
            reason = None

        if reason is not None:
            self.finish_cycle(f"stop: {reason}")
            return self.end_exploration(reason, detail)

        if self.snapshot is None:
            # Acquisition failed this cycle: try again from here.
            self.finish_cycle()
            return MissionState.ACQUIRE

        if not self.map_anchored():
            self.finish_cycle("map not anchored yet")
            return MissionState.ACQUIRE

        return MissionState.PLAN

    def state_plan(self):
        robot_xy = self.spot.get_body_position_xy()
        grid_map = self.traversability.grid_map

        with self.timer.measure("planning") as span:
            cost_layer = self.traversability.get_cost_layer("total")
            result = self.plan_with_retries(cost_layer, robot_xy, grid_map)

        status = self.planner.last_status

        self.cycle["planning_s"] = span.duration
        self.cycle["plan_status"] = status

        if result is not None:
            self.report.add_planned_path(self.cycle_index, "exploration", 1, result["waypoints"])
            self.cycle["frontier_x"], self.cycle["frontier_y"] = result["goal_world"]
            self.cycle["waypoints"] = len(result["waypoints"])

            self.log(
                f"Plan: frontier at ({result['goal_world'][0]:+.2f}, {result['goal_world'][1]:+.2f}), "
                f"{len(result['waypoints'])} waypoints, {span.duration:.2f} s"
            )
        else:
            self.log(f"Plan: none ({status}), {span.duration:.2f} s")

        self.plot_cycle(cost_layer, result, robot_xy)

        if result is None:
            if status == PlanningPipeline.STATUS_NO_FRONTIERS:
                reason, detail = EndReason.EXPLORATION_COMPLETE, "no frontier left"
            else:
                reason = EndReason.NO_REACHABLE_FRONTIER
                detail = (
                    f"planner status {status}, "
                    f"{len(self.planner.target_blacklist)} target(s) blacklisted"
                )

            self.finish_cycle(f"plan: {status}")

            if self.plan_only:
                return self.end_exploration(EndReason.PLAN_ONLY, detail)

            return self.end_exploration(reason, detail)

        if self.plan_only:
            self.finish_cycle("plan only: not navigating")
            return self.end_exploration(EndReason.PLAN_ONLY, "one cycle planned, robot not moved")

        self.plan_result = result

        return MissionState.NAVIGATE

    def state_navigate(self):
        result = self.plan_result

        with self.timer.measure("navigation") as span:
            nav = self.follower.follow(result["waypoints"])

        self.stats["walked_exploration_m"] += nav.traveled_distance_m

        self.cycle["navigation_s"] = span.duration
        self.cycle["nav_status"] = nav.status
        self.cycle["nav_distance_to_goal_m"] = nav.distance_to_goal
        self.cycle["nav_walked_m"] = nav.traveled_distance_m

        self.log(f"Navigation: {nav.summary()}")

        if nav.succeeded:
            self.stats["nav_succeeded"] += 1
            self.consecutive_nav_failures = 0
            self.finish_cycle()

            return MissionState.ACQUIRE

        failed = self.stats["nav_failed"]
        failed[nav.status] = failed.get(nav.status, 0) + 1

        if nav.status == NavResult.FAULT:
            self.finish_cycle("robot fault")
            self.end_exploration(EndReason.ROBOT_FAULT, nav.message)

            return MissionState.SHUTDOWN

        if nav.status == NavResult.COMMAND_FAILED:
            self.finish_cycle("command failed")
            self.end_exploration(EndReason.OPERATOR_OVERRIDE, nav.message)

            return MissionState.SHUTDOWN

        # Timeout / stalled / blocked: give up on this frontier and
        # acquire wherever the robot stopped.
        self.planner.blacklist_target(result["goal_world"])
        self.consecutive_nav_failures += 1

        self.finish_cycle(f"target blacklisted ({nav.status})")
        self.log(
            f"Target blacklisted, {self.consecutive_nav_failures} consecutive "
            "navigation failure(s)."
        )

        return MissionState.ACQUIRE

    def state_return_home(self):
        info = self.return_home
        info["attempted"] = True

        home_xy = self.home_pose[:2]

        with self.timer.measure("return_home"):
            for attempt in range(1, self.config.return_home_max_attempts + 1):
                info["attempts"] = attempt

                if attempt > 1:
                    # Update the map from where the last attempt stopped.
                    self.acquire_and_fuse_quietly()

                outcome = self.return_home_attempt(home_xy, attempt)

                if outcome in ("reached", "abort", "no_plan"):
                    break

        robot_xy = self.spot.get_body_position_xy()
        info["final_distance_m"] = math.hypot(robot_xy[0] - home_xy[0], robot_xy[1] - home_xy[1])

        self.timer.mark("return_home_end")
        self.report.flush_paths()

        self.log(
            f"Return home: {'reached' if info['reached'] else 'NOT reached'} after "
            f"{info['attempts']} attempt(s), {info['final_distance_m']:.2f} m from home"
        )

        return MissionState.SHUTDOWN

    def state_shutdown(self):
        with self.timer.measure("shutdown"):
            self.stats["battery_end"] = self.safe_battery()
            self.spot.shutdown()

        self.shutdown_done = True

        return MissionState.DONE

    # -----------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------

    def on_robot_pose(self, pose):
        """WaypointFollower callback: one walked-path row per polled pose."""

        phase = "return_home" if self.state == MissionState.RETURN_HOME else "exploration"
        self.record_path_pose(pose, phase, "walk")

    def record_capture_pose(self, snapshot, event):
        R, t = snapshot.vision_tform_body
        pose = (float(t[0]), float(t[1]), yaw_from_rotation(R))
        phase = "return_home" if self.state == MissionState.RETURN_HOME else "exploration"

        self.record_path_pose(pose, phase, event)

    def record_path_pose(self, pose, phase, event):
        elapsed = self.timer.elapsed_since("mission_start") or 0.0

        self.report.add_path_pose(elapsed, self.cycle_index, phase, event, pose)

    def on_waypoint_passed(self, index, waypoint, pose):
        """
        WaypointFollower callback (moving capture): one capture attempt
        if far enough from the last capture, buffered for the next stop.
        Runs while Spot walks on toward the next waypoint.
        """

        if self.state != MissionState.NAVIGATE:
            return

        if self.last_capture_xy is not None:
            distance = math.hypot(pose[0] - self.last_capture_xy[0], pose[1] - self.last_capture_xy[1])

            if distance < self.config.moving_capture_min_distance_m:
                return

        with self.timer.measure("moving_acquisition") as span:
            try:
                snapshot = self.acquirer.capture(settle=False, retry=False)
            except AcquisitionError as error:
                snapshot = None
                failure = error

        if self.cycle is not None:
            self.cycle["moving_captured"] = self.cycle.get("moving_captured", 0) + (snapshot is not None)
            self.cycle["moving_acquisition_s"] = self.cycle.get("moving_acquisition_s", 0.0) + span.duration

        if snapshot is None:
            self.stats["moving_capture_failures"] += 1
            self.log(f"Moving capture at waypoint {index} failed: {failure}")
            return

        self.moving_buffer.append(snapshot)
        self.stats["moving_snapshots"] += 1
        self.last_capture_xy = snapshot.body_position_xy()
        self.record_capture_pose(snapshot, "moving_capture")

        self.log(
            f"Moving capture at waypoint {index} ({pose[0]:+.2f}, {pose[1]:+.2f}) "
            f"in {span.duration:.2f} s, {len(self.moving_buffer)} buffered"
        )

    def take_moving_buffer(self):
        buffered = self.moving_buffer
        self.moving_buffer = []

        return buffered

    def fuse_snapshot(self, snapshot):
        """Fuse each keyframe camera; returns (fused, skipped)."""

        fused = 0
        skipped = 0

        for camera_name, obs in snapshot.cameras.items():
            keep = self.is_keyframe(camera_name, obs.R, obs.t)

            self.report.add_camera_pose(obs.acquisition_time, camera_name, obs.R, obs.t, keep)

            if not keep:
                skipped += 1
                continue

            self.traversability.update(obs.rgb, obs.depth, obs.R, obs.t, camera_name)
            self.last_fused_pose[camera_name] = (np.asarray(obs.R), np.asarray(obs.t))
            fused += 1

        self.stats["frames_fused"] += fused
        self.stats["frames_skipped"] += skipped

        return fused, skipped

    def is_keyframe(self, camera_name, R, t):
        if not self.config.keyframes_enabled or camera_name not in self.last_fused_pose:
            return True

        last_R, last_t = self.last_fused_pose[camera_name]

        moved = float(np.linalg.norm(np.asarray(t) - last_t))

        cos_angle = (np.trace(last_R.T @ np.asarray(R)) - 1.0) / 2.0
        rotated = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

        return (
            moved >= self.config.keyframe_min_translation_m
            or rotated >= self.config.keyframe_min_rotation_deg
        )

    def plan_with_retries(self, cost_layer, robot_xy, grid_map):
        """
        plan(); while A* fails on every tried frontier (STATUS_NO_PATH),
        blacklist those targets and plan again.
        """

        origin = grid_map.get_origin()[:2]

        for _ in range(MAX_REPLANS_PER_CYCLE):
            result = self.planner.plan(
                cost_layer, robot_xy, grid_map.resolution, origin, grid_map.cell_n
            )

            if result is not None or self.planner.last_status != PlanningPipeline.STATUS_NO_PATH:
                return result

            unreachable = [
                rejection["cell"] for rejection in self.planner.last_rejections
                if rejection["stage"] == "path"
            ]

            if not unreachable:
                return None

            for cell in unreachable:
                self.planner.blacklist_target(grid_map.cell_to_world(*cell))

            self.log(f"A* failed on {len(unreachable)} frontier(s): blacklisted, replanning.")

        return None

    def return_home_attempt(self, home_xy, attempt):
        """
        One planning + navigation attempt toward home.

        Returns "reached", "retry", "abort" (fault / operator) or
        "no_plan".
        """

        info = self.return_home
        grid_map = self.traversability.grid_map
        robot_xy = self.spot.get_body_position_xy()

        if not self.map_anchored():
            info["detail"] = "no map"
            return "no_plan"

        with self.timer.measure("return_planning"):
            result = self.planner.plan_to_goal(
                home_xy, self.traversability.get_cost_layer("total"), robot_xy,
                grid_map.resolution, grid_map.get_origin()[:2], grid_map.cell_n
            )

        status = self.planner.last_status

        if status == PlanningPipeline.STATUS_ALREADY_AT_GOAL:
            info["reached"] = True
            return "reached"

        if result is None:
            info["detail"] = f"no path home ({status})"
            self.log(f"Return home attempt {attempt}: {info['detail']}")
            return "no_plan"

        self.report.add_planned_path(self.cycle_index, "return_home", attempt, result["waypoints"])
        self.log(f"Return home attempt {attempt}: {len(result['waypoints'])} waypoints")

        with self.timer.measure("return_navigation"):
            nav = self.follower.follow(result["waypoints"])

        self.stats["walked_return_m"] += nav.traveled_distance_m
        self.log(f"Return home attempt {attempt}: {nav.summary()}")

        if nav.succeeded:
            info["reached"] = True
            return "reached"

        info["detail"] = f"{nav.status}: {nav.message}"

        if nav.status in (NavResult.FAULT, NavResult.COMMAND_FAILED):
            return "abort"

        return "retry"

    def acquire_and_fuse_quietly(self):
        """Best-effort map update (return home): errors only logged."""

        try:
            with self.timer.measure("return_acquisition"):
                snapshot = self.acquirer.capture(settle=True)

            self.record_capture_pose(snapshot, "stop_capture")

            if self.observation_logger is not None:
                self.observation_logger.log(snapshot)

            with self.timer.measure("return_mapping"):
                self.fuse_snapshot(snapshot)

            self.stats["snapshots"] += 1

        except AcquisitionError as error:
            self.stats["acquisition_failures"] += 1
            self.log(f"Acquisition during return home failed: {error}")

    # -----------------------------------------------------------------
    # Cycle bookkeeping
    # -----------------------------------------------------------------

    def start_cycle(self):
        self.cycle_index += 1

        x, y, yaw = self.spot.get_body_pose_2d()

        self.cycle = {
            "cycle": self.cycle_index,
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_monotonic": time.monotonic(),
            "robot_x": x,
            "robot_y": y,
            "robot_yaw_deg": math.degrees(yaw),
        }

        self.snapshot = None
        self.plan_result = None

        self.log(f"===== Cycle {self.cycle_index} at ({x:+.2f}, {y:+.2f}) =====")

    def finish_cycle(self, note=None):
        if self.cycle is None:
            return

        if note:
            self.cycle["note"] = note

        if self.last_coverage is not None:
            self.cycle["coverage"] = self.last_coverage["coverage"]
            self.cycle["observed_cells"] = self.last_coverage["observed_cells"]

        self.cycle["cycle_s"] = time.monotonic() - self.cycle["start_monotonic"]

        self.report.write_cycle(self.cycle)
        self.report.flush_paths()
        self.timer.add("cycle", self.cycle["cycle_s"])

        self.cycle = None

    def end_exploration(self, reason, detail):
        """Record why the exploration ended and pick the next state."""

        self.set_end(reason, detail)

        if not self.timer.has_mark("exploration_end"):
            self.timer.mark("exploration_end")

        self.log(f"Exploration ended: {reason} ({detail})")

        abort = reason in (EndReason.ROBOT_FAULT, EndReason.OPERATOR_OVERRIDE, EndReason.PLAN_ONLY)

        if self.config.return_home_enabled and not abort and not self.plan_only:
            return MissionState.RETURN_HOME

        return MissionState.SHUTDOWN

    def set_end(self, reason, detail):
        # The first reason wins: e.g. an error during the return home
        # does not hide why the exploration ended.
        if self.end_reason is None:
            self.end_reason = reason
            self.end_detail = detail

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    def finalize(self):
        """Shut down (if not done), save everything, print the summary."""

        if self.cycle is not None:
            self.finish_cycle(f"interrupted in {self.state}")

        if self.timer.has_mark("exploration_start") and not self.timer.has_mark("exploration_end"):
            self.timer.mark("exploration_end")

        if not self.shutdown_done:
            try:
                self.state_shutdown()
            except Exception as error:
                self.log(f"WARNING: shutdown failed: {error!r}")

        self.timer.mark("mission_end")

        # Captured on the move but never fused (mission ended while
        # walking): at least keep the raw data.
        leftover = self.take_moving_buffer()

        if leftover and self.observation_logger is not None:
            self.log(f"Saving {len(leftover)} unfused moving snapshot(s) to disk.")

            for snapshot in leftover:
                self.safe_step("moving snapshot saving", lambda: self.observation_logger.log(snapshot))

        self.safe_step("map saving", self.save_map)
        self.safe_step("trajectory saving", self.report.save_trajectory)
        self.safe_step("path saving", self.report.flush_paths)

        summary = self.build_summary()

        self.safe_step("summary saving", lambda: self.report.save_summary(summary))

        if self.mission_plotter is not None:
            self.safe_step("mission plot", lambda: self.plot_mission(summary))

        self.print_summary(summary)

        return summary

    def plot_mission(self, summary):
        """mission_path.png: walked path over the final cost map."""

        if not self.report.path_rows:
            return

        grid_map = self.traversability.grid_map
        anchored = self.map_anchored()

        self.mission_plotter.plot(
            save_path=self.report.mission_plot_path,
            cost_layer=self.traversability.get_cost_layer("total") if anchored else None,
            resolution=grid_map.resolution,
            origin=grid_map.get_origin()[:2] if anchored else (0.0, 0.0),
            cell_n=grid_map.cell_n,
            path_rows=self.report.path_rows,
            planned_paths=self.report.planned_paths,
            home_pose=self.home_pose,
            title=f"Mission {self.report.run_dir.name}",
            subtitle=self.mission_plotter.subtitle_from_summary(self.report.to_jsonable(summary))
        )

        self.log(f"Mission plot saved to {self.report.mission_plot_path}")

    def build_summary(self):
        final_pose = None

        try:
            final_pose = self.spot.get_body_pose_2d()
        except Exception:
            pass

        return {
            "run_dir": self.report.run_dir,
            "plan_only": self.plan_only,
            "end_reason": self.end_reason,
            "end_detail": self.end_detail,
            "error": self.error,
            "cycles": self.cycle_index,
            "timing": self.timer.summary(),
            "coverage": self.last_coverage,
            "acquisition": {
                "snapshots": self.stats["snapshots"],
                "failures": self.stats["acquisition_failures"],
                "moving_snapshots": self.stats["moving_snapshots"],
                "moving_capture_failures": self.stats["moving_capture_failures"],
                "frames_fused": self.stats["frames_fused"],
                "frames_skipped": self.stats["frames_skipped"],
            },
            "navigation": {
                "succeeded": self.stats["nav_succeeded"],
                "failed": self.stats["nav_failed"],
                "blacklisted_targets": len(self.planner.target_blacklist),
                "walked_exploration_m": self.stats["walked_exploration_m"],
                "walked_return_m": self.stats["walked_return_m"],
            },
            "return_home": self.return_home,
            "battery_percent": {
                "start": self.stats["battery_start"],
                "end": self.stats["battery_end"],
            },
            "home_pose": self.home_pose,
            "final_pose": final_pose,
            "config": self.config.as_dict(),
        }

    def print_summary(self, summary):
        print("\n" + "=" * 60)
        print("MISSION SUMMARY")
        print("=" * 60)
        print(f"End reason: {summary['end_reason']} ({summary['end_detail']})")

        if summary["error"]:
            print(f"Error: {summary['error']}")

        coverage = summary["coverage"]

        print(f"Cycles: {summary['cycles']}")
        print(
            "Coverage: " + (
                "n/a" if coverage is None
                else f"{coverage['coverage'] * 100:.1f} % "
                     f"({coverage['observed_cells']}/{coverage['region_cells']} cells)"
            )
        )

        navigation = summary["navigation"]

        print(
            f"Navigation: {navigation['succeeded']} reached, failed {navigation['failed'] or 'none'}, "
            f"{navigation['blacklisted_targets']} blacklisted"
        )
        print(
            f"Walked: {navigation['walked_exploration_m']:.1f} m exploring, "
            f"{navigation['walked_return_m']:.1f} m returning"
        )

        if summary["return_home"]["attempted"]:
            home = summary["return_home"]
            print(
                f"Return home: {'reached' if home['reached'] else 'NOT reached'}, "
                f"{home['final_distance_m']:.2f} m from home"
            )

        battery = summary["battery_percent"]
        print(f"Battery: {self.format_battery(battery['start'])} -> {self.format_battery(battery['end'])}")

        acquisition = summary["acquisition"]
        print(
            f"Snapshots: {acquisition['snapshots']} at stops ({acquisition['failures']} failed), "
            f"{acquisition['moving_snapshots']} on the move "
            f"({acquisition['moving_capture_failures']} failed), "
            f"frames fused {acquisition['frames_fused']}, skipped {acquisition['frames_skipped']}"
        )

        print()
        print(self.timer.format_summary())
        print(f"\nOutputs in {summary['run_dir']}")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def update_coverage(self):
        self.last_coverage = self.coverage.compute(
            self.traversability.grid_map.get_count_layer()
        )

    def map_anchored(self):
        try:
            return self.traversability.grid_map.get_origin() is not None
        except Exception:
            return False

    def plot_cycle(self, cost_layer, result, robot_xy):
        if self.cycle_plotter is None:
            return

        path = self.report.cycle_plot_path(self.cycle_index)
        coverage = 0.0 if self.last_coverage is None else self.last_coverage["coverage"]
        title = (
            f"Cycle {self.cycle_index} - {self.planner.last_status} - "
            f"coverage {coverage * 100:.1f} %"
        )

        with self.timer.measure("plotting"):
            self.safe_step(
                "cycle plot",
                lambda: self.cycle_plotter(
                    path, cost_layer, self.planner.last_detection, result, robot_xy, title
                )
            )

    def save_map(self):
        if self.map_anchored():
            self.traversability.save_map(self.report.run_dir / "grid_map.npz")

    def stop_robot(self):
        if self.follower is not None and getattr(self.spot, "has_lease", False):
            self.follower.stop_robot()

    def safe_battery(self):
        try:
            return self.spot.get_battery_percentage()
        except Exception:
            return None

    def safe_step(self, description, function):
        try:
            return function()
        except Exception as error:
            self.log(f"WARNING: {description} failed: {error!r}")
            return None

    @staticmethod
    def format_battery(value):
        return "n/a" if value is None else f"{value:.0f} %"

    @staticmethod
    def log(message):
        print(f"[Mission {time.strftime('%H:%M:%S')}] {message}")