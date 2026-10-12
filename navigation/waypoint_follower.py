"""
Follows a list of waypoints (PlanningPipeline.plan() / plan_to_goal()
output) with Spot's own trajectory controller, in the vision frame.

ONE WAYPOINT AT A TIME, NO STOPS IN BETWEEN: each waypoint is sent as
a synchro_se2_trajectory_point_command (x, y, heading). As soon as the
body is within pass_through_radius_m of an intermediate waypoint, the
next one is commanded -- the new command replaces the running one, so
Spot keeps walking instead of decelerating to a stop at every
waypoint. Only the final waypoint (the frontier) is driven to a full
stop: it counts as reached when Spot reports the trajectory STOPPED
with the body within goal_tolerance_m.

HEADING: each waypoint's heading points to the next one
(WaypointGenerator), so Spot walks facing forward and arrives at the
frontier facing past it -- the front cameras look at the unexplored
area for the next acquisition.

START: following begins from the waypoint closest to the robot, and
waypoints within pass_through_radius_m of it are skipped (the first
one is normally the robot's own start cell) -- the robot never walks
back to waypoints it is already past.

FAILURE DETECTION (any of these ends follow() and stops the robot):
    TIMEOUT        the current waypoint's budget ran out
                   (base_timeout_s + distance / nominal_speed_mps), or
                   the whole path exceeded max_path_duration_s.
    STALLED        the distance to the current waypoint did not drop
                   by min_progress_m within stall_window_s.
    BLOCKED        Spot reported FINAL_GOAL_STATUS_BLOCKED (its obstacle
                   avoidance cannot get there) for blocked_patience_s,
                   or the final trajectory STOPPED short of the goal.
    COMMAND_FAILED the mobility command is no longer processing:
                   overridden (operator took control from the tablet),
                   timed out on the robot, robot frozen.
    FAULT          a behavior fault is active (e.g. a fall).
Ctrl+C stops the robot and re-raises KeyboardInterrupt.

FEEDBACK STATUSES (SDK 5.x): SE2TrajectoryCommand.Feedback.status is
STATUS_STOPPED (= the old STATUS_AT_GOAL), STATUS_STOPPING (= old
NEAR_GOAL) or STATUS_IN_PROGRESS (= old GOING_TO_GOAL) -- same numeric
values, only the names changed; the numeric values are used here.

The robot must already be powered on and standing, with this client
holding the lease (SpotInterface.acquire_lease/power_on/stand).
"""

import math
import time

from bosdyn.api import basic_command_pb2, geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from google.protobuf import wrappers_pb2

from navigation.nav_result import NavResult


TrajectoryFeedback = basic_command_pb2.SE2TrajectoryCommand.Feedback
CommandStatus = basic_command_pb2.RobotCommandFeedbackStatus

# SE2 trajectory feedback (numeric values are stable across SDK versions).
TRAJ_STOPPED = 1      # STATUS_STOPPED / STATUS_AT_GOAL
TRAJ_IN_PROGRESS = 2  # STATUS_IN_PROGRESS / STATUS_GOING_TO_GOAL
TRAJ_STOPPING = 3     # STATUS_STOPPING / STATUS_NEAR_GOAL

FINAL_GOAL_BLOCKED = 3  # FINAL_GOAL_STATUS_BLOCKED

# Extra validity given to each command on the robot beyond the
# follower's own per-waypoint budget, so the robot-side expiry never
# fires first.
COMMAND_END_TIME_MARGIN_S = 5.0


class WaypointFollower:

    def __init__(self, navigation_config, spot):
        """
        Parameters
        ----------
        navigation_config : NavigationConfig

        spot : SpotInterface
            Connected, with the lease, powered on and standing.
        """

        self.config = navigation_config
        self.spot = spot

        self.mobility_params = self.build_mobility_params()

    def follow(self, waypoints):
        """
        Walk through all waypoints, stopping only at the last one.

        Parameters
        ----------
        waypoints : list of dict
            Each with "position" (x, y) and "heading" (rad), in the
            vision frame -- PlanningPipeline's "waypoints".

        Returns
        -------
        NavResult
        """

        self.spot.require_lease()

        start_time = time.monotonic()
        state = self.spot.get_robot_state()
        pose = self.spot.get_body_pose_2d(state)

        tracker = {
            "pose": pose,
            "traveled": 0.0,
            "last_reached": -1,
        }

        if not waypoints:
            return self.make_result(NavResult.NOTHING_TO_DO, None, tracker, 0, start_time, "no waypoints")

        goal = self.position(waypoints[-1])
        first = self.first_target_index(waypoints, pose)

        if first is None:
            tracker["last_reached"] = len(waypoints) - 1

            return self.make_result(
                NavResult.NOTHING_TO_DO, goal, tracker, len(waypoints), start_time,
                "already at the final waypoint"
            )

        tracker["last_reached"] = first - 1

        try:
            for index in range(first, len(waypoints)):
                is_final = index == len(waypoints) - 1

                outcome, message = self.drive_to(waypoints[index], is_final, tracker, start_time)

                if outcome != NavResult.REACHED:
                    self.stop_robot()

                    result = self.make_result(outcome, goal, tracker, len(waypoints), start_time, message)
                    result.faults = tracker.get("faults", [])

                    return result

                tracker["last_reached"] = index

        except KeyboardInterrupt:
            self.stop_robot()
            raise

        result = self.make_result(NavResult.REACHED, goal, tracker, len(waypoints), start_time)
        result.heading_error_deg = math.degrees(abs(
            self.angle_difference(tracker["pose"][2], waypoints[-1]["heading"])
        ))

        if result.heading_error_deg > self.config.goal_heading_tolerance_deg:
            result.message = (
                f"final heading off by {result.heading_error_deg:.1f} deg "
                f"(tolerance {self.config.goal_heading_tolerance_deg:.0f})"
            )

        return result

    def go_to(self, x, y, heading):
        """Single-goal convenience: follow([that goal])."""

        return self.follow([{"position": (x, y), "heading": heading}])

    def stop_robot(self):
        """Stop in place (best effort, never raises)."""

        try:
            self.spot.robot_command_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception as error:
            print(f"[WaypointFollower] WARNING: stop command failed: {error}")

    def drive_to(self, waypoint, is_final, tracker, start_time):
        """
        Command one waypoint and poll until it is passed (intermediate),
        reached (final) or a failure is detected.

        Returns
        -------
        (status, message) -- status NavResult.REACHED on success.
        """

        config = self.config
        x, y = self.position(waypoint)

        distance = self.distance(tracker["pose"], (x, y))
        budget = config.base_timeout_s + distance / config.nominal_speed_mps

        command_id = self.send_trajectory(x, y, waypoint["heading"], budget + COMMAND_END_TIME_MARGIN_S)

        waypoint_start = time.monotonic()

        # Stall detection: best distance so far, and when it last
        # improved by at least min_progress_m.
        best_distance = distance
        last_progress_time = waypoint_start

        blocked_since = None

        while True:
            time.sleep(config.feedback_period_s)
            now = time.monotonic()

            state = self.spot.get_robot_state()
            self.update_pose(tracker, self.spot.get_body_pose_2d(state))

            distance = self.distance(tracker["pose"], (x, y))

            faults = self.spot.get_active_faults(state)["behavior"]

            if faults:
                tracker["faults"] = faults
                return NavResult.FAULT, f"behavior fault(s): {', '.join(faults)}"

            feedback = self.spot.robot_command_client.robot_command_feedback(command_id)
            mobility = feedback.feedback.synchronized_feedback.mobility_command_feedback

            if mobility.status != CommandStatus.STATUS_PROCESSING:
                return NavResult.COMMAND_FAILED, (
                    "mobility command "
                    f"{CommandStatus.Status.Name(mobility.status)}"
                )

            trajectory = mobility.se2_trajectory_feedback

            # Success
            if not is_final and distance <= config.pass_through_radius_m:
                return NavResult.REACHED, ""

            if trajectory.status == TRAJ_STOPPED:
                if distance <= config.goal_tolerance_m or not is_final:
                    # Intermediate waypoint reached exactly (e.g. the
                    # pass-through radius is smaller than Spot's own
                    # goal tolerance): just move on.
                    return NavResult.REACHED, ""

                return NavResult.BLOCKED, (
                    f"trajectory stopped {distance:.2f} m from the final waypoint"
                )

            # Blocked
            if trajectory.final_goal_status == FINAL_GOAL_BLOCKED:
                if blocked_since is None:
                    blocked_since = now
                elif now - blocked_since >= config.blocked_patience_s:
                    return NavResult.BLOCKED, (
                        f"Spot reports the goal blocked for {now - blocked_since:.1f} s"
                    )
            else:
                blocked_since = None

            # Stalled
            if distance <= best_distance - config.min_progress_m:
                best_distance = distance
                last_progress_time = now
            elif now - last_progress_time >= config.stall_window_s:
                return NavResult.STALLED, (
                    f"no progress for {now - last_progress_time:.1f} s, "
                    f"{distance:.2f} m from waypoint"
                )

            # Timeouts
            if now - waypoint_start >= budget:
                return NavResult.TIMEOUT, f"waypoint budget {budget:.1f} s exceeded"

            if now - start_time >= config.max_path_duration_s:
                return NavResult.TIMEOUT, (
                    f"path duration cap {config.max_path_duration_s:.0f} s exceeded"
                )

    def send_trajectory(self, x, y, heading, validity_s):
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=x,
            goal_y=y,
            goal_heading=heading,
            frame_name=VISION_FRAME_NAME,
            params=self.mobility_params
        )

        # end_time_secs is local time: RobotCommandClient converts it to
        # robot time with the time sync established at connection.
        return self.spot.robot_command_client.robot_command(
            command, end_time_secs=time.time() + validity_s
        )

    def build_mobility_params(self):
        config = self.config

        max_velocity = geometry_pb2.SE2Velocity(
            linear=geometry_pb2.Vec2(
                x=config.max_linear_velocity_mps, y=config.max_lateral_velocity_mps
            ),
            angular=config.max_angular_velocity_rps
        )

        min_velocity = geometry_pb2.SE2Velocity(
            linear=geometry_pb2.Vec2(
                x=-config.max_linear_velocity_mps, y=-config.max_lateral_velocity_mps
            ),
            angular=-config.max_angular_velocity_rps
        )

        disable_avoidance = not config.obstacle_avoidance_enabled

        obstacle_params = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=disable_avoidance,
            disable_vision_foot_obstacle_avoidance=disable_avoidance,
            disable_vision_foot_constraint_avoidance=disable_avoidance,
            obstacle_avoidance_padding=config.obstacle_avoidance_padding_m
        )

        params = spot_command_pb2.MobilityParams(
            vel_limit=geometry_pb2.SE2VelocityLimit(max_vel=max_velocity, min_vel=min_velocity),
            obstacle_params=obstacle_params,
            locomotion_hint=spot_command_pb2.LocomotionHint.Value(
                "HINT_" + config.locomotion_hint.upper()
            ),
            stair_hint=False
        )

        if config.body_height_m != 0.0:
            body_params = RobotCommandBuilder.mobility_params(body_height=config.body_height_m)
            params.body_control.CopyFrom(body_params.body_control)

        if config.ground_mu_hint is not None:
            params.terrain_params.ground_mu_hint.CopyFrom(
                wrappers_pb2.DoubleValue(value=config.ground_mu_hint)
            )

        return params

    # Helpers
    def first_target_index(self, waypoints, pose):
        """
        Where to start: from the waypoint closest to the robot (never
        walk back to waypoints it is already past), the first one not
        within pass_through_radius_m. None if the robot is already at
        the final waypoint.
        """

        last = len(waypoints) - 1

        if self.distance(pose, self.position(waypoints[last])) <= self.config.goal_tolerance_m:
            return None

        distances = [self.distance(pose, self.position(waypoint)) for waypoint in waypoints]
        closest = min(range(len(waypoints)), key=lambda index: distances[index])

        for index in range(closest, last):
            if distances[index] > self.config.pass_through_radius_m:
                return index

        return last

    @staticmethod
    def update_pose(tracker, pose):
        tracker["traveled"] += math.hypot(pose[0] - tracker["pose"][0], pose[1] - tracker["pose"][1])
        tracker["pose"] = pose

    def make_result(self, status, goal, tracker, total, start_time, message=""):
        pose = tracker["pose"]

        return NavResult(
            status=status,
            goal=goal,
            final_pose=tuple(float(value) for value in pose),
            distance_to_goal=0.0 if goal is None else self.distance(pose, goal),
            last_reached_index=tracker["last_reached"],
            total_waypoints=total,
            elapsed_s=time.monotonic() - start_time,
            traveled_distance_m=tracker["traveled"],
            message=message
        )

    @staticmethod
    def position(waypoint):
        x, y = waypoint["position"][:2]

        return float(x), float(y)

    @staticmethod
    def distance(pose, point):
        return math.hypot(point[0] - pose[0], point[1] - pose[1])

    @staticmethod
    def angle_difference(a, b):
        """a - b wrapped to (-pi, pi]."""

        return math.atan2(math.sin(a - b), math.cos(a - b))
