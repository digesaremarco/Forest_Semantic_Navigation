"""
Outcome of one WaypointFollower.follow() call, consumed by the main
loop: on success it acquires at the frontier; on any failure it
blacklists the target (PlanningPipeline.blacklist_target()) and counts
a consecutive navigation failure.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class NavResult:

    # The robot reached the final waypoint.
    REACHED = "reached"

    # The current waypoint's time budget ran out.
    TIMEOUT = "timeout"

    # No progress toward the current waypoint for stall_window_s.
    STALLED = "stalled"

    # Spot's own obstacle avoidance reported the goal as blocked, or
    # the trajectory stopped short of the final waypoint.
    BLOCKED = "blocked"

    # The command was overridden (e.g. the operator took control from
    # the tablet), timed out on the robot, or the robot froze.
    COMMAND_FAILED = "command_failed"

    # A behavior fault (e.g. a fall) is active.
    FAULT = "fault"

    # follow() was interrupted (Ctrl+C) -- the robot was stopped.
    ABORTED = "aborted"

    # Empty waypoint list, or the robot already at the final waypoint.
    NOTHING_TO_DO = "nothing_to_do"

    status: str

    # Final waypoint (x, y) in the vision frame, None if there was none.
    goal: Optional[Tuple[float, float]]

    # Body pose (x, y, yaw) in vision when follow() returned.
    final_pose: Tuple[float, float, float]

    # Distance from final_pose to goal [m] (0 if goal is None).
    distance_to_goal: float

    # Index of the last waypoint passed or reached, -1 if none.
    last_reached_index: int
    total_waypoints: int

    elapsed_s: float

    # Length of the path actually walked (sum of polled positions) [m].
    traveled_distance_m: float

    message: str = ""

    # Final heading error [deg] (only when REACHED).
    heading_error_deg: Optional[float] = None

    # Names of the active behavior faults (only when FAULT).
    faults: list = field(default_factory=list)

    @property
    def succeeded(self):
        return self.status in (self.REACHED, self.NOTHING_TO_DO)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        text = (
            f"{self.status}: waypoint {self.last_reached_index + 1}/{self.total_waypoints}, "
            f"{self.distance_to_goal:.2f} m from goal, {self.traveled_distance_m:.2f} m "
            f"walked in {self.elapsed_s:.1f} s"
        )

        return text + (f" -- {self.message}" if self.message else "")
