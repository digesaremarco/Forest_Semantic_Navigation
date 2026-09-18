"""
Replanning triggers: decides when the exploration loop should
compute a new plan instead of continuing to follow the current one.

Two triggers are implemented for this first pass -- both standard,
low-risk choices for a first working loop:
    - goal_reached: the robot is within goal_reached_distance_m of
      the current goal.
    - timeout: more than timeout_s have passed since the current
      plan was started, whether or not progress is being made.

A third trigger -- replanning when the traversability cost map has
changed enough along the current path to make it worth re-evaluating
-- is deliberately NOT implemented here. It is the most valuable
trigger in principle (it reacts to genuinely new information, not
just elapsed time or proximity) but also the hardest to tune well
(what counts as "changed enough"?), and there is no real field data
yet to calibrate it against. Left as a documented extension point
(should_replan_due_to_cost_change(), currently returns False
unconditionally) rather than guessed at now.

Works in world-frame meters throughout (both the goal and the robot
position), matching what WaypointGenerator's output and the robot's
live pose already are -- no grid-cell conversion needed here.
"""

import time

import numpy as np


class ReplanningTrigger:

    def __init__(self, planning_config):
        self.planning_config = planning_config

        self.current_goal = None
        self.plan_start_time = None

    def start_plan(self, goal):
        """
        Record that a new plan has just been computed, targeting
        `goal` (world-frame (x, y), meters). Call this once, right
        after a new plan (PathPlanner + WaypointGenerator) is
        produced.
        """

        self.current_goal = goal
        self.plan_start_time = time.monotonic()

    def should_replan(self, robot_position):
        """
        Whether the exploration loop should compute a new plan now.

        Parameters
        ----------
        robot_position : tuple of float
            (x, y), the robot's current position in world-frame
            meters (same frame as the `goal` passed to
            start_plan()).

        Returns
        -------
        (bool, str or None)
            Whether to replan, and, if True, a short reason string
            ("goal_reached", "timeout", or "cost_change") useful for
            logging.
        """

        if self.current_goal is None:
            raise RuntimeError(
                "should_replan() called before any start_plan() -- "
                "there is no active plan to check against."
            )

        if self.is_goal_reached(robot_position):
            return True, "goal_reached"

        if self.is_timed_out():
            return True, "timeout"

        if self.should_replan_due_to_cost_change():
            return True, "cost_change"

        return False, None

    def is_goal_reached(self, robot_position):
        distance = np.linalg.norm(
            np.array(robot_position, dtype=np.float64)
            - np.array(self.current_goal, dtype=np.float64)
        )

        return distance <= self.planning_config.goal_reached_distance_m

    def is_timed_out(self):
        elapsed = time.monotonic() - self.plan_start_time

        return elapsed >= self.planning_config.timeout_s

    def should_replan_due_to_cost_change(self):
        """
        Placeholder for a future trigger based on how much the
        traversability cost map has changed along the current path
        since the plan was made. Always False for now -- see module
        docstring.
        """

        return False