"""
Planning sub-pipeline: orchestrates frontier detection, refinement,
selection, A* path planning, smoothing and waypoint generation
behind a single plan()/should_replan() interface -- the planning-side
counterpart to TraversabilityPipeline.

Consumes plain arrays and scalars (a cost layer, resolution, origin,
cell_n), not a live TraversabilityPipeline or GridMap reference, so
the two sub-pipelines stay decoupled: the eventual main loop is
expected to pull TraversabilityPipeline.get_cost_layer("total") and
the relevant GridMap metadata itself and pass them in here, rather
than this module reaching into the other one.

KNOWN LIMITATION, left for later rather than guessed at now: if
PathPlanner fails to reach the single best-scored frontier, or the
resulting path is trivial (start == goal after rounding -- e.g. the
selected frontier's refined centroid landed back on the robot's own
position, see "CLUSTER TARGET POINT" below), plan() currently gives
up for this cycle instead of retrying the next-best-utility
frontier. Worth revisiting once there's real field behavior to see
how often this actually happens.

CLUSTER TARGET POINT: a frontier cluster's raw arithmetic-mean
centroid (as returned by FrontierDetector) can be a poor navigation
target for large or wide clusters -- in the degenerate case of a
frontier ring that fully surrounds the robot (e.g. after exploring a
roughly circular area with the robot near its center), the mean of
all ring cells coincides with the robot's OWN position, which is
useless as a target. plan() replaces each cluster's centroid with
the actual frontier CELL closest to the robot before refinement/
selection -- guaranteed to be a real point on the frontier, and
practically identical to the mean for the typical compact,
non-encircling clusters this degenerate case doesn't apply to. The
original mean is kept under "mean_centroid" for reference.
"""

import numpy as np

from planning.frontier_detector import FrontierDetector
from planning.frontier_refiner import FrontierRefiner
from planning.frontier_selector import FrontierSelector
from planning.path_planner import PathPlanner
from planning.path_smoother import PathSmoother
from planning.waypoint_generator import WaypointGenerator
from planning.replanning_trigger import ReplanningTrigger


class PlanningPipeline:

    def __init__(self, planning_config):
        self.planning_config = planning_config

        self.frontier_detector = FrontierDetector(planning_config)
        self.frontier_refiner = FrontierRefiner(planning_config)
        self.frontier_selector = FrontierSelector(planning_config)
        self.path_planner = PathPlanner(planning_config)
        self.path_smoother = PathSmoother(planning_config)
        self.waypoint_generator = WaypointGenerator(planning_config)
        self.replanning_trigger = ReplanningTrigger(planning_config)

        # Set by the most recent plan() call, even when it returned
        # None -- useful for diagnostics/visualization (e.g. showing
        # detected frontiers even on a cycle where planning failed).
        self.last_detection = None
        self.last_frontier = None

    def plan(self, cost_layer, robot_position_world, resolution, origin, cell_n):
        """
        Run the full planning pipeline for one cycle.

        Parameters
        ----------
        cost_layer : numpy.ndarray
            Shape (H, W), C_total -- typically
            TraversabilityPipeline.get_cost_layer("total").

        robot_position_world : tuple of float
            (x, y), the robot's current position in world-frame
            meters.

        resolution : float
            Grid cell size, in meters -- GridMap.resolution.

        origin : numpy.ndarray or tuple
            (x, y), the grid's anchored origin -- GridMap.get_origin()[:2].

        cell_n : int
            Cells per side of the (square) grid -- GridMap.cell_n.

        Returns
        -------
        dict or None
            None if no frontier was found (exploration complete or
            everything currently unreachable). Otherwise:
                "waypoints" : list of dict, from WaypointGenerator
                    (each with "position" and "heading").
                "frontier" : the selected, scored frontier cluster.
                "path_cells" : the smoothed path, in (row, col) grid
                    cells.
        """

        detection = self.frontier_detector.detect(cost_layer)

        self.last_detection = detection

        if len(detection["clusters"]) == 0:
            self.last_frontier = None
            return None

        robot_position_cell = self.world_to_cell(
            robot_position_world, resolution, origin, cell_n
        )

        clusters_with_target = self.retarget_clusters(
            detection["clusters"], robot_position_cell
        )

        refined_clusters = self.frontier_refiner.refine(
            clusters_with_target, cost_layer, detection["free"]
        )

        best_frontier = self.frontier_selector.select(
            refined_clusters, cost_layer, detection["unknown"],
            robot_position_cell
        )

        self.last_frontier = best_frontier

        if best_frontier is None:
            return None

        raw_path = self.path_planner.plan(
            robot_position_cell, best_frontier["refined_centroid"],
            cost_layer, detection["occupied"], resolution
        )

        if raw_path is None or len(raw_path) < 2:
            # No path at all, OR a trivial one (start == goal after
            # rounding -- e.g. the selected frontier's refined
            # centroid landed back on the robot's own position, see
            # "CLUSTER TARGET POINT" above). Either way there is
            # nothing meaningful to navigate to this cycle.
            return None

        smoothed_path = self.path_smoother.smooth(
            raw_path, cost_layer, detection["occupied"]
        )

        waypoints = self.waypoint_generator.generate(
            smoothed_path, resolution, origin, cell_n
        )

        goal_world = (
            waypoints[-1]["position"] if waypoints else robot_position_world
        )

        self.replanning_trigger.start_plan(goal_world)

        return {
            "waypoints": waypoints,
            "frontier": best_frontier,
            "path_cells": smoothed_path
        }

    def should_replan(self, robot_position_world):
        """
        Whether the current plan should be abandoned in favor of a
        new one. Delegates to ReplanningTrigger -- see there for the
        goal_reached / timeout logic.

        Returns
        -------
        (bool, str or None) -- see ReplanningTrigger.should_replan().
        """

        return self.replanning_trigger.should_replan(robot_position_world)

    @staticmethod
    def retarget_clusters(clusters, robot_position_cell):
        """
        Replace each cluster's "centroid" with the cluster's own
        frontier cell closest to the robot -- see module docstring,
        "CLUSTER TARGET POINT". The original mean centroid is kept
        under "mean_centroid".
        """

        retargeted = []

        for cluster in clusters:
            rows = cluster["rows"]
            cols = cluster["cols"]

            distances = np.sqrt(
                (rows - robot_position_cell[0]) ** 2
                + (cols - robot_position_cell[1]) ** 2
            )

            closest_idx = int(np.argmin(distances))

            target_point = (
                float(rows[closest_idx]), float(cols[closest_idx])
            )

            new_cluster = dict(cluster)
            new_cluster["mean_centroid"] = cluster["centroid"]
            new_cluster["centroid"] = target_point

            retargeted.append(new_cluster)

        return retargeted

    @staticmethod
    def world_to_cell(position_world, resolution, origin, cell_n):
        """
        Inverse of GridMap.cell_to_world() / WaypointGenerator's own
        formula: convert a world-frame (x, y) position, in meters,
        to (row, col) grid cell coordinates. Returned as float --
        PathPlanner and FrontierSelector both already accept and
        round/handle float positions themselves.
        """

        half = cell_n // 2

        x, y = position_world

        col = (x - origin[0]) / resolution + half
        row = (y - origin[1]) / resolution + half

        return (row, col)