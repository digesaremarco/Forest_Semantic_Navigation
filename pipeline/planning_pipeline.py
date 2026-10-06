"""
Planning sub-pipeline: orchestrates frontier detection, refinement,
selection, A* path planning, smoothing and waypoint generation
behind a small plan() / plan_to_goal() / should_replan() interface --
the planning-side counterpart to TraversabilityPipeline.

Consumes plain arrays and scalars (a cost layer, resolution, origin,
cell_n), not a live TraversabilityPipeline or GridMap reference, so
the two sub-pipelines stay decoupled: the main loop pulls
TraversabilityPipeline.get_cost_layer("total") and the relevant
GridMap metadata itself and passes them in here.

OBSTACLE INFLATION (planning/obstacle_inflation.py): A*, the smoother
and the frontier refinement work on the occupied cells dilated by
the robot radius (impassable), plus a soft cost zone around them
combined with C_total -- so paths keep Spot's whole body clear of
obstacles and stay in the middle of passages. The robot's own disc is
made traversable again around the start (it is standing there), and
targets inside the impassable zone are snapped to the nearest
traversable cell or discarded. Frontier DETECTION still runs on the
raw C_total: inflation changes where the robot can go, not what is
known.

CLUSTER TARGET POINT: a frontier cluster's raw arithmetic-mean
centroid (as returned by FrontierDetector) can be a poor navigation
target for large or wide clusters -- in the degenerate case of a
frontier ring that fully surrounds the robot, the mean of all ring
cells coincides with the robot's OWN position. plan() replaces each
cluster's centroid with the cluster's frontier CELL closest to the
robot, among those at least min_target_distance_m away: the body
cameras never see the ground right around Spot, so that ring stays
unknown after every stop and would otherwise be the "nearest
frontier" every time. Clusters with no cell that far are dropped. The
original mean is kept under "mean_centroid".

FALLBACK AND BLACKLIST: candidates are ranked by utility and A* is
tried on them in order, up to max_plan_attempts, so a best frontier
A* cannot reach no longer ends the cycle. Targets the robot failed to
reach while NAVIGATING are blacklisted by the main loop
(blacklist_target()) and discarded from then on.

STATUS: plan() returning None used to mean both "exploration
complete" and "planning failed". last_status now tells them apart --
see the STATUS_* constants. The main loop ends the exploration on
STATUS_NO_FRONTIERS; on the others there are frontiers it cannot
currently use.
"""

import numpy as np

from planning.frontier_detector import FrontierDetector
from planning.frontier_refiner import FrontierRefiner
from planning.frontier_selector import FrontierSelector
from planning.path_planner import PathPlanner
from planning.path_smoother import PathSmoother
from planning.waypoint_generator import WaypointGenerator
from planning.replanning_trigger import ReplanningTrigger
from planning.obstacle_inflation import ObstacleInflation
from planning.target_blacklist import TargetBlacklist


class PlanningPipeline:

    # A path was found (plan(): to a frontier; plan_to_goal(): to the goal).
    STATUS_PLANNED = "planned"

    # plan(): no frontier at all -- exploration complete.
    STATUS_NO_FRONTIERS = "no_frontiers"

    # plan(): frontiers exist, but none survived retargeting, snapping
    # and the blacklist (too close, enclosed by obstacles, already
    # failed).
    STATUS_NO_VALID_FRONTIER = "no_valid_frontier"

    # A* found no path to any tried target.
    STATUS_NO_PATH = "no_path"

    # The robot's own cell is outside the grid.
    STATUS_START_OUTSIDE_GRID = "start_outside_grid"

    # plan_to_goal(): goal outside the grid / inside the impassable
    # zone with no traversable cell within goal_snap_radius_m.
    STATUS_GOAL_OUTSIDE_GRID = "goal_outside_grid"
    STATUS_GOAL_BLOCKED = "goal_blocked"

    # plan_to_goal(): the robot is already within goal_reached_distance_m.
    STATUS_ALREADY_AT_GOAL = "already_at_goal"

    def __init__(self, planning_config):
        self.planning_config = planning_config

        self.frontier_detector = FrontierDetector(planning_config)
        self.frontier_refiner = FrontierRefiner(planning_config)
        self.frontier_selector = FrontierSelector(planning_config)
        self.path_planner = PathPlanner(planning_config)
        self.path_smoother = PathSmoother(planning_config)
        self.waypoint_generator = WaypointGenerator(planning_config)
        self.replanning_trigger = ReplanningTrigger(planning_config)
        self.obstacle_inflation = ObstacleInflation(planning_config)
        self.target_blacklist = TargetBlacklist(planning_config.blacklist_radius_m)

        # Diagnostics of the most recent plan()/plan_to_goal() call,
        # set even when it returned None.
        self.last_status = None
        self.last_detection = None
        self.last_frontier = None
        self.last_context = None
        self.last_rejections = []

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def plan(self, cost_layer, robot_position_world, resolution, origin, cell_n):
        """
        Plan the path to the next frontier.

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
            None if no path was planned -- last_status says why.
            Otherwise:
                "waypoints" : list of dict, from WaypointGenerator
                    (each with "position" and "heading").
                "frontier" : the selected, scored frontier cluster.
                "path_cells" : the smoothed path, in (row, col) cells.
                "goal_world" : (x, y) of the target, meters -- what to
                    pass to blacklist_target() if navigation fails.
        """

        self.reset_diagnostics()

        detection = self.frontier_detector.detect(cost_layer)
        self.last_detection = detection

        if len(detection["clusters"]) == 0:
            return self.finish(self.STATUS_NO_FRONTIERS)

        robot_cell = self.world_to_cell(robot_position_world, resolution, origin, cell_n)

        if not self.obstacle_inflation.is_inside(self.round_cell(robot_cell), cost_layer.shape):
            return self.finish(self.STATUS_START_OUTSIDE_GRID)

        context = self.build_context(
            cost_layer, detection["free"], detection["occupied"], robot_cell, resolution
        )
        self.last_context = context

        min_distance_cells = self.planning_config.min_target_distance_m / resolution

        clusters = self.retarget_clusters(
            detection["clusters"], robot_cell, min_distance_cells, self.last_rejections
        )

        refined = self.frontier_refiner.refine(
            clusters, context["cost"], context["traversable_free"]
        )

        candidates = self.filter_candidates(
            refined, context, robot_cell, min_distance_cells, resolution, origin, cell_n
        )

        if not candidates:
            return self.finish(self.STATUS_NO_VALID_FRONTIER)

        ranked = self.frontier_selector.rank(
            candidates, context["cost"], detection["unknown"], robot_cell
        )

        for frontier in ranked[:self.planning_config.max_plan_attempts]:
            path = self.plan_path(robot_cell, frontier["refined_centroid"], context, resolution)

            if path is None:
                self.reject(frontier["refined_centroid"], "path", "A* found no non-trivial path")
                continue

            self.last_frontier = frontier

            result = self.build_result(path, resolution, origin, cell_n)
            result["frontier"] = frontier

            return self.finish(self.STATUS_PLANNED, result)

        return self.finish(self.STATUS_NO_PATH)

    def plan_to_goal(self, goal_world, cost_layer, robot_position_world, resolution, origin, cell_n):
        """
        Plan the path to an arbitrary world-frame goal (e.g. the start
        position, to return home), with the same inflation, smoothing
        and waypoint generation as plan(). No frontier logic, no
        blacklist.

        Parameters
        ----------
        goal_world : tuple of float
            (x, y), meters, same frame as robot_position_world.

        Other parameters as plan().

        Returns
        -------
        dict or None
            None if no path was planned -- last_status says why.
            Otherwise "waypoints", "path_cells" and "goal_world" (the
            goal actually used: snapped if the requested one was
            inside the impassable zone). With STATUS_ALREADY_AT_GOAL,
            "waypoints" and "path_cells" are empty.
        """

        self.reset_diagnostics()

        distance = np.linalg.norm(
            np.asarray(goal_world[:2], dtype=np.float64)
            - np.asarray(robot_position_world[:2], dtype=np.float64)
        )

        if distance <= self.planning_config.goal_reached_distance_m:
            return self.finish(self.STATUS_ALREADY_AT_GOAL, {
                "waypoints": [],
                "path_cells": [],
                "goal_world": (float(goal_world[0]), float(goal_world[1])),
            })

        self.frontier_detector.validate_cost_layer(cost_layer)
        _, free, occupied = self.frontier_detector.classify(cost_layer)

        robot_cell = self.world_to_cell(robot_position_world, resolution, origin, cell_n)

        if not self.obstacle_inflation.is_inside(self.round_cell(robot_cell), cost_layer.shape):
            return self.finish(self.STATUS_START_OUTSIDE_GRID)

        goal_cell = self.world_to_cell(goal_world, resolution, origin, cell_n)

        if not self.obstacle_inflation.is_inside(self.round_cell(goal_cell), cost_layer.shape):
            return self.finish(self.STATUS_GOAL_OUTSIDE_GRID)

        context = self.build_context(cost_layer, free, occupied, robot_cell, resolution)
        self.last_context = context

        snapped = self.obstacle_inflation.snap_to_traversable(
            goal_cell, context["blocked"], self.planning_config.goal_snap_radius_m, resolution
        )

        if snapped is None:
            self.reject(goal_cell, "snap", "goal inside the impassable zone")
            return self.finish(self.STATUS_GOAL_BLOCKED)

        path = self.plan_path(robot_cell, snapped, context, resolution)

        if path is None:
            self.reject(snapped, "path", "A* found no non-trivial path")
            return self.finish(self.STATUS_NO_PATH)

        return self.finish(self.STATUS_PLANNED, self.build_result(path, resolution, origin, cell_n))

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

    def blacklist_target(self, goal_world):
        """
        Exclude a frontier target the robot failed to reach -- pass
        plan()'s result["goal_world"]. Frontier targets within
        blacklist_radius_m of it are discarded from then on.
        """

        self.target_blacklist.add(goal_world)

    def clear_blacklist(self):
        self.target_blacklist.clear()

    # -----------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------

    def build_context(self, cost_layer, free_mask, occupied_mask, robot_cell, resolution):
        """
        Everything the search works on:
            "cost" : C_total combined with the soft inflation cost.
            "blocked" : impassable mask (hard inflation), with the
                robot's own disc cleared.
            "traversable_free" : free & not blocked -- what the
                frontier refinement may shift targets toward.
            "inflation" : ObstacleInflation.inflate() output.
        """

        start_radius = self.planning_config.start_clear_radius_m

        # Occupied cells under the robot are artifacts (it is standing
        # there): drop them BEFORE inflating, otherwise their inflated
        # ring would still trap the robot inside its cleared disc.
        under_robot = self.obstacle_inflation.disc_mask(
            occupied_mask.shape, robot_cell, start_radius / resolution
        )

        inflation = self.obstacle_inflation.inflate(occupied_mask & ~under_robot, resolution)

        # Then clear the robot's disc of the hard zone of obstacles
        # just outside it (robot standing next to a tree).
        blocked = self.obstacle_inflation.clear_around(
            inflation["hard"], robot_cell, start_radius, resolution
        )

        return {
            "cost": self.obstacle_inflation.apply_to_cost(cost_layer, inflation["inflation_cost"]),
            "blocked": blocked,
            "traversable_free": free_mask & ~blocked,
            "inflation": inflation,
            "occupied_cleared_under_robot": int(np.sum(occupied_mask & under_robot)),
        }

    def filter_candidates(self, refined, context, robot_cell, min_distance_cells, resolution, origin, cell_n):
        """
        Snap each refined target to a traversable cell, then drop the
        ones too close to the robot or blacklisted.
        """

        candidates = []

        for cluster in refined:
            target = cluster["refined_centroid"]

            snapped = self.obstacle_inflation.snap_to_traversable(
                target, context["blocked"], self.planning_config.goal_snap_radius_m, resolution
            )

            if snapped is None:
                self.reject(target, "snap", "target inside the impassable zone")
                continue

            if self.cell_distance(snapped, robot_cell) < min_distance_cells:
                self.reject(snapped, "distance", "target too close to the robot")
                continue

            if self.target_blacklist.contains(
                WaypointGenerator.cell_to_world(snapped[0], snapped[1], resolution, origin, cell_n)
            ):
                self.reject(snapped, "blacklist", "target near a blacklisted point")
                continue

            candidate = dict(cluster)
            candidate["unsnapped_centroid"] = target
            candidate["refined_centroid"] = snapped

            candidates.append(candidate)

        return candidates

    def plan_path(self, start_cell, goal_cell, context, resolution):
        """Smoothed A* path on the inflated context, or None."""

        try:
            raw_path = self.path_planner.plan(
                start_cell, goal_cell, context["cost"], context["blocked"], resolution
            )
        except ValueError:
            # Start/goal blocked or outside the grid: both are
            # prevented upstream, kept as a safety net.
            return None

        if raw_path is None or len(raw_path) < 2:
            return None

        return self.path_smoother.smooth(raw_path, context["cost"], context["blocked"])

    def build_result(self, path, resolution, origin, cell_n):
        waypoints = self.waypoint_generator.generate(path, resolution, origin, cell_n)

        goal_world = WaypointGenerator.cell_to_world(
            path[-1][0], path[-1][1], resolution, origin, cell_n
        )

        self.replanning_trigger.start_plan(goal_world)

        return {
            "waypoints": waypoints,
            "path_cells": path,
            "goal_world": goal_world,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def reset_diagnostics(self):
        self.last_status = None
        self.last_detection = None
        self.last_frontier = None
        self.last_context = None
        self.last_rejections = []

    def finish(self, status, result=None):
        self.last_status = status

        return result

    def reject(self, cell, stage, reason):
        self.last_rejections.append({
            "cell": (float(cell[0]), float(cell[1])),
            "stage": stage,
            "reason": reason,
        })

    @staticmethod
    def retarget_clusters(clusters, robot_position_cell, min_distance_cells=0.0, rejections=None):
        """
        Replace each cluster's "centroid" with its frontier cell
        closest to the robot among those at least min_distance_cells
        away -- see module docstring, "CLUSTER TARGET POINT". Clusters
        with no such cell are dropped (and recorded in rejections, if
        given). The original mean centroid is kept under
        "mean_centroid".
        """

        retargeted = []

        for cluster in clusters:
            rows = cluster["rows"]
            cols = cluster["cols"]

            distances = np.sqrt(
                (rows - robot_position_cell[0]) ** 2
                + (cols - robot_position_cell[1]) ** 2
            )

            eligible = distances >= min_distance_cells

            if not np.any(eligible):
                if rejections is not None:
                    rejections.append({
                        "cell": cluster["centroid"],
                        "stage": "retarget",
                        "reason": "whole cluster within min_target_distance of the robot",
                    })
                continue

            closest_idx = int(np.argmin(np.where(eligible, distances, np.inf)))

            new_cluster = dict(cluster)
            new_cluster["mean_centroid"] = cluster["centroid"]
            new_cluster["centroid"] = (float(rows[closest_idx]), float(cols[closest_idx]))

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

        x, y = position_world[0], position_world[1]

        col = (x - origin[0]) / resolution + half
        row = (y - origin[1]) / resolution + half

        return (row, col)

    @staticmethod
    def round_cell(cell):
        return (int(round(cell[0])), int(round(cell[1])))

    @staticmethod
    def cell_distance(cell_a, cell_b):
        return float(np.hypot(cell_a[0] - cell_b[0], cell_a[1] - cell_b[1]))