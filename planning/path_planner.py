"""
A* path planning on the fused traversability cost grid.

Edge cost from a cell to an 8-connected neighbor B:

    edge_cost(A, B) = weight_distance * step_distance(A, B)
                     + weight_traversability * cost(B)

where step_distance is `resolution` for cardinal moves and
`resolution * sqrt(2)` for diagonal moves, and cost(B) is C_total(B),
with NaN (unobserved) cells substituted by nan_cost_penalty -- the
same "optimistic, moderate penalty instead of infinite" policy
already used elsewhere in the project. This matters here because
frontier targets sit right at the edge of unobserved regions:
treating NaN as impassable would often make the chosen frontier
literally unreachable.

OCCUPIED cells (FrontierDetector's classification: known cells with
cost >= occupied_threshold) are treated as genuinely IMPASSABLE --
excluded from the search entirely, not just penalized. Cost values
near 1.0 correspond to effectively solid obstacles in
class_costs.yaml (tree, building, vehicle, container), so allowing a
path straight through one just because the numeric cost is merely
"high" rather than infinite would be unsafe. This reuses the exact
same occupied_threshold FrontierDetector already uses, so
"free vs occupied" means the same thing everywhere in planning/.

The heuristic is Euclidean distance to the goal scaled by
weight_distance alone (ignoring any traversability contribution,
which can only be >= 0) -- an admissible lower bound as long as
weight_distance and weight_traversability are both non-negative,
which PlanningConfig already enforces, so the search remains optimal.
"""

import heapq
import math

import numpy as np


class PathPlanner:

    # 8-connected neighbor offsets: (d_row, d_col)
    NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def plan(self, start, goal, cost_layer, occupied_mask, resolution):
        """
        Find the lowest-cost path from start to goal on the grid.

        Parameters
        ----------
        start, goal : tuple of float or int
            (row, col) grid cell coordinates. Rounded to the nearest
            integer cell internally, so a raw frontier centroid can
            be passed directly.

        cost_layer : numpy.ndarray
            Shape (H, W), the C_total layer.

        occupied_mask : numpy.ndarray
            Shape (H, W), bool -- FrontierDetector.detect()["occupied"].
            Occupied cells are excluded from the search entirely.

        resolution : float
            Grid cell size, in meters -- used to scale step_distance.

        Returns
        -------
        list of (row, col) int tuples, from start to goal inclusive,
        or None if no path exists.
        """

        start = (int(round(start[0])), int(round(start[1])))
        goal = (int(round(goal[0])), int(round(goal[1])))

        self.validate_inputs(start, goal, cost_layer, occupied_mask)

        if occupied_mask[start[0], start[1]]:
            raise ValueError(
                f"start position {start} is inside an occupied cell "
                "-- cannot plan from there."
            )

        if occupied_mask[goal[0], goal[1]]:
            raise ValueError(
                f"goal position {goal} is inside an occupied cell "
                "-- cannot plan to there."
            )

        height, width = cost_layer.shape

        weight_distance = self.planning_config.weight_distance
        weight_traversability = self.planning_config.weight_traversability
        nan_penalty = self.planning_config.nan_cost_penalty

        open_heap = []
        counter = 0  # tie-breaker so the heap never compares tuples directly

        heapq.heappush(open_heap, (0.0, counter, start))

        came_from = {}
        g_score = {start: 0.0}

        visited = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for d_row, d_col in self.NEIGHBOR_OFFSETS:
                neighbor = (current[0] + d_row, current[1] + d_col)

                if not (
                    0 <= neighbor[0] < height and 0 <= neighbor[1] < width
                ):
                    continue

                if occupied_mask[neighbor[0], neighbor[1]]:
                    continue

                if neighbor in visited:
                    continue

                step_distance = resolution * math.sqrt(
                    d_row ** 2 + d_col ** 2
                )

                neighbor_cost = cost_layer[neighbor[0], neighbor[1]]

                if np.isnan(neighbor_cost):
                    neighbor_cost = nan_penalty

                edge_cost = (
                    weight_distance * step_distance
                    + weight_traversability * neighbor_cost
                )

                tentative_g = g_score[current] + edge_cost

                if (
                    neighbor not in g_score
                    or tentative_g < g_score[neighbor]
                ):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    heuristic = weight_distance * resolution * math.sqrt(
                        (goal[0] - neighbor[0]) ** 2
                        + (goal[1] - neighbor[1]) ** 2
                    )

                    priority = tentative_g + heuristic

                    counter += 1
                    heapq.heappush(open_heap, (priority, counter, neighbor))

        return None

    @staticmethod
    def reconstruct_path(came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    @staticmethod
    def validate_inputs(start, goal, cost_layer, occupied_mask):
        if not isinstance(cost_layer, np.ndarray) or cost_layer.ndim != 2:
            raise ValueError(
                "cost_layer must be a 2D numpy.ndarray, got "
                f"{getattr(cost_layer, 'shape', type(cost_layer))}."
            )

        if (
            not isinstance(occupied_mask, np.ndarray)
            or occupied_mask.shape != cost_layer.shape
        ):
            raise ValueError(
                "occupied_mask must have the same shape as "
                f"cost_layer, got "
                f"{getattr(occupied_mask, 'shape', type(occupied_mask))} "
                f"vs {cost_layer.shape}."
            )

        height, width = cost_layer.shape

        for name, point in (("start", start), ("goal", goal)):
            row, col = point

            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(
                    f"{name} position {point} is outside the grid "
                    f"({height}x{width})."
                )