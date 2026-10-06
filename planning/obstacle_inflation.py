"""
Robot-footprint inflation of the occupied cells: turns the point-robot
planning problem into a disc-robot one, so A* and the smoother keep
Spot's body -- not just its center -- clear of obstacles.

Without inflation, PathPlanner only excludes the occupied cells
themselves (4 cm each): a path may pass one cell away from a trunk,
i.e. with Spot's ~0.5 m wide body overlapping it. Spot then either
hits the obstacle or stops on its own obstacle avoidance.

TWO ZONES, from the euclidean distance d of each cell to the nearest
occupied cell (FrontierDetector's classification):

    d <= robot_radius_m
        HARD: impassable for A* and for the smoother's line-of-sight
        check, exactly like an occupied cell.

    robot_radius_m < d < robot_radius_m + inflation_radius_m
        SOFT: an extra cost, linear from inflation_max_cost at the
        hard edge down to 0 at the outer edge, combined with C_total
        by max(): A* prefers the middle of a passage to grazing the
        hard limit, but can still squeeze through when needed.

Spot is not a disc (~1.1 x 0.5 m). robot_radius_m sits between half
its width (the minimum, walking straight through a gap) and half its
length (which would close most forest passages); the soft zone adds
the margin a disc model lacks when the robot turns.

Only KNOWN cells get the soft cost: unknown (NaN) cells are left NaN,
they already cost nan_cost_penalty in A*. The hard zone, on the other
hand, does extend into unknown cells -- an unknown cell 10 cm from a
known trunk is not reachable by Spot's body, whatever it contains.

START / GOAL: the robot physically stands on its own cells, which may
fall inside the hard zone (robot next to a tree) or even be classified
occupied (a cost artifact under the legs): clear_around() makes a disc
around the start traversable again. A target inside the hard zone is
moved to the nearest traversable cell by snap_to_traversable(), or
discarded if there is none within the snap radius.
"""

import numpy as np
from scipy import ndimage


class ObstacleInflation:

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def inflate(self, occupied_mask, resolution):
        """
        Parameters
        ----------
        occupied_mask : numpy.ndarray
            Shape (H, W), bool -- FrontierDetector's "occupied".

        resolution : float
            Grid cell size, in meters.

        Returns
        -------
        dict with keys:
            "distance_m" : (H, W) float32, distance of each cell to the
                nearest occupied cell [m] (inf if there is none).
            "hard" : (H, W) bool, the impassable zone.
            "inflation_cost" : (H, W) float32 in [0, 1], the soft
                cost (inflation_max_cost on the hard zone too, for
                display; hard cells are blocked anyway).
        """

        config = self.planning_config

        distance_m = self.distance_to_occupied(occupied_mask, resolution)

        robot_radius = config.robot_radius_m
        soft_width = config.inflation_radius_m
        max_cost = config.inflation_max_cost

        hard = distance_m <= robot_radius

        inflation_cost = np.zeros(occupied_mask.shape, dtype=np.float32)

        if soft_width > 0 and max_cost > 0:
            soft = (distance_m > robot_radius) & (distance_m < robot_radius + soft_width)

            inflation_cost[soft] = max_cost * (
                1.0 - (distance_m[soft] - robot_radius) / soft_width
            )

        inflation_cost[hard] = max_cost

        return {
            "distance_m": distance_m,
            "hard": hard,
            "inflation_cost": inflation_cost
        }

    @staticmethod
    def distance_to_occupied(occupied_mask, resolution):
        """Euclidean distance [m] of every cell to the nearest occupied one."""

        if not np.any(occupied_mask):
            return np.full(occupied_mask.shape, np.inf, dtype=np.float32)

        # distance_transform_edt: distance of each non-zero cell to the
        # nearest zero cell -> zeros must be the occupied cells.
        return (
            ndimage.distance_transform_edt(~occupied_mask) * resolution
        ).astype(np.float32)

    @staticmethod
    def apply_to_cost(cost_layer, inflation_cost):
        """
        C_total combined with the soft cost by max(), on known cells
        only; unknown (NaN) cells stay NaN.
        """

        planning_cost = cost_layer.astype(np.float32, copy=True)
        known = np.isfinite(cost_layer)

        planning_cost[known] = np.maximum(cost_layer[known], inflation_cost[known])

        return planning_cost

    def clear_around(self, blocked_mask, center_cell, radius_m, resolution):
        """
        Copy of blocked_mask with a disc of radius_m around center_cell
        made traversable. The center cell itself is always cleared,
        even with radius_m = 0.
        """

        cleared = blocked_mask.copy()

        disc = self.disc_mask(blocked_mask.shape, center_cell, radius_m / resolution)
        cleared[disc] = False

        row, col = int(round(center_cell[0])), int(round(center_cell[1]))

        if self.is_inside((row, col), blocked_mask.shape):
            cleared[row, col] = False

        return cleared

    def snap_to_traversable(self, cell, blocked_mask, radius_m, resolution):
        """
        The target itself if its cell is traversable, otherwise the
        nearest traversable cell within radius_m; None if there is
        none (or the target is outside the grid).

        Returns
        -------
        (row, col) float tuple, or None.
        """

        row, col = int(round(cell[0])), int(round(cell[1]))

        if not self.is_inside((row, col), blocked_mask.shape):
            return None

        if not blocked_mask[row, col]:
            return (float(cell[0]), float(cell[1]))

        radius_cells = radius_m / resolution

        if radius_cells <= 0:
            return None

        height, width = blocked_mask.shape

        row_min = max(0, int(np.floor(row - radius_cells)))
        row_max = min(height, int(np.ceil(row + radius_cells)) + 1)
        col_min = max(0, int(np.floor(col - radius_cells)))
        col_max = min(width, int(np.ceil(col + radius_cells)) + 1)

        rows, cols = np.meshgrid(
            np.arange(row_min, row_max),
            np.arange(col_min, col_max),
            indexing="ij"
        )

        distances = np.sqrt((rows - row) ** 2 + (cols - col) ** 2)

        candidates = (~blocked_mask[row_min:row_max, col_min:col_max]) & (
            distances <= radius_cells
        )

        if not np.any(candidates):
            return None

        masked = np.where(candidates, distances, np.inf)
        index = np.unravel_index(int(np.argmin(masked)), masked.shape)

        return (float(rows[index]), float(cols[index]))

    @staticmethod
    def disc_mask(shape, center_cell, radius_cells):
        """(H, W) bool mask of the cells within radius_cells of center_cell."""

        mask = np.zeros(shape, dtype=bool)

        if radius_cells <= 0:
            return mask

        height, width = shape
        row, col = center_cell

        row_min = max(0, int(np.floor(row - radius_cells)))
        row_max = min(height, int(np.ceil(row + radius_cells)) + 1)
        col_min = max(0, int(np.floor(col - radius_cells)))
        col_max = min(width, int(np.ceil(col + radius_cells)) + 1)

        if row_min >= row_max or col_min >= col_max:
            return mask

        rows, cols = np.meshgrid(
            np.arange(row_min, row_max),
            np.arange(col_min, col_max),
            indexing="ij"
        )

        mask[row_min:row_max, col_min:col_max] = (
            (rows - row) ** 2 + (cols - col) ** 2 <= radius_cells ** 2
        )

        return mask

    @staticmethod
    def is_inside(cell, shape):
        return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]
