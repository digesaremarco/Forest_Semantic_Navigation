"""
Frontier selection: scores each refined frontier candidate by a
utility function trading off information gain against an economical
estimate of the cost to reach it, and picks the best one.

    utility = beta_gain * normalized_information_gain
            - beta_cost  * normalized_navigation_cost

Information gain (per candidate) is the standard frontier-exploration
proxy: the number of UNKNOWN cells within information_gain_radius_cells
of the candidate's refined centroid -- more unknown neighborhood means
visiting it reveals more new area.

Navigation cost (per candidate) is a CHEAP ESTIMATE, not a real A*
search: euclidean distance from the robot's current position to the
candidate, scaled by (1 + average cost sampled along the straight
line between them). Running full A* on every candidate just to
discard most of them would be wasteful -- A* itself only runs once,
in PathPlanner, on the single frontier this module selects.

Both raw scores are normalized by their own maximum across the
CURRENT candidate set (min-max style, so whichever frontier has the
largest raw gain/cost among the current options scores 1.0 on that
term). This makes the two terms comparable to each other and keeps
the utility scale meaningful regardless of grid size or how many
candidates exist on a given call -- rather than normalizing against
some fixed theoretical maximum.

Unknown/NaN cells sampled along the straight-line cost estimate are
treated with the same nan_cost_penalty used later by PathPlanner's
A*, for consistency between this cheap estimate and the real search
that follows.
"""

import numpy as np


class FrontierSelector:

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def select(self, refined_clusters, cost_layer, unknown_mask, robot_position):
        """
        Parameters
        ----------
        refined_clusters : list of dict
            As returned by FrontierRefiner.refine(); each needs a
            "refined_centroid" (row, col) float tuple.

        cost_layer : numpy.ndarray
            Shape (H, W), the C_total layer.

        unknown_mask : numpy.ndarray
            Shape (H, W), bool -- FrontierDetector.detect()["unknown"].

        robot_position : tuple of float
            (row, col), the robot's current position in grid cell
            coordinates.

        Returns
        -------
        dict or None
            The selected cluster (one of refined_clusters, with
            "information_gain", "navigation_cost",
            "navigation_distance" and "utility" keys added), or None
            if refined_clusters is empty.
        """

        if len(refined_clusters) == 0:
            return None

        radius = self.planning_config.information_gain_radius_cells
        nan_penalty = self.planning_config.nan_cost_penalty

        scored = []

        for cluster in refined_clusters:
            centroid = cluster["refined_centroid"]

            gain = self.information_gain(centroid, unknown_mask, radius)

            cost, distance = self.navigation_cost(
                robot_position, centroid, cost_layer, nan_penalty
            )

            scored_cluster = dict(cluster)
            scored_cluster["information_gain"] = gain
            scored_cluster["navigation_cost"] = cost
            scored_cluster["navigation_distance"] = distance

            scored.append(scored_cluster)

        max_gain = max(item["information_gain"] for item in scored)
        max_cost = max(item["navigation_cost"] for item in scored)

        for item in scored:
            normalized_gain = (
                item["information_gain"] / max_gain if max_gain > 0 else 0.0
            )

            normalized_cost = (
                item["navigation_cost"] / max_cost if max_cost > 0 else 0.0
            )

            item["utility"] = (
                self.planning_config.beta_gain * normalized_gain
                - self.planning_config.beta_cost * normalized_cost
            )

        return max(scored, key=lambda item: item["utility"])

    @staticmethod
    def information_gain(centroid, unknown_mask, radius):
        """
        Number of unknown cells within `radius` cells of `centroid`
        (circular window, clipped to the grid).
        """

        height, width = unknown_mask.shape

        row, col = centroid

        row_min = max(0, int(np.floor(row - radius)))
        row_max = min(height, int(np.ceil(row + radius)) + 1)
        col_min = max(0, int(np.floor(col - radius)))
        col_max = min(width, int(np.ceil(col + radius)) + 1)

        if row_min >= row_max or col_min >= col_max:
            return 0

        rows, cols = np.meshgrid(
            np.arange(row_min, row_max),
            np.arange(col_min, col_max),
            indexing="ij"
        )

        distances = np.sqrt((rows - row) ** 2 + (cols - col) ** 2)

        window_unknown = unknown_mask[row_min:row_max, col_min:col_max]

        in_range = window_unknown & (distances <= radius)

        return int(np.sum(in_range))

    @staticmethod
    def navigation_cost(start, end, cost_layer, nan_penalty, num_samples=None):
        """
        Cheap navigation-cost estimate: euclidean distance scaled by
        (1 + average cost sampled along the straight line from start
        to end). NOT a real path search -- see module docstring.

        Returns
        -------
        cost, distance : float, float
        """

        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)

        distance = float(np.linalg.norm(end - start))

        if distance == 0:
            return 0.0, 0.0

        if num_samples is None:
            num_samples = max(2, int(np.ceil(distance)) + 1)

        ts = np.linspace(0.0, 1.0, num_samples)
        points = start[None, :] + ts[:, None] * (end - start)[None, :]

        height, width = cost_layer.shape

        rows = np.clip(np.round(points[:, 0]).astype(int), 0, height - 1)
        cols = np.clip(np.round(points[:, 1]).astype(int), 0, width - 1)

        sampled = cost_layer[rows, cols]
        sampled = np.where(np.isnan(sampled), nan_penalty, sampled)

        average_cost = float(sampled.mean())

        cost = distance * (1.0 + average_cost)

        return cost, distance