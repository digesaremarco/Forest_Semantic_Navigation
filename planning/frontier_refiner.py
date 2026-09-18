"""
Frontier refinement: shifts each frontier cluster's raw centroid
(from FrontierDetector) toward a nearby lower-cost area, via an
iterative mean-shift over free cells within a local window.

For a candidate position (row, col), each free cell f within
mean_shift_bandwidth_cells is weighted by:

    weight(f) = gaussian_kernel(distance(f, position), bandwidth)
                * (1 - cost(f))

i.e. closer AND cheaper cells pull the centroid harder. The position
is moved to the weighted mean of its neighborhood, repeated until
convergence (movement below a small fixed threshold) or a fixed
maximum number of iterations -- these two numbers are numerical
details of the local hill-climb, not exposed as config, unlike
bandwidth/max_shift which genuinely change the qualitative behavior.

The total displacement from the ORIGINAL raw centroid is then capped
at max_shift_cells: this keeps the mean-shift from drifting the
target so far into a comfortable, low-cost area that it stops
representing the direction of the frontier it came from.

If a candidate position's neighborhood contains no free cells at
some iteration (e.g. right at the edge of the valid region), the
refinement simply stops there and returns the last valid position --
there is nothing nearby to shift toward.
"""

import numpy as np


class FrontierRefiner:

    MAX_ITERATIONS = 10
    CONVERGENCE_THRESHOLD_CELLS = 0.05

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def refine(self, clusters, cost_layer, free_mask):
        """
        Parameters
        ----------
        clusters : list of dict
            As returned by FrontierDetector.detect()["clusters"];
            each needs at least a "centroid" (row, col) float tuple.

        cost_layer : numpy.ndarray
            Shape (H, W), the same C_total layer the clusters were
            detected from.

        free_mask : numpy.ndarray
            Shape (H, W), bool -- FrontierDetector.detect()["free"].

        Returns
        -------
        list of dict
            The same clusters, each with an added
            "refined_centroid" (row, col) float tuple. All original
            keys are preserved.
        """

        refined_clusters = []

        for cluster in clusters:
            refined_centroid = self.mean_shift(
                cluster["centroid"], cost_layer, free_mask
            )

            refined_cluster = dict(cluster)
            refined_cluster["refined_centroid"] = refined_centroid

            refined_clusters.append(refined_cluster)

        return refined_clusters

    def mean_shift(self, start_centroid, cost_layer, free_mask):
        """Iteratively shift one centroid toward nearby low-cost area."""

        bandwidth = self.planning_config.mean_shift_bandwidth_cells
        max_shift = self.planning_config.max_shift_cells

        position = np.array(start_centroid, dtype=np.float64)
        origin = position.copy()

        for _ in range(self.MAX_ITERATIONS):
            new_position = self.weighted_mean_neighborhood(
                position, cost_layer, free_mask, bandwidth
            )

            if new_position is None:
                # No free cells in range -- nothing to shift toward,
                # keep the current position.
                break

            movement = np.linalg.norm(new_position - position)
            position = new_position

            if movement < self.CONVERGENCE_THRESHOLD_CELLS:
                break

        position = self.clamp_shift(origin, position, max_shift)

        return (float(position[0]), float(position[1]))

    @staticmethod
    def weighted_mean_neighborhood(position, cost_layer, free_mask, bandwidth):
        """
        Weighted mean position of free cells within `bandwidth` cells
        of `position`, weighted by a Gaussian distance kernel times
        (1 - cost). Returns None if no free cell is in range.
        """

        height, width = cost_layer.shape

        row, col = position

        row_min = max(0, int(np.floor(row - bandwidth)))
        row_max = min(height, int(np.ceil(row + bandwidth)) + 1)
        col_min = max(0, int(np.floor(col - bandwidth)))
        col_max = min(width, int(np.ceil(col + bandwidth)) + 1)

        if row_min >= row_max or col_min >= col_max:
            return None

        rows, cols = np.meshgrid(
            np.arange(row_min, row_max),
            np.arange(col_min, col_max),
            indexing="ij"
        )

        distances = np.sqrt((rows - row) ** 2 + (cols - col) ** 2)

        window_free = free_mask[row_min:row_max, col_min:col_max]
        window_cost = cost_layer[row_min:row_max, col_min:col_max]

        in_range = window_free & (distances <= bandwidth)

        if not np.any(in_range):
            return None

        kernel = np.exp(-0.5 * (distances[in_range] / bandwidth) ** 2)
        value_weight = 1.0 - window_cost[in_range]

        weights = kernel * value_weight

        total_weight = weights.sum()

        if total_weight <= 0:
            return None

        new_row = np.sum(weights * rows[in_range]) / total_weight
        new_col = np.sum(weights * cols[in_range]) / total_weight

        return np.array([new_row, new_col], dtype=np.float64)

    @staticmethod
    def clamp_shift(origin, position, max_shift):
        """Cap the displacement from `origin` at `max_shift` cells."""

        displacement = position - origin
        distance = np.linalg.norm(displacement)

        if distance <= max_shift or distance == 0:
            return position

        return origin + displacement * (max_shift / distance)