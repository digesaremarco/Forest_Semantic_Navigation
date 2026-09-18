"""
Path smoothing: simplifies an A* path via greedy line-of-sight
shortcutting ("string pulling"), while making sure a shortcut never
makes the path materially worse than what A* actually found.

For each point on the path, the algorithm tries to connect directly
to the FARTHEST point still reachable by straight line, skipping
everything in between, then repeats from there -- the standard way
to turn an A* grid path's staircase pattern into a more natural
route.

A straight-line shortcut is only accepted if BOTH:
    - it does not cross any occupied cell, and
    - its average traversability cost is not more than
      (1 + cost_increase_tolerance) times the average cost of the
      ORIGINAL A* path segment it would replace.

The second condition is the one worth explaining: a purely
geometric smoother would happily cut a corner through a "free but
costly" region (e.g. dense bush, cost 0.5, well under
occupied_threshold) that A* had deliberately routed around because
it was meaningfully worse than the alternative -- passing the
"not occupied" check alone is not enough to guarantee the shortcut
is actually a good idea. Comparing against the original segment's
own cost, rather than a fixed absolute threshold, keeps the
criterion adaptive to whatever cost level the path is already
moving through.

NaN (unobserved) cells, sampled along a candidate line or found in
the original path, are substituted with nan_cost_penalty -- the
same value PathPlanner itself already used to cost them, so the
comparison stays apples-to-apples with what A* actually optimized.
"""

import numpy as np


class PathSmoother:

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def smooth(self, path, cost_layer, occupied_mask):
        """
        Parameters
        ----------
        path : list of (row, col) int tuples, or None
            As returned by PathPlanner.plan().

        cost_layer : numpy.ndarray
            Shape (H, W), the C_total layer.

        occupied_mask : numpy.ndarray
            Shape (H, W), bool -- FrontierDetector.detect()["occupied"].

        Returns
        -------
        list of (row, col) int tuples, or None if `path` was None.
        A subsequence of the original path -- no new points are
        introduced, only intermediate ones are dropped.
        """

        if path is None:
            return None

        if len(path) <= 2:
            return list(path)

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Always valid at minimum: A* already walked this exact
            # edge, so falling back to it can never fail.
            farthest_idx = current_idx + 1

            for candidate_idx in range(len(path) - 1, current_idx, -1):
                if self.is_shortcut_valid(
                    path, current_idx, candidate_idx,
                    cost_layer, occupied_mask
                ):
                    farthest_idx = candidate_idx
                    break

            smoothed.append(path[farthest_idx])
            current_idx = farthest_idx

        return smoothed

    def is_shortcut_valid(
        self, path, start_idx, end_idx, cost_layer, occupied_mask
    ):
        """
        Whether a straight-line shortcut from path[start_idx] to
        path[end_idx] is acceptable: no occupied cell crossed, and
        not meaningfully costlier than the original path segment it
        would replace.
        """

        start = path[start_idx]
        end = path[end_idx]

        line_cells = self.sample_line_cells(start, end, cost_layer.shape)

        if any(occupied_mask[row, col] for row, col in line_cells):
            return False

        nan_penalty = self.planning_config.nan_cost_penalty

        line_cost = self.average_cost(line_cells, cost_layer, nan_penalty)

        original_segment = path[start_idx:end_idx + 1]

        original_cost = self.average_cost(
            original_segment, cost_layer, nan_penalty
        )

        tolerance = self.planning_config.cost_increase_tolerance

        return line_cost <= original_cost * (1.0 + tolerance)

    @staticmethod
    def sample_line_cells(start, end, shape, num_samples=None):
        """
        Integer grid cells along the straight line from start to end
        (inclusive), via linear interpolation + rounding.
        """

        height, width = shape

        start_arr = np.array(start, dtype=np.float64)
        end_arr = np.array(end, dtype=np.float64)

        distance = np.linalg.norm(end_arr - start_arr)

        if num_samples is None:
            num_samples = max(2, int(np.ceil(distance)) + 1)

        ts = np.linspace(0.0, 1.0, num_samples)

        points = (
            start_arr[None, :]
            + ts[:, None] * (end_arr - start_arr)[None, :]
        )

        rows = np.clip(np.round(points[:, 0]).astype(int), 0, height - 1)
        cols = np.clip(np.round(points[:, 1]).astype(int), 0, width - 1)

        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def average_cost(cells, cost_layer, nan_penalty):
        values = [cost_layer[row, col] for row, col in cells]
        values = [nan_penalty if np.isnan(v) else v for v in values]

        return float(np.mean(values))