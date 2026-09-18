"""
Frontier detection: classifies grid cells into unknown/free/occupied
from the fused traversability cost, then finds and clusters frontier
cells -- unknown cells adjacent to free cells (Yamauchi, 1997) --
into candidate exploration targets.

Classification (on C_total, in [0, 1]):
    unknown  = NaN (never observed by GridMap)
    occupied = known and cost >= occupied_threshold
    free     = known and cost <  occupied_threshold

Frontier cells are unknown cells with at least one 4-connected free
neighbor -- 4-connectivity here matches the neighbor convention
already used for GeometricCost's step factor. Frontier CELLS are
then grouped into candidate CLUSTERS using 8-connectivity (diagonal
neighbors count as connected), the more common choice for this step
in the literature: it merges frontier cells along a diagonal "wall"
between known and unknown space into one contiguous region instead
of splintering it into many single-cell clusters.

Cells within border_margin_cells of the STATIC grid's own edge are
excluded from frontier detection entirely: the grid boundary is a
limit of the data structure (GridMap has a fixed extent), not a real
unexplored boundary of the world, and treating it as a frontier
would repeatedly pull the robot toward the area where GridMap
already starts dropping out-of-bounds points.
"""

import numpy as np
from scipy import ndimage


class FrontierDetector:

    def __init__(self, planning_config):
        self.planning_config = planning_config

    def detect(self, cost_layer):
        """
        Parameters
        ----------
        cost_layer : numpy.ndarray
            Shape (H, W), float32, in [0, 1]. NaN for unobserved
            cells -- typically
            TraversabilityPipeline.get_cost_layer("total").

        Returns
        -------
        dict with keys:
            "clusters" : list of dict, one per surviving frontier
                cluster, each with:
                    "rows", "cols" : numpy.ndarray of cell indices
                        belonging to the cluster.
                    "centroid" : (row, col) float tuple, the raw
                        (un-refined) centroid of the cluster.
                    "size" : int, number of cells in the cluster.
            "unknown", "free", "occupied", "frontier_mask" : the
                underlying (H, W) boolean layers, for diagnostics or
                visualization.
        """

        self.validate_cost_layer(cost_layer)

        unknown, free, occupied = self.classify(cost_layer)

        valid_region = self.build_valid_region(cost_layer.shape)

        frontier_mask = self.find_frontier_cells(unknown, free, valid_region)

        clusters = self.cluster_frontier_cells(frontier_mask)

        return {
            "clusters": clusters,
            "unknown": unknown,
            "free": free,
            "occupied": occupied,
            "frontier_mask": frontier_mask
        }

    def classify(self, cost_layer):
        """Split cost_layer into unknown / free / occupied boolean masks."""

        unknown = np.isnan(cost_layer)
        known = ~unknown

        occupied = known & (
            cost_layer >= self.planning_config.occupied_threshold
        )

        free = known & (
            cost_layer < self.planning_config.occupied_threshold
        )

        return unknown, free, occupied

    def build_valid_region(self, shape):
        """
        Boolean mask, True everywhere except within
        border_margin_cells of the grid's own edge.
        """

        height, width = shape
        margin = self.planning_config.border_margin_cells

        if 2 * margin >= height or 2 * margin >= width:
            raise ValueError(
                f"border_margin_cells={margin} leaves no valid "
                f"region on a {height}x{width} grid."
            )

        valid_region = np.zeros(shape, dtype=bool)
        valid_region[margin:height - margin, margin:width - margin] = True

        return valid_region

    @staticmethod
    def find_frontier_cells(unknown, free, valid_region):
        """
        Unknown cells with at least one 4-connected free neighbor,
        restricted to valid_region.
        """

        free_padded = np.pad(
            free, 1, mode="constant", constant_values=False
        )

        left = free_padded[1:-1, :-2]
        right = free_padded[1:-1, 2:]
        up = free_padded[:-2, 1:-1]
        down = free_padded[2:, 1:-1]

        has_free_neighbor = left | right | up | down

        return unknown & has_free_neighbor & valid_region

    def cluster_frontier_cells(self, frontier_mask):
        """
        Group frontier cells into connected clusters (8-connectivity),
        discarding clusters smaller than min_cluster_size.
        """

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity

        labeled, num_clusters = ndimage.label(
            frontier_mask, structure=structure
        )

        clusters = []

        for label in range(1, num_clusters + 1):
            rows, cols = np.nonzero(labeled == label)

            if rows.size < self.planning_config.min_cluster_size:
                continue

            centroid = (float(rows.mean()), float(cols.mean()))

            clusters.append({
                "rows": rows,
                "cols": cols,
                "centroid": centroid,
                "size": int(rows.size)
            })

        return clusters

    @staticmethod
    def validate_cost_layer(cost_layer):
        if not isinstance(cost_layer, np.ndarray) or cost_layer.ndim != 2:
            raise ValueError(
                "cost_layer must be a 2D numpy.ndarray, got "
                f"{getattr(cost_layer, 'shape', type(cost_layer))}."
            )