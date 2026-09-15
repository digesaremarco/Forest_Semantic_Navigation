"""
Cost fusion: combines the geometric and semantic traversability
costs into a single per-cell cost, via a weighted average.

    C_total(i,j) = w_geo * C_geo(i,j) + w_sem * C_sem(i,j)

with w_geo + w_sem = 1 (costmap_config.yaml, "fusion" section), so
C_total stays in [0, 1] whenever both inputs are.

FALLBACK ON PARTIAL DATA: if only one of the two costs is available
for a cell, C_total falls back to that single cost instead of being
NaN -- equivalent to renormalizing the weights over whichever
factor is actually present.
"""

import numpy as np


class CostFusion:

    def __init__(self, costmap_config):
        self.costmap_config = costmap_config

    def compute(self, geo_cost, sem_cost):
        """
        Fuse the geometric and semantic costs.

        Parameters
        ----------
        geo_cost, sem_cost : numpy.ndarray
            Shape (H, W), float32, in [0, 1]. NaN for cells where
            that cost could not be computed -- i.e.
            GeometricCost.compute(...)["cost"] and
            SemanticCost.compute(...).

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32, in [0, 1]. NaN only where BOTH
            inputs are NaN.
        """

        self.validate_shapes(geo_cost, sem_cost)

        geo_valid = ~np.isnan(geo_cost)
        sem_valid = ~np.isnan(sem_cost)

        both_valid = geo_valid & sem_valid
        only_geo = geo_valid & ~sem_valid
        only_sem = sem_valid & ~geo_valid
        # Cells where neither is valid are left as NaN.

        fused = np.full(geo_cost.shape, np.nan, dtype=np.float32)

        fused[both_valid] = (
            self.costmap_config.weight_geo * geo_cost[both_valid]
            + self.costmap_config.weight_sem * sem_cost[both_valid]
        )

        fused[only_geo] = geo_cost[only_geo]
        fused[only_sem] = sem_cost[only_sem]

        return fused

    @staticmethod
    def validate_shapes(geo_cost, sem_cost):
        if not isinstance(geo_cost, np.ndarray) or geo_cost.ndim != 2:
            raise ValueError(
                "geo_cost must be a 2D numpy.ndarray, got "
                f"{getattr(geo_cost, 'shape', type(geo_cost))}."
            )

        if not isinstance(sem_cost, np.ndarray) or sem_cost.ndim != 2:
            raise ValueError(
                "sem_cost must be a 2D numpy.ndarray, got "
                f"{getattr(sem_cost, 'shape', type(sem_cost))}."
            )

        if geo_cost.shape != sem_cost.shape:
            raise ValueError(
                "geo_cost and sem_cost must have the same shape, "
                f"got {geo_cost.shape} and {sem_cost.shape}."
            )
