"""
Traversability sub-pipeline: builds a semantic point cloud from
RGB+depth, fuses it into a static GridMap, and computes the
geometric/semantic/fused traversability costs on demand.

Wraps PointCloudBuilder, GridMap, GeometricCost, SemanticCost and
CostFusion behind a single update()/get_cost_layer() interface, so
downstream consumers (the planning sub-pipeline, a future top-level
main loop) depend on this one class instead of assembling and
wiring all five components themselves.

This is the traversability-specific slice of the overall system.
Other sub-pipelines (e.g. a future PlanningPipeline) are expected to
live alongside this one under pipeline/, each wrapping one
functional area behind a similarly small interface, with the
eventual main loop composing them.
"""

import numpy as np

from perception.class_reducer import ClassReducer
from fusion.pointcloud_builder import PointCloudBuilder
from mapping.grid_map import GridMap
from costmap.geometric_cost import GeometricCost
from costmap.semantic_cost import SemanticCost
from costmap.cost_fusion import CostFusion


class TraversabilityPipeline:

    def __init__(
        self,
        perception_config,
        camera_calibration,
        grid_map_config,
        costmap_config
    ):
        self.perception_config = perception_config
        self.camera_calibration = camera_calibration
        self.grid_map_config = grid_map_config
        self.costmap_config = costmap_config

        self.class_reducer = ClassReducer(perception_config)
        self.point_cloud_builder = PointCloudBuilder(
            perception_config, camera_calibration
        )

        # Public: downstream code (visualization, diagnostics) is
        # expected to read from this directly, e.g. via
        # GridMapVisualizer(pipeline.grid_map).
        self.grid_map = GridMap(grid_map_config, self.class_reducer)

        self.geometric_cost = GeometricCost(costmap_config)
        self.semantic_cost = SemanticCost(costmap_config, self.class_reducer)
        self.cost_fusion = CostFusion(costmap_config)

    def update(self, rgb, depth, R, t, camera_name):
        """
        Process one frame: run semantic segmentation + point cloud
        construction on (rgb, depth), then fuse the result into the
        grid map at pose (R, t).

        Parameters
        ----------
        rgb : numpy.ndarray
            Shape (H, W, 3), uint8.

        depth : numpy.ndarray
            Shape (H, W). Raw depth image, same convention expected
            by PointCloudBuilder.build().

        R : numpy.ndarray
            Shape (3, 3). Sensor orientation in the world/odom frame.

        t : numpy.ndarray
            Shape (3,). Sensor position in the world/odom frame. On
            the very first call, this position anchors the grid
            (see GridMap).

        camera_name : str
            Which camera (rgb, depth) came from (e.g. "right"),
            looked up in camera_calibration for the right
            intrinsics/depth_scale. Several cameras can share one
            TraversabilityPipeline/GridMap: just call update() once
            per camera per timestep, with that camera's name.
        """

        pc = self.point_cloud_builder.build(rgb, depth, camera_name)

        points_xyz = pc["points_xyz"].astype(np.float32, copy=False)
        semantic_colors = pc["semantic_colors"].astype(np.uint8, copy=False)
        semantic_probs = pc["semantic_probs"].astype(np.float32, copy=False)

        self.grid_map.update(
            points_xyz, semantic_colors, semantic_probs, R, t
        )

    def get_cost_layer(self, kind="total"):
        """
        Compute and return one traversability cost layer, from the
        grid map's CURRENT state (recomputed fresh on every call,
        not cached -- cheap enough at this grid size, and always
        reflects the latest fused data).

        Parameters
        ----------
        kind : str
            One of "geo", "sem", "total".

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32, in [0, 1]. NaN for cells with no
            cost.
        """

        if kind not in ("geo", "sem", "total"):
            raise ValueError(
                f"kind must be one of 'geo', 'sem', 'total', got '{kind}'."
            )

        if kind == "geo":
            return self.compute_geometric_cost()["cost"]

        if kind == "sem":
            return self.compute_semantic_cost()

        geo_cost = self.compute_geometric_cost()["cost"]
        sem_cost = self.compute_semantic_cost()

        return self.cost_fusion.compute(geo_cost, sem_cost)

    def compute_geometric_cost(self):
        """
        Return the full GeometricCost.compute() output (the combined
        cost plus the slope/roughness/step sub-factors), useful for
        diagnostics/visualization beyond what get_cost_layer("geo")
        alone gives you.
        """

        elevation = self.grid_map.get_elevation_layer()

        return self.geometric_cost.compute(
            elevation, resolution=self.grid_map.resolution
        )

    def compute_semantic_cost(self):
        """Return the semantic cost layer alone."""

        semantic_probs_layer = self.grid_map.get_semantic_probs_layer()

        return self.semantic_cost.compute(semantic_probs_layer)

    def reset(self):
        """Discard all fused data and un-anchor the grid."""

        self.grid_map.reset()

    def save_map(self, path):
        """Save the current grid map state to disk (see GridMap.save())."""

        self.grid_map.save(path)

    def load_map(self, path):
        """Load a previously saved grid map state (see GridMap.load())."""

        self.grid_map.load(path)