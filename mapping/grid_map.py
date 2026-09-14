"""
Static semantic elevation grid, fusing the point cloud produced by
PointCloudBuilder (x, y, z, rgb, softmax per class) into a per-cell
representation, following the fusion formulation of Erni et al.,
"MEM: Multi-Modal Elevation Mapping for Robotics and Learning"
(IROS 2023).

Pipeline:

    Semantic point cloud
    (points_xyz, rgb, semantic_probs)
        |
        v
    Sensor pose (R, t)
        |
        v
    GridMap.update()
        |
        v
    elevation / rgb / semantic_probs / count
    (one static grid, 4 aligned layers)

Design choices (see conversation history for the reasoning):

- The grid is STATIC and world-axis-aligned, not robot-centric.
  There is no rolling/shifting buffer: at the very first update()
  call, the received position anchors the grid's center cell once
  and for the whole session. The robot moves through the grid, the
  grid itself never moves. This is deliberately simpler than
  elevation_mapping_cupy's rolling buffer, and is the right choice
  at this scale (a fixed ~20x20 m field-test area), where the
  memory savings of a rolling buffer are not needed.

- Fusion across frames follows the MEM paper's two closed-form,
  "no forgetting" special cases (Sec. III-C):
    * elevation, rgb: Bayesian inference of Gaussians (Eq. 3-7)
      with EQUAL, CONSTANT variance for every point. Under that
      assumption the closed-form posterior mean degenerates to a
      plain running mean weighted by observation count -- so it is
      implemented as a running sum + count, divided lazily in the
      getters (avoids incremental floating-point drift).
    * semantic_probs: Dirichlet Bayesian inference (Eq. 8-12) with
      a flat/uninformative prior. The update rule is additive
      (alpha_j,t = alpha_j,t-1 + sum of per-point probability
      vectors), which is exactly a running sum of the softmax
      channels -- so elevation/rgb and semantic_probs share the
      same accumulation mechanism, only the normalization at
      read-time differs conceptually (it isn't, mathematically:
      both are "sum divided by count").
  Exponential averaging (Eq. 2, which deliberately forgets old
  data) is NOT implemented: it's the right choice for dynamic
  scenes or continuous drift, neither of which applies to a single,
  static field-test session.

- Cells never observed are NaN in every layer, not zero: zero is a
  valid observed value (e.g. z=0, or a class probability of 0), so
  it cannot double as "unobserved". Downstream code (visualization,
  cost) is expected to handle NaN explicitly (e.g. np.isfinite,
  np.ma.masked_invalid), matching the pattern already used in
  test_emc_wrapper.py.
"""

import numpy as np


class GridMap:

    def __init__(self, grid_map_config, class_reducer):
        self.grid_map_config = grid_map_config
        self.class_reducer = class_reducer

        self.cell_n = self.grid_map_config.cell_n
        self.resolution = self.grid_map_config.resolution

        # Semantic channel order, fixed for the lifetime of this
        # GridMap: index i in semantic_probs / semantic_alpha
        # corresponds to self.class_names[i].
        self.class_names = [
            self.class_reducer.get_class_name(class_id)
            for class_id in self.class_reducer.get_kept_classes()
        ]

        self.num_classes = len(self.class_names)

        # Set on the first update() call; None means "not yet
        # initialized", not "at the world origin".
        self.origin = None

        self.elevation_sum = None
        self.rgb_sum = None
        self.semantic_alpha = None
        self.count = None

    def is_initialized(self):
        """Whether the grid has been anchored by a first update()."""

        return self.origin is not None

    def initialize_grid(self, t):
        """
        Anchor the grid's center cell to position t and allocate the
        (empty) layers. Called automatically, once, by update().
        """

        self.origin = np.array(t, dtype=np.float32).copy()

        shape_2d = (self.cell_n, self.cell_n)

        self.elevation_sum = np.zeros(shape_2d, dtype=np.float32)
        self.rgb_sum = np.zeros((*shape_2d, 3), dtype=np.float32)

        self.semantic_alpha = np.zeros(
            (*shape_2d, self.num_classes),
            dtype=np.float32
        )

        self.count = np.zeros(shape_2d, dtype=np.float32)

        print(
            f"[GridMap] Initialized {self.cell_n}x{self.cell_n} grid "
            f"({self.grid_map_config.map_length} m, "
            f"{self.resolution} m/cell), "
            f"origin anchored at {self.origin}."
        )

    def update(self, points_xyz, rgb, semantic_probs, R, t):
        """
        Fuse a new semantic point cloud into the grid.

        Parameters
        ----------
        points_xyz : numpy.ndarray
            Shape (N, 3). Point coordinates in the sensor frame.

        rgb : numpy.ndarray
            Shape (N, 3). Per-point RGB color, 0-255 range.

        semantic_probs : numpy.ndarray
            Shape (N, C). Per-point softmax probability for each of
            the C retained semantic classes, in the same order as
            self.class_names.

        R : numpy.ndarray
            Shape (3, 3). Sensor orientation in the world/odometry
            frame.

        t : numpy.ndarray
            Shape (3,). Sensor position in the world/odometry frame.
            On the very first call, this position anchors the grid.
        """

        self.validate_points(points_xyz, rgb, semantic_probs)
        self.validate_pose(R, t)

        if not self.is_initialized():
            self.initialize_grid(t)

        points_world = self.transform_to_world(points_xyz, R, t)

        row_idx, col_idx, in_bounds = self.compute_cell_indices(
            points_world
        )

        n_dropped = int(np.count_nonzero(~in_bounds))

        if n_dropped > 0 and self.grid_map_config.out_of_bounds_warning:
            print(
                f"[GridMap] WARNING: {n_dropped} point(s) fell "
                f"outside the {self.grid_map_config.map_length} m "
                "grid and were dropped."
            )

        row_idx = row_idx[in_bounds]
        col_idx = col_idx[in_bounds]

        if row_idx.size == 0:
            return

        z_values = points_world[in_bounds, 2]
        rgb_values = rgb[in_bounds].astype(np.float32)
        semantic_values = semantic_probs[in_bounds]

        # Flatten (row, col) into a single index so repeated indices
        # (multiple points in the same cell) accumulate correctly
        # via np.add.at, which -- unlike plain fancy-index assignment
        # -- handles duplicate indices by summing, not overwriting.
        flat_idx = row_idx * self.cell_n + col_idx

        np.add.at(
            self.elevation_sum.reshape(-1),
            flat_idx,
            z_values
        )

        np.add.at(
            self.rgb_sum.reshape(-1, 3),
            flat_idx,
            rgb_values
        )

        np.add.at(
            self.semantic_alpha.reshape(-1, self.num_classes),
            flat_idx,
            semantic_values
        )

        np.add.at(
            self.count.reshape(-1),
            flat_idx,
            1.0
        )

    def transform_to_world(self, points_xyz, R, t):
        """Transform points from the sensor frame to the world frame."""

        return points_xyz @ R.T + t

    def compute_cell_indices(self, points_world):
        """
        Map world-frame (x, y) coordinates to (row, col) grid
        indices, relative to the anchored origin.

        Returns
        -------
        row_idx, col_idx : numpy.ndarray
            Shape (N,) each, int64. May contain out-of-range values;
            see in_bounds.

        in_bounds : numpy.ndarray
            Shape (N,), bool. True where (row_idx, col_idx) falls
            inside the grid.
        """

        half = self.cell_n // 2

        dx = points_world[:, 0] - self.origin[0]
        dy = points_world[:, 1] - self.origin[1]

        col_idx = np.floor(dx / self.resolution).astype(np.int64) + half
        row_idx = np.floor(dy / self.resolution).astype(np.int64) + half

        in_bounds = (
            (row_idx >= 0) & (row_idx < self.cell_n) &
            (col_idx >= 0) & (col_idx < self.cell_n)
        )

        return row_idx, col_idx, in_bounds

    def get_elevation_layer(self):
        """
        Return the elevation layer as a numpy array.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32. NaN where the cell was never
            observed.
        """

        return self.safe_divide(self.elevation_sum, self.count)

    def get_rgb_layer(self):
        """
        Return the RGB layer as a numpy array.

        Returns
        -------
        numpy.ndarray
            Shape (H, W, 3), float32 in [0, 255]. NaN where the cell
            was never observed. Left as float (not uint8) so NaN can
            represent "unobserved"; round/cast at display time.
        """

        return self.safe_divide(self.rgb_sum, self.count[..., None])

    def get_semantic_layer(self, class_name):
        """
        Return a single semantic class probability layer.

        Parameters
        ----------
        class_name : str
            Name of a retained semantic class
            (e.g. "tree", "rock", "water").

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32 in [0, 1]. NaN where the cell was
            never observed.
        """

        if class_name not in self.class_names:
            raise ValueError(
                f"Unknown semantic class: {class_name}. "
                f"Available classes: {self.class_names}"
            )

        class_idx = self.class_names.index(class_name)

        return self.safe_divide(
            self.semantic_alpha[..., class_idx],
            self.count
        )

    def get_all_semantic_layers(self):
        """
        Return every semantic class layer as a dictionary.

        Returns
        -------
        dict
            Maps each semantic class name to its (H, W) numpy array.
        """

        return {
            class_name: self.get_semantic_layer(class_name)
            for class_name in self.class_names
        }

    def get_count_layer(self):
        """
        Return the observation-count layer.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32. Number of points fused into each
            cell so far (0 for never-observed cells).
        """

        return self.count.copy()

    def get_origin(self):
        """
        Return the world position the grid's center cell is anchored
        to, or None if update() has not been called yet.
        """

        if self.origin is None:
            return None

        return self.origin.copy()

    def cell_to_world(self, row, col):
        """
        Convert (row, col) grid indices to (x, y) world-frame
        coordinates of the cell center. Inverse of the mapping used
        by compute_cell_indices(). Accepts scalars or numpy arrays.

        Returns
        -------
        x, y : matching the input's shape (scalar or array)
        """

        if not self.is_initialized():
            raise RuntimeError(
                "GridMap has not been initialized yet -- call "
                "update() at least once before converting cell "
                "indices to world coordinates."
            )

        half = self.cell_n // 2

        x = (np.asarray(col) - half) * self.resolution + self.origin[0]
        y = (np.asarray(row) - half) * self.resolution + self.origin[1]

        return x, y

    def get_observed_bounds(self):
        """
        Return the (row_min, row_max, col_min, col_max) bounding box
        (inclusive) of cells with at least one observation.

        Returns
        -------
        tuple of int, or None
            None if the grid is not initialized yet, or has no
            observed cells (e.g. every point so far fell out of
            bounds).
        """

        if not self.is_initialized():
            return None

        rows, cols = np.nonzero(self.count > 0)

        if rows.size == 0:
            return None

        return (
            int(rows.min()), int(rows.max()),
            int(cols.min()), int(cols.max())
        )

    def reset(self):
        """Discard all fused data and un-anchor the grid."""

        self.origin = None
        self.elevation_sum = None
        self.rgb_sum = None
        self.semantic_alpha = None
        self.count = None

    def validate_points(self, points_xyz, rgb, semantic_probs):
        """Validate the point cloud arrays passed to update()."""

        if (
            not isinstance(points_xyz, np.ndarray)
            or points_xyz.ndim != 2
            or points_xyz.shape[1] != 3
        ):
            raise ValueError(
                "points_xyz must have shape (N, 3), got "
                f"{getattr(points_xyz, 'shape', type(points_xyz))}."
            )

        if (
            not isinstance(rgb, np.ndarray)
            or rgb.ndim != 2
            or rgb.shape[1] != 3
        ):
            raise ValueError(
                "rgb must have shape (N, 3), got "
                f"{getattr(rgb, 'shape', type(rgb))}."
            )

        if (
            not isinstance(semantic_probs, np.ndarray)
            or semantic_probs.ndim != 2
        ):
            raise ValueError(
                "semantic_probs must have shape (N, C), got "
                f"{getattr(semantic_probs, 'shape', type(semantic_probs))}."
            )

        n = points_xyz.shape[0]

        if rgb.shape[0] != n or semantic_probs.shape[0] != n:
            raise ValueError(
                "points_xyz, rgb and semantic_probs must share the "
                f"same number of points: {points_xyz.shape[0]}, "
                f"{rgb.shape[0]}, {semantic_probs.shape[0]}."
            )

        if semantic_probs.shape[1] != self.num_classes:
            raise ValueError(
                f"semantic_probs must have {self.num_classes} "
                f"channels (one per retained class), got "
                f"{semantic_probs.shape[1]}."
            )

    @staticmethod
    def validate_pose(R, t):
        """Validate the pose arrays passed to update()."""

        if not isinstance(R, np.ndarray) or R.shape != (3, 3):
            raise ValueError(
                "R must be a numpy.ndarray with shape (3, 3), got "
                f"{getattr(R, 'shape', type(R))}."
            )

        if not isinstance(t, np.ndarray) or t.shape != (3,):
            raise ValueError(
                "t must be a numpy.ndarray with shape (3,), got "
                f"{getattr(t, 'shape', type(t))}."
            )

    @staticmethod
    def safe_divide(numerator, denominator):
        """
        Divide numerator by denominator, returning NaN wherever
        denominator is zero instead of raising or returning inf.
        """

        denominator = np.asarray(denominator, dtype=np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / denominator

        result = np.where(denominator > 0, result, np.nan)

        return result.astype(np.float32)