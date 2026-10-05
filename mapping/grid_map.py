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
    elevation / elevation_variance / roughness /
    rgb / semantic_probs / count
    (one static grid, aligned layers)

Design choices (see conversation history for the reasoning):

- The grid is STATIC and world-axis-aligned, not robot-centric.
  There is no rolling/shifting buffer: at the very first update()
  call, the received position anchors the grid's center cell once
  and for the whole session. The robot moves through the grid, the
  grid itself never moves. This is deliberately simpler than
  elevation_mapping_cupy's rolling buffer, and is the right choice
  at this scale (a fixed ~20x20 m field-test area), where the
  memory savings of a rolling buffer are not needed.

- Per-point measurement noise depends on the DEPTH z of the point
  along the camera optical axis (points_xyz[:, 2], since points_xyz
  is in the camera optical frame):

      sigma^2(z) = sigma0^2 + alpha * z^p

  Stereo depth error comes from disparity error, so it is a function
  of z, not of the 3D Euclidean distance. With Spot's wide-FOV
  cameras the two differ a lot off-axis (on the test set: p99 depth
  3.7 m vs p99 distance 4.6 m). Using z also keeps the model
  consistent with the point cloud pre-filter, which cuts on z at
  4 m.

  The parameters come from the 'noise' section of
  grid_map_config.yaml (via GridMapConfig) and were calibrated with
  fit_depth_noise.py: per-pixel temporal variance over frames with
  Spot standing still, median per depth bin, fit on the relative
  error. The log-log slope of sigma vs z was 2.01 (stereo theory:
  2.0), confirming p = 4; alpha = 1.7e-5. The fitted sigma0 was ~0;
  the configured sigma0 (3 mm) is a deliberate conservative floor
  for errors the temporal method cannot observe (calibration, pose
  error, systematic bias), which also prevents very close points
  from getting near-infinite weight. Setting alpha = 0 makes every
  point equally weighted and reproduces the plain-mean behavior
  exactly.

  Weights are normalized to the reference depth z_ref:

      w = sigma^2(z_ref) / sigma^2(z)

  so w = 1 at z_ref, and accumulated weights keep the meaning of
  "equivalent number of observations at z_ref".

- Fusion across frames follows the MEM paper's two closed-form,
  "no forgetting" special cases (Sec. III-C):
    * elevation, rgb: Bayesian inference of Gaussians (Eq. 3-7).
      With independent Gaussian measurements of variance sigma_i^2,
      the posterior mean is the inverse-variance weighted mean
      sum(w_i z_i) / sum(w_i), and the posterior variance is
      1 / sum(1 / sigma_i^2) = sigma^2(z_ref) / sum(w_i). Both are
      implemented as running sums divided lazily in the getters
      (avoids incremental floating-point drift).
    * semantic_probs: Dirichlet Bayesian inference (Eq. 8-12) with
      a flat/uninformative prior. The update rule is additive; here
      each point's probability vector is scaled by its weight w, so
      a far, noisy point contributes less evidence than a near one.
      Since each probability vector sums to 1, sum_j alpha_j equals
      sum(w), so dividing by the weight sum yields a normalized
      distribution.
  Exponential averaging (Eq. 2, which deliberately forgets old
  data) is NOT implemented: it's the right choice for dynamic
  scenes or continuous drift, neither of which applies to a single,
  static field-test session.

- The posterior elevation variance is OPTIMISTIC: points from the
  same frame share correlated errors (stereo bias, pose error), so
  they are not truly independent. Treat it as a relative confidence
  measure, not a calibrated one. The separate roughness layer
  (weighted empirical variance of heights inside a cell) measures
  real height spread (grass, rocks), not estimation uncertainty.

- Cells never observed are NaN in every layer, not zero: zero is a
  valid observed value (e.g. z=0, or a class probability of 0), so
  it cannot double as "unobserved". Downstream code (visualization,
  cost) is expected to handle NaN explicitly (e.g. np.isfinite,
  np.ma.masked_invalid), matching the pattern already used in
  test_emc_wrapper.py.
"""

from pathlib import Path

import numpy as np


class GridMap:

    def __init__(self, grid_map_config, class_reducer):
        self.grid_map_config = grid_map_config
        self.class_reducer = class_reducer

        self.cell_n = self.grid_map_config.cell_n
        self.resolution = self.grid_map_config.resolution

        # Depth noise model (see module docstring). Read directly:
        # GridMapConfig always sets these attributes (with a warning
        # and alpha = 0 if the 'noise' section is missing), so a
        # missing attribute here means a wrong config object and
        # should fail loudly, not fall back silently.
        self.noise_sigma0 = float(self.grid_map_config.noise_sigma0)
        self.noise_alpha = float(self.grid_map_config.noise_alpha)
        self.noise_exponent = float(self.grid_map_config.noise_exponent)
        self.noise_reference_distance = float(
            self.grid_map_config.noise_reference_distance
        )

        self.validate_noise_params()

        self.reference_variance = self.point_variance(
            self.noise_reference_distance
        )

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

        self.weight_sum = None
        self.elevation_wsum = None
        self.elevation_wsum2 = None
        self.rgb_wsum = None
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

        # Elevation accumulators in float64: the roughness layer is
        # computed as E[z^2] - E[z]^2, which suffers from
        # cancellation in float32.
        self.weight_sum = np.zeros(shape_2d, dtype=np.float64)
        self.elevation_wsum = np.zeros(shape_2d, dtype=np.float64)
        self.elevation_wsum2 = np.zeros(shape_2d, dtype=np.float64)

        self.rgb_wsum = np.zeros((*shape_2d, 3), dtype=np.float32)

        self.semantic_alpha = np.zeros(
            (*shape_2d, self.num_classes),
            dtype=np.float32
        )

        # Raw number of points per cell (unweighted), kept for
        # diagnostics.
        self.count = np.zeros(shape_2d, dtype=np.float32)

        print(
            f"[GridMap] Initialized {self.cell_n}x{self.cell_n} grid "
            f"({self.grid_map_config.map_length} m, "
            f"{self.resolution} m/cell), "
            f"origin anchored at {self.origin}. "
            f"Noise model: sigma^2 = {self.noise_sigma0}^2 + "
            f"{self.noise_alpha} * z^{self.noise_exponent} "
            f"(z_ref = {self.noise_reference_distance} m)."
        )

    def point_variance(self, depth):
        """
        Measurement variance [m^2] of a point at the given depth z
        [m] along the camera optical axis. Accepts scalars or numpy
        arrays.
        """

        depth = np.asarray(depth, dtype=np.float64)

        return (
            self.noise_sigma0 ** 2
            + self.noise_alpha * depth ** self.noise_exponent
        )

    def compute_point_weights(self, points_xyz):
        """
        Inverse-variance weight of each point, normalized so that a
        point at depth noise_reference_distance has weight 1.

        Parameters
        ----------
        points_xyz : numpy.ndarray
            Shape (N, 3). Points in the camera OPTICAL frame
            (x right, y down, z forward): column 2 is the depth z
            the noise model was calibrated on. Passing points in any
            other frame (e.g. body or world) silently produces wrong
            weights.

        Returns
        -------
        numpy.ndarray
            Shape (N,), float64, strictly positive.
        """

        # Depth along the optical axis, not the 3D distance: see the
        # module docstring. abs() only guards against the sign; the
        # pre-filter already keeps 0 < z < 4 m.
        depth = np.abs(points_xyz[:, 2])

        return self.reference_variance / self.point_variance(depth)

    def update(self, points_xyz, rgb, semantic_probs, R, t):
        """
        Fuse a new semantic point cloud into the grid.

        Parameters
        ----------
        points_xyz : numpy.ndarray
            Shape (N, 3). Point coordinates in the camera optical
            frame (x right, y down, z forward).

        rgb : numpy.ndarray
            Shape (N, 3). Per-point RGB color, 0-255 range.

        semantic_probs : numpy.ndarray
            Shape (N, C). Per-point softmax probability for each of
            the C retained semantic classes, in the same order as
            self.class_names.

        R : numpy.ndarray
            Shape (3, 3). Camera orientation in the world/odometry
            frame.

        t : numpy.ndarray
            Shape (3,). Camera position in the world/odometry frame.
            On the very first call, this position anchors the grid.
        """

        self.validate_points(points_xyz, rgb, semantic_probs)
        self.validate_pose(R, t)

        if not self.is_initialized():
            self.initialize_grid(t)

        # Weights depend on the depth in the camera frame, so they
        # must be computed BEFORE transforming to the world frame.
        weights = self.compute_point_weights(points_xyz)

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

        w = weights[in_bounds]
        z_values = points_world[in_bounds, 2].astype(np.float64)
        rgb_values = rgb[in_bounds].astype(np.float32)
        semantic_values = semantic_probs[in_bounds].astype(np.float32)

        w32 = w.astype(np.float32)[:, None]

        # Flatten (row, col) into a single index so repeated indices
        # (multiple points in the same cell) accumulate correctly
        # via np.add.at, which -- unlike plain fancy-index assignment
        # -- handles duplicate indices by summing, not overwriting.
        flat_idx = row_idx * self.cell_n + col_idx

        np.add.at(self.weight_sum.reshape(-1), flat_idx, w)

        np.add.at(
            self.elevation_wsum.reshape(-1),
            flat_idx,
            w * z_values
        )

        np.add.at(
            self.elevation_wsum2.reshape(-1),
            flat_idx,
            w * z_values * z_values
        )

        np.add.at(
            self.rgb_wsum.reshape(-1, 3),
            flat_idx,
            rgb_values * w32
        )

        np.add.at(
            self.semantic_alpha.reshape(-1, self.num_classes),
            flat_idx,
            semantic_values * w32
        )

        np.add.at(self.count.reshape(-1), flat_idx, 1.0)

    def transform_to_world(self, points_xyz, R, t):
        """Transform points from the camera frame to the world frame."""

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
        Return the elevation layer (inverse-variance weighted mean).

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32. NaN where the cell was never
            observed.
        """

        return self.safe_divide(self.elevation_wsum, self.weight_sum)

    def get_elevation_variance_layer(self):
        """
        Return the posterior variance of the elevation estimate,
        sigma^2(z_ref) / sum(w). Optimistic (see module docstring):
        use it as a relative confidence measure.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32, in m^2. NaN where the cell was
            never observed.
        """

        return self.safe_divide(self.reference_variance, self.weight_sum)

    def get_roughness_layer(self):
        """
        Return the weighted empirical variance of the heights fused
        into each cell, E_w[z^2] - E_w[z]^2. Measures real height
        spread inside the cell (vegetation, stones), not estimation
        uncertainty. Zero for cells with a single point.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32, in m^2. NaN where the cell was
            never observed.
        """

        observed = self.weight_sum > 0

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.elevation_wsum / self.weight_sum
            variance = self.elevation_wsum2 / self.weight_sum - mean ** 2

        # Clip tiny negative values from floating-point cancellation.
        variance = np.where(observed, np.maximum(variance, 0.0), np.nan)

        return variance.astype(np.float32)

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

        return self.safe_divide(self.rgb_wsum, self.weight_sum[..., None])

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
            self.weight_sum
        )

    def get_semantic_probs_layer(self):
        """
        Return the per-cell class probability distribution as one
        stacked array (all channels at once), in the same channel
        order as self.class_names -- the natural input shape for
        SemanticCost.compute().

        Returns
        -------
        numpy.ndarray
            Shape (H, W, C), float32, in [0, 1]. NaN for unobserved
            cells.
        """

        return self.safe_divide(
            self.semantic_alpha, self.weight_sum[..., None]
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
            Shape (H, W), float32. Raw (unweighted) number of points
            fused into each cell so far (0 for never-observed cells).
        """

        return self.count.copy()

    def get_weight_layer(self):
        """
        Return the accumulated weight per cell.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32. Equivalent number of observations
            at depth noise_reference_distance (0 for never-observed
            cells).
        """

        return self.weight_sum.astype(np.float32)

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

    def save(self, path):
        """
        Save the grid's raw state to a compressed .npz file: the
        weighted accumulators plus the origin, the noise model and
        enough metadata (cell_n, resolution, class_names) to validate
        a later load() against a compatible GridMap.

        Saves the RAW accumulators, not the divided-out layers
        (get_elevation_layer() etc.) -- this keeps a loaded map
        mathematically able to keep fusing more frames afterward,
        and avoids baking "NaN for unobserved" into the file (it is
        recomputed from the weights on read instead).
        """

        if not self.is_initialized():
            raise RuntimeError(
                "Cannot save an uninitialized GridMap -- call "
                "update() at least once first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            origin=self.origin,
            weight_sum=self.weight_sum,
            elevation_wsum=self.elevation_wsum,
            elevation_wsum2=self.elevation_wsum2,
            rgb_wsum=self.rgb_wsum,
            semantic_alpha=self.semantic_alpha,
            count=self.count,
            cell_n=np.array(self.cell_n),
            resolution=np.array(self.resolution),
            class_names=np.array(self.class_names),
            noise_params=np.array(self.noise_params())
        )

        print(f"[GridMap] Saved grid state to {path}.")

    def load(self, path):
        """
        Load a previously saved grid state (see save()) into this
        GridMap instance, replacing whatever it currently holds.

        Validates that cell_n, resolution, and the semantic class
        order all match this GridMap's own configuration first --
        loading a map saved with a different grid_map_config or a
        different retained-class set would otherwise silently
        misalign every cell/channel instead of failing loudly.

        Files saved before the noise model existed (unweighted
        accumulators) are still accepted: every old point is treated
        as having weight 1, and the roughness of those cells is
        unknown (set to 0).
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Grid map file not found: {path}")

        data = np.load(path, allow_pickle=False)
        files = set(data.files)

        common_keys = {
            "origin", "semantic_alpha", "count",
            "cell_n", "resolution", "class_names"
        }
        weighted_keys = {
            "weight_sum", "elevation_wsum", "elevation_wsum2", "rgb_wsum"
        }
        legacy_keys = {"elevation_sum", "rgb_sum"}

        is_legacy = not (weighted_keys <= files) and legacy_keys <= files

        expected = common_keys | (legacy_keys if is_legacy else weighted_keys)
        missing = expected - files

        if missing:
            raise ValueError(
                f"Grid map file {path} is missing expected arrays: "
                f"{missing}"
            )

        saved_cell_n = int(data["cell_n"])

        if saved_cell_n != self.cell_n:
            raise ValueError(
                f"Saved grid has cell_n={saved_cell_n}, but this "
                f"GridMap was configured with cell_n={self.cell_n}. "
                "Load into a GridMap built from the same "
                "grid_map_config."
            )

        saved_resolution = float(data["resolution"])

        if abs(saved_resolution - self.resolution) > 1e-9:
            raise ValueError(
                f"Saved grid has resolution={saved_resolution}, but "
                "this GridMap was configured with resolution="
                f"{self.resolution}."
            )

        saved_class_names = [str(name) for name in data["class_names"]]

        if saved_class_names != self.class_names:
            raise ValueError(
                "Saved grid's semantic class order does not match "
                f"this GridMap's: saved={saved_class_names}, "
                f"current={self.class_names}."
            )

        self.origin = data["origin"].astype(np.float32)
        self.semantic_alpha = data["semantic_alpha"].astype(np.float32)
        self.count = data["count"].astype(np.float32)

        if is_legacy:
            self.weight_sum = data["count"].astype(np.float64)
            self.elevation_wsum = data["elevation_sum"].astype(np.float64)

            # Zero spread: E[z^2] = E[z]^2 for every observed cell.
            with np.errstate(invalid="ignore", divide="ignore"):
                self.elevation_wsum2 = np.where(
                    self.weight_sum > 0,
                    self.elevation_wsum ** 2 / self.weight_sum,
                    0.0
                )

            self.rgb_wsum = data["rgb_sum"].astype(np.float32)

            print(
                f"[GridMap] WARNING: {path} uses the legacy unweighted "
                "format; old points were given weight 1 and their "
                "roughness is unknown."
            )
        else:
            self.weight_sum = data["weight_sum"].astype(np.float64)
            self.elevation_wsum = data["elevation_wsum"].astype(np.float64)
            self.elevation_wsum2 = data["elevation_wsum2"].astype(np.float64)
            self.rgb_wsum = data["rgb_wsum"].astype(np.float32)

            if "noise_params" in files:
                saved_noise = data["noise_params"].astype(np.float64)

                if not np.allclose(saved_noise, self.noise_params()):
                    print(
                        "[GridMap] WARNING: saved noise model "
                        f"{saved_noise.tolist()} differs from the "
                        f"current one {self.noise_params()}; further "
                        "fusion will mix inconsistent weights."
                    )

        print(f"[GridMap] Loaded grid state from {path}.")

    def reset(self):
        """Discard all fused data and un-anchor the grid."""

        self.origin = None
        self.weight_sum = None
        self.elevation_wsum = None
        self.elevation_wsum2 = None
        self.rgb_wsum = None
        self.semantic_alpha = None
        self.count = None

    def noise_params(self):
        """Noise model parameters, in a fixed order (for save/load)."""

        return [
            self.noise_sigma0,
            self.noise_alpha,
            self.noise_exponent,
            self.noise_reference_distance
        ]

    def validate_noise_params(self):
        """
        Validate the noise model parameters. GridMapConfig already
        validates them; this is a second check for config objects
        built or modified elsewhere (e.g. the A/B copy in the test).
        """

        if self.noise_sigma0 <= 0:
            raise ValueError(
                "noise_sigma0 must be > 0 (it is the variance floor), "
                f"got {self.noise_sigma0}."
            )

        if self.noise_alpha < 0:
            raise ValueError(
                f"noise_alpha must be >= 0, got {self.noise_alpha}."
            )

        if self.noise_exponent < 0:
            raise ValueError(
                f"noise_exponent must be >= 0, got {self.noise_exponent}."
            )

        if self.noise_reference_distance <= 0:
            raise ValueError(
                "noise_reference_distance must be > 0, got "
                f"{self.noise_reference_distance}."
            )

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

        denominator = np.asarray(denominator)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / denominator

        result = np.where(denominator > 0, result, np.nan)

        return result.astype(np.float32)