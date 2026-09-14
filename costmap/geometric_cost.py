"""
Geometric traversability cost.

Computes C_geo(i,j), the geometric component of the traversability
cost, as a weighted sum of three factors derived from GridMap's
elevation layer: slope, roughness, and step height.

Cost convention: every returned layer is in [0, 1], 0 = fully
traversable, 1 = maximally costly. A cell is NaN in every output
layer whenever its own elevation is NaN (never observed) -- "no
information" is never silently treated as "safe" (0) or "unsafe" (1).

"""

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class GeometricCost:

    def __init__(self, costmap_config):
        self.costmap_config = costmap_config

        self.theta_max = np.radians(costmap_config.theta_max_deg)
        self.sigma_max = costmap_config.sigma_max
        self.s_max = costmap_config.s_max
        self.roughness_window = costmap_config.roughness_window

    def compute(self, elevation, resolution):
        """
        Compute the geometric traversability cost.

        Parameters
        ----------
        elevation : numpy.ndarray
            Shape (H, W), float32. NaN for unobserved cells (as
            returned by GridMap.get_elevation_layer()).

        resolution : float
            Grid cell size, in meters.

        Returns
        -------
        dict of numpy.ndarray, each shape (H, W), float32:
            "cost": the combined C_geo.
            "slope_cost", "roughness_cost", "step_cost": the three
            individual factors, useful for diagnostics/tuning.
        """

        self.validate_elevation(elevation)

        slope_cost = self.compute_slope_cost(elevation, resolution)
        roughness_cost = self.compute_roughness_cost(elevation)
        step_cost = self.compute_step_cost(elevation)

        cost = (
            self.costmap_config.weight_slope * slope_cost
            + self.costmap_config.weight_roughness * roughness_cost
            + self.costmap_config.weight_step * step_cost
        )

        # A cell never observed carries no information: every layer
        # is forced to NaN there, regardless of what neighboring
        # cells alone might have let a formula produce.
        unobserved = np.isnan(elevation)

        slope_cost = np.where(unobserved, np.nan, slope_cost)
        roughness_cost = np.where(unobserved, np.nan, roughness_cost)
        step_cost = np.where(unobserved, np.nan, step_cost)
        cost = np.where(unobserved, np.nan, cost)

        return {
            "cost": cost.astype(np.float32),
            "slope_cost": slope_cost.astype(np.float32),
            "roughness_cost": roughness_cost.astype(np.float32),
            "step_cost": step_cost.astype(np.float32)
        }

    def compute_slope_cost(self, elevation, resolution):
        """
        Slope factor c_slope, from the elevation gradient via
        central differences. NaN wherever any of the four immediate
        neighbors is missing -- a proper central difference needs
        both opposite neighbors, there's no meaningful partial
        version of it.
        """

        padded = np.pad(
            elevation, 1, mode="constant", constant_values=np.nan
        )

        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]

        with np.errstate(invalid="ignore"):
            dz_dx = (right - left) / (2 * resolution)
            dz_dy = (down - up) / (2 * resolution)

            theta = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

            cost = np.clip(theta / self.theta_max, 0.0, 1.0)

        return cost

    def compute_roughness_cost(self, elevation):
        """
        Roughness factor c_rough, from the standard deviation of
        elevation within a local window. NaN wherever fewer than 2
        cells in the window are observed -- a single sample gives a
        degenerate std of exactly 0, which would misleadingly read
        as "perfectly smooth" instead of "not enough information".
        """

        half = self.roughness_window // 2

        padded = np.pad(
            elevation, half, mode="constant", constant_values=np.nan
        )

        windows = sliding_window_view(
            padded, (self.roughness_window, self.roughness_window)
        )

        with warnings.catch_warnings():
            # nanstd warns on all-NaN windows; NaN is exactly the
            # correct, intended result there, not an error.
            warnings.simplefilter("ignore", category=RuntimeWarning)

            sigma = np.nanstd(windows, axis=(-2, -1))
            valid_count = np.sum(~np.isnan(windows), axis=(-2, -1))

        sigma = np.where(valid_count >= 2, sigma, np.nan)

        with np.errstate(invalid="ignore"):
            cost = np.clip(sigma / self.sigma_max, 0.0, 1.0)

        return cost

    def compute_step_cost(self, elevation):
        """
        Step factor c_step, from the maximum absolute elevation
        difference with the 4-connected immediate neighbors. Unlike
        slope, this uses whichever neighbors ARE observed (does not
        require all four): a discontinuity revealed by even a single
        neighbor is still a meaningful signal, unlike a symmetric
        derivative which genuinely needs both opposite neighbors to
        mean anything.
        """

        padded = np.pad(
            elevation, 1, mode="constant", constant_values=np.nan
        )

        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]

        with np.errstate(invalid="ignore"):
            diffs = np.stack([
                np.abs(elevation - left),
                np.abs(elevation - right),
                np.abs(elevation - up),
                np.abs(elevation - down)
            ], axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            step = np.nanmax(diffs, axis=0)

        with np.errstate(invalid="ignore"):
            cost = np.clip(step / self.s_max, 0.0, 1.0)

        return cost

    @staticmethod
    def validate_elevation(elevation):
        if not isinstance(elevation, np.ndarray) or elevation.ndim != 2:
            raise ValueError(
                "elevation must be a 2D numpy.ndarray, got "
                f"{getattr(elevation, 'shape', type(elevation))}."
            )
