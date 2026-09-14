"""
Visualization helpers for GridMap.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


class GridMapVisualizer:

    def __init__(self, grid_map):
        self.grid_map = grid_map

    def plot_elevation_heatmap(
        self,
        ax=None,
        crop_to_observed=True,
        cmap="terrain"
    ):
        """
        Draw the elevation layer as a 2D heatmap, in world-frame
        meters, with unobserved cells left blank (not zero).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into. A new figure/axis is created if None.

        crop_to_observed : bool
            If True (default), crop to the bounding box of observed
            cells instead of the full static grid extent.

        cmap : str
            Matplotlib colormap name.

        Returns
        -------
        (fig, ax)
        """

        self.require_initialized()

        elevation = self.grid_map.get_elevation_layer()

        row_slice, col_slice = self.resolve_crop(crop_to_observed)
        elevation_view = elevation[row_slice, col_slice]
        extent = self.compute_extent(row_slice, col_slice)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        im = ax.imshow(
            np.ma.masked_invalid(elevation_view),
            cmap=cmap,
            origin="lower",
            extent=extent
        )

        ax.set_title("Elevation")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046, label="z [m]")

        return fig, ax

    def plot_semantic_3d(self, ax=None, point_size=2):
        """
        Draw a 3D scatter of observed cells: (x, y) in world-frame
        meters, z = elevation, colored by the per-cell semantic
        color layer.

        Parameters
        ----------
        ax : matplotlib.axes.Axes (with 3D projection), optional
            Axis to draw into. A new 3D figure/axis is created if
            None.

        point_size : float
            Marker size passed to scatter().

        Returns
        -------
        (fig, ax)
        """

        self.require_initialized()

        count = self.grid_map.get_count_layer()
        observed = count > 0

        if not np.any(observed):
            raise RuntimeError(
                "GridMap has no observed cells yet -- nothing to plot."
            )

        elevation = self.grid_map.get_elevation_layer()
        rgb = self.grid_map.get_rgb_layer()

        rows, cols = np.nonzero(observed)

        xs, ys = self.grid_map.cell_to_world(rows, cols)
        zs = elevation[rows, cols]

        colors = np.clip(rgb[rows, cols], 0, 255).astype(np.float32)
        colors = colors / 255.0

        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.scatter(xs, ys, zs, c=colors, s=point_size)

        ax.set_title("Semantic Point Cloud")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        return fig, ax

    def require_initialized(self):
        if not self.grid_map.is_initialized():
            raise RuntimeError(
                "GridMap has not been initialized yet -- call "
                "update() at least once before plotting."
            )

    def resolve_crop(self, crop_to_observed):
        """Return (row_slice, col_slice) to index the layers with."""

        if not crop_to_observed:
            return slice(None), slice(None)

        bounds = self.grid_map.get_observed_bounds()

        if bounds is None:
            return slice(None), slice(None)

        row_min, row_max, col_min, col_max = bounds

        return slice(row_min, row_max + 1), slice(col_min, col_max + 1)

    def compute_extent(self, row_slice, col_slice):
        """
        Compute imshow's (left, right, bottom, top) extent, in
        world-frame meters, for the given row/col slices.
        """

        cell_n = self.grid_map.cell_n

        row_start, row_stop, _ = row_slice.indices(cell_n)
        col_start, col_stop, _ = col_slice.indices(cell_n)

        row_min = row_start
        row_max = row_stop - 1
        col_min = col_start
        col_max = col_stop - 1

        x_min, y_min = self.grid_map.cell_to_world(row_min, col_min)
        x_max, y_max = self.grid_map.cell_to_world(row_max, col_max)

        half_res = self.grid_map.resolution / 2.0

        return (
            x_min - half_res,
            x_max + half_res,
            y_min - half_res,
            y_max + half_res
        )
