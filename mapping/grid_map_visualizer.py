"""
Visualization helpers for GridMap.

Semantic plots color each cell by its WINNING class (argmax of the
fused per-cell class probabilities) and carry a legend with one entry
per class present. The rgb layer (running mean of palette colors) is
still available via color_by="rgb", but mixed cells then get blended
colors matching no class, so no legend is drawn in that mode.
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

    def plot_semantic_topdown(
        self,
        ax=None,
        crop_to_observed=True,
        background="white",
        color_by="class",
        class_colors=None,
        show_counts=True
    ):
        """
        Draw the GridMap seen from above (bird's-eye view) as a 2D
        image, in world-frame meters. Unobserved cells are left
        transparent so the axis background shows through.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw into. A new figure/axis is created if None.

        crop_to_observed : bool
            If True (default), crop to the bounding box of observed
            cells instead of the full static grid extent.

        background : str or None
            Axis facecolor shown behind unobserved cells. None keeps
            Matplotlib's default.

        color_by : {"class", "rgb"}
            "class" (default): winning class color, with a legend.
            "rgb": mean palette color of the cell, no legend.

        class_colors : dict, optional
            Per-class color overrides, {class_name: (r, g, b)} in
            [0, 255] (color_by="class" only).

        show_counts : bool
            Append the number of cells to each legend entry.

        Returns
        -------
        (fig, ax)
        """

        self.require_initialized()

        observed = self.grid_map.get_count_layer() > 0

        if not np.any(observed):
            raise RuntimeError(
                "GridMap has no observed cells yet -- nothing to plot."
            )

        row_slice, col_slice = self.resolve_crop(crop_to_observed)
        observed_view = observed[row_slice, col_slice]
        extent = self.compute_extent(row_slice, col_slice)

        # RGBA image: colors in [0, 1], alpha = 0 where unobserved.
        rgba = np.zeros(observed_view.shape + (4,), dtype=np.float32)
        rgba[..., 3] = observed_view.astype(np.float32)

        labels = None

        if color_by == "class":
            palette = self.get_class_palette(class_colors)
            labels = self.get_winning_classes(observed_view, row_slice, col_slice)[0]
            rgba[observed_view, :3] = palette[labels] / 255.0
        elif color_by == "rgb":
            rgb = self.grid_map.get_rgb_layer()[row_slice, col_slice]
            colors = np.nan_to_num(rgb.astype(np.float32), nan=0.0)
            rgba[..., :3] = np.clip(colors, 0, 255) / 255.0
        else:
            raise ValueError(
                f"color_by must be 'class' or 'rgb', got {color_by!r}."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure

        if background is not None:
            ax.set_facecolor(background)

        ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            interpolation="nearest"
        )

        ax.set_title("Semantic map (top-down)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

        if labels is not None:
            self.add_class_legend(ax, labels, palette, show_counts)

        return fig, ax

    def plot_semantic_3d_plotly(
        self,
        class_colors=None,
        point_size=2,
        opacity=1.0,
        equal_aspect=True,
        show_counts=True,
        title="Semantic Point Cloud"
    ):
        """
        Interactive Plotly 3D scatter of observed cells: (x, y) in
        world-frame meters, z = elevation, each cell colored by its
        WINNING semantic class (argmax of the fused per-cell class
        probabilities), with one legend entry per class.

        Unlike plot_semantic_3d(), this does NOT use the rgb layer:
        that layer is a running mean of palette colors, so mixed
        cells get blended colors matching no class. The argmax of
        the fused Dirichlet probabilities is the cell's actual
        most likely class.

        Parameters
        ----------
        class_colors : dict, optional
            Per-class color OVERRIDES, {class_name: (r, g, b)} with
            r, g, b in [0, 255]. By default every class uses the
            segmentation palette from
            ClassReducer.get_filtered_class_colors(), so colors
            match the segmentation masks exactly.

        point_size : float
            Marker size in pixels.

        opacity : float
            Marker opacity in [0, 1].

        equal_aspect : bool
            If True (default), use the same scale on x, y, z
            (aspectmode="data"), so z isn't visually stretched.

        show_counts : bool
            If True (default), append the number of cells to each
            legend entry, e.g. "tree (1234)".

        title : str
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Not shown automatically: call fig.show() or
            fig.write_html(...).
        """

        # Lazy import: plotly is only required if this method is used.
        import plotly.graph_objects as go

        self.require_initialized()

        count = self.grid_map.get_count_layer()
        elevation = self.grid_map.get_elevation_layer()

        # Observed AND with a valid elevation (NaN z breaks the scene).
        valid = (count > 0) & np.isfinite(elevation)

        if not np.any(valid):
            raise RuntimeError(
                "GridMap has no observed cells yet -- nothing to plot."
            )

        class_names = self.grid_map.class_names

        rows, cols = np.nonzero(valid)

        xs, ys = self.grid_map.cell_to_world(rows, cols)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = elevation[rows, cols]
        obs = count[rows, cols]

        # Winning class per cell and its fused probability.
        labels, confidence = self.get_winning_classes(valid)

        palette = self.get_class_palette(class_colors)

        trace_colors = [
            f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in palette
        ]

        # ---------------------------------------------------------
        # One trace per class -> one legend entry per class
        # ---------------------------------------------------------

        fig = go.Figure()

        for k, name in enumerate(class_names):
            mask = labels == k
            n = int(mask.sum())

            if n == 0:
                continue

            label = f"{name} ({n})" if show_counts else name

            fig.add_trace(
                go.Scatter3d(
                    x=xs[mask],
                    y=ys[mask],
                    z=zs[mask],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=point_size,
                        color=trace_colors[k],
                        opacity=opacity
                    ),
                    customdata=np.stack(
                        [
                            rows[mask],
                            cols[mask],
                            obs[mask],
                            confidence[mask]
                        ],
                        axis=-1
                    ),
                    hovertemplate=(
                        f"<b>{name}</b>"
                        " (p=%{customdata[3]:.2f})<br>"
                        "x=%{x:.2f} m<br>"
                        "y=%{y:.2f} m<br>"
                        "z=%{z:.2f} m<br>"
                        "cell=(%{customdata[0]:.0f}, "
                        "%{customdata[1]:.0f})<br>"
                        "obs=%{customdata[2]:.0f}"
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="x [m]",
                yaxis_title="y [m]",
                zaxis_title="z [m]",
                aspectmode="data" if equal_aspect else "auto"
            ),
            legend=dict(
                title="Classes",
                itemsizing="constant"
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def plot_semantic_3d(
        self,
        ax=None,
        point_size=1,
        color_by="class",
        class_colors=None,
        show_counts=True,
        equal_aspect=True,
        max_points=150000,
        seed=0
    ):
        """
        Static Matplotlib 3D scatter of observed cells: (x, y) in
        world-frame meters, z = elevation, colored by winning class
        (with a legend) or by the rgb layer.

        Matplotlib draws every point on the CPU, so above max_points
        cells a random (seeded, reproducible) subset is drawn; the
        legend counts still refer to ALL observed cells. Use
        plot_semantic_3d_plotly() to explore the full map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes (with 3D projection), optional
            Axis to draw into. A new 3D figure/axis is created if
            None.

        point_size : float
            Marker size passed to scatter().

        color_by : {"class", "rgb"}
            "class" (default): winning class color, with a legend.
            "rgb": mean palette color of the cell, no legend.

        class_colors : dict, optional
            Per-class color overrides, {class_name: (r, g, b)} in
            [0, 255] (color_by="class" only).

        show_counts : bool
            Append the number of cells to each legend entry.

        equal_aspect : bool
            Same scale on x, y, z, so z isn't visually stretched.

        max_points : int or None
            Maximum number of cells drawn (None = all).

        seed : int
            Seed of the subsampling.

        Returns
        -------
        (fig, ax)
        """

        self.require_initialized()

        elevation = self.grid_map.get_elevation_layer()
        valid = (self.grid_map.get_count_layer() > 0) & np.isfinite(elevation)

        if not np.any(valid):
            raise RuntimeError(
                "GridMap has no observed cells yet -- nothing to plot."
            )

        rows, cols = np.nonzero(valid)

        labels = None

        if color_by == "class":
            palette = self.get_class_palette(class_colors)
            labels = self.get_winning_classes(valid)[0]
            colors = palette[labels] / 255.0
        elif color_by == "rgb":
            rgb = self.grid_map.get_rgb_layer()[rows, cols]
            colors = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255) / 255.0
        else:
            raise ValueError(
                f"color_by must be 'class' or 'rgb', got {color_by!r}."
            )

        n_cells = rows.size
        drawn = np.arange(n_cells)

        if max_points is not None and n_cells > max_points:
            rng = np.random.default_rng(seed)
            drawn = np.sort(rng.choice(n_cells, max_points, replace=False))

        xs, ys = self.grid_map.cell_to_world(rows[drawn], cols[drawn])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = elevation[rows[drawn], cols[drawn]]

        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.scatter(
            xs, ys, zs,
            c=colors[drawn],
            s=point_size,
            depthshade=False,
            linewidths=0
        )

        title = "Semantic Point Cloud"

        if drawn.size < n_cells:
            title += f" ({drawn.size} of {n_cells} cells shown)"

        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        if equal_aspect:
            ax.set_box_aspect((
                max(np.ptp(xs), 1e-3),
                max(np.ptp(ys), 1e-3),
                max(np.ptp(zs), 1e-3)
            ))

            # z is short at true scale: few ticks, or labels overlap.
            from matplotlib.ticker import MaxNLocator
            ax.zaxis.set_major_locator(MaxNLocator(3))

        if labels is not None:
            self.add_class_legend(ax, labels, palette, show_counts)

        return fig, ax

    def plot_cost_heatmap(
        self,
        cost_layer,
        kind="total",
        ax=None,
        crop_to_observed=True,
        cmap="inferno"
    ):
        """
        Draw a traversability cost layer as a 2D heatmap, in
        world-frame meters, with unobserved cells left blank.

        This method does NOT compute any cost itself -- it only
        draws whatever array you pass it. Get that array from
        GeometricCost.compute(...)["cost"], SemanticCost.compute(...),
        or CostFusion.compute(...) (or TraversabilityPipeline's
        get_cost_layer()).

        Parameters
        ----------
        cost_layer : numpy.ndarray
            Shape (H, W), matching this GridMap's own shape. Values
            in [0, 1], NaN for cells with no cost.

        kind : str
            One of "geo", "sem", "total" -- purely cosmetic, only
            picks a default title. Doesn't affect the values shown.

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

        titles = {
            "geo": "Geometric cost (C_geo)",
            "sem": "Semantic cost (C_sem)",
            "total": "Fused cost (C_total)"
        }

        if kind not in titles:
            raise ValueError(
                f"kind must be one of {list(titles)}, got '{kind}'."
            )

        expected_shape = (self.grid_map.cell_n, self.grid_map.cell_n)

        if (
            not isinstance(cost_layer, np.ndarray)
            or cost_layer.shape != expected_shape
        ):
            raise ValueError(
                f"cost_layer must have shape {expected_shape} "
                f"(matching the grid), got "
                f"{getattr(cost_layer, 'shape', type(cost_layer))}."
            )

        row_slice, col_slice = self.resolve_crop(crop_to_observed)
        cost_view = cost_layer[row_slice, col_slice]
        extent = self.compute_extent(row_slice, col_slice)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        im = ax.imshow(
            np.ma.masked_invalid(cost_view),
            cmap=cmap,
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=1.0
        )

        ax.set_title(titles[kind])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046, label="cost")

        return fig, ax

    def get_class_palette(self, class_colors=None):
        """
        One RGB color per class, (C, 3) float array in [0, 255], in
        the GridMap's channel order: the segmentation palette from
        ClassReducer.get_filtered_class_colors(), with optional
        {class_name: (r, g, b)} overrides.
        """

        class_names = self.grid_map.class_names

        palette = np.asarray(
            self.grid_map.class_reducer.get_filtered_class_colors(),
            dtype=np.float32
        ).copy()

        if palette.shape != (len(class_names), 3):
            raise RuntimeError(
                "ClassReducer palette has shape "
                f"{palette.shape}, expected ({len(class_names)}, 3) "
                "-- palette and GridMap channel order are out of sync."
            )

        if class_colors:
            unknown = [n for n in class_colors if n not in class_names]

            if unknown:
                print(
                    "[GridMapVisualizer] WARNING: class_colors has "
                    f"names not in the GridMap: {unknown} "
                    f"(available: {class_names})."
                )

            for k, name in enumerate(class_names):
                if name in class_colors:
                    palette[k] = [float(v) for v in class_colors[name]]

        return palette

    def get_winning_classes(self, mask, row_slice=None, col_slice=None):
        """
        Winning class (argmax of the fused probabilities) and its
        probability for the cells selected by a boolean mask.

        Parameters
        ----------
        mask : numpy.ndarray of bool
            Over the full grid, or over the [row_slice, col_slice]
            crop if the slices are given.

        Returns
        -------
        labels : (N,) int, confidence : (N,) float, in np.nonzero(mask)
        order.
        """

        probs = self.grid_map.get_semantic_probs_layer()

        if row_slice is not None or col_slice is not None:
            probs = probs[row_slice, col_slice]

        cell_probs = np.nan_to_num(probs[mask], nan=0.0)
        labels = np.argmax(cell_probs, axis=1)
        confidence = cell_probs[np.arange(labels.size), labels]

        return labels, confidence

    def add_class_legend(self, ax, labels, palette, show_counts=True):
        """
        Legend with one entry per class present in labels, colored
        like the plot, in class order, placed outside the axis on the
        right (kept by savefig(bbox_inches="tight")).
        """

        from matplotlib.patches import Patch

        counts = np.bincount(labels, minlength=len(self.grid_map.class_names))

        handles = [
            Patch(
                facecolor=palette[k] / 255.0,
                edgecolor="none",
                label=f"{name} ({int(counts[k])})" if show_counts else name
            )
            for k, name in enumerate(self.grid_map.class_names)
            if counts[k] > 0
        ]

        ax.legend(
            handles=handles,
            title="Classes",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False
        )

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