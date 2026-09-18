"""
Planning visualization: renders the overlay used throughout testing
(cost heatmap + detected frontiers + selected target + path +
waypoints), and can save it straight to an image file instead of
only showing it interactively.

Built for the field-test workflow described in conversation: call
plot() once per planning cycle with save_path set to a per-cycle
file name, so a full session's worth of snapshots (map before, map
after, and how the robot got there) can later be assembled into a
GIF with save_sequence_as_gif() -- which only needs Pillow, already
a project dependency (PIL.Image is used throughout fusion/mapping
for image I/O), no new dependency introduced.

Deliberately does NOT save the raw path/frontier data separately --
just this rendered image, per the decision that the image alone is
what's actually needed for the "map before / after / how it got
there" record of each exploration step.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from mapping.grid_map_visualizer import GridMapVisualizer


class PlanningVisualizer:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.grid_map_visualizer = GridMapVisualizer(grid_map)

    def plot(
        self,
        cost_layer,
        detection,
        result,
        robot_position_world,
        title="Planning result",
        save_path=None,
        show=False
    ):
        """
        Render the cost heatmap with frontiers/selection/path/
        waypoints overlaid.

        Parameters
        ----------
        cost_layer : numpy.ndarray
            C_total, as passed to PlanningPipeline.plan().

        detection : dict
            PlanningPipeline.last_detection (or
            FrontierDetector.detect()'s own output) -- needs at
            least "clusters".

        result : dict or None
            PlanningPipeline.plan()'s return value. None is handled
            gracefully: frontiers are still drawn, nothing else.

        robot_position_world : tuple of float
            (x, y), meters.

        title : str
            Plot title.

        save_path : str or Path, optional
            If given, the figure is saved there (parent directories
            created as needed).

        show : bool
            If True, also display the figure interactively
            (plt.show()). Independent of save_path -- both, either,
            or neither can be requested.

        Returns
        -------
        (fig, ax) if neither save_path nor show was given (caller
        keeps ownership and is responsible for closing the figure);
        None otherwise -- the figure is saved/shown and then closed,
        so this is safe to call in a loop over many planning cycles
        without accumulating open figures.
        """

        fig, ax = plt.subplots(figsize=(9, 9))

        self.grid_map_visualizer.plot_cost_heatmap(
            cost_layer, kind="total", ax=ax
        )

        ax.scatter(
            [robot_position_world[0]], [robot_position_world[1]],
            c="cyan", s=120, marker="*", edgecolors="black",
            label="robot", zorder=5
        )

        for cluster in detection["clusters"]:
            x, y = self.grid_map.cell_to_world(*cluster["centroid"])

            ax.scatter(
                [x], [y], c="white", s=40, marker="o",
                edgecolors="black", zorder=4
            )

        if result is not None:
            frontier = result["frontier"]

            rx, ry = self.grid_map.cell_to_world(
                *frontier["refined_centroid"]
            )

            ax.scatter(
                [rx], [ry], c="lime", s=150, marker="X",
                edgecolors="black", label="selected frontier", zorder=6
            )

            path_world = [
                self.grid_map.cell_to_world(row, col)
                for row, col in result["path_cells"]
            ]

            path_x = [point[0] for point in path_world]
            path_y = [point[1] for point in path_world]

            ax.plot(
                path_x, path_y, c="lime", linewidth=2, zorder=5,
                label="path"
            )

            waypoint_x = [wp["position"][0] for wp in result["waypoints"]]
            waypoint_y = [wp["position"][1] for wp in result["waypoints"]]

            ax.scatter(
                waypoint_x, waypoint_y, c="yellow", s=30, marker="s",
                edgecolors="black", zorder=6, label="waypoints"
            )

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)

        if show:
            plt.show()

        if save_path is not None or show:
            plt.close(fig)
            return None

        return fig, ax


def save_sequence_as_gif(image_paths, output_path, duration_ms=500, loop=0):
    """
    Assemble a sequence of already-saved image snapshots (e.g. from
    repeated PlanningVisualizer.plot(..., save_path=...) calls) into
    a single animated GIF.

    Parameters
    ----------
    image_paths : list of str or Path
        In the order they should appear in the GIF.

    output_path : str or Path

    duration_ms : int
        Time each frame is shown, in milliseconds.

    loop : int
        Number of times the GIF repeats; 0 means loop forever.
    """

    if len(image_paths) == 0:
        raise ValueError("image_paths is empty -- nothing to assemble.")

    frames = [Image.open(path).convert("RGB") for path in image_paths]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop
    )

    print(
        f"[PlanningVisualizer] Saved GIF ({len(frames)} frames) to "
        f"{output_path}."
    )
