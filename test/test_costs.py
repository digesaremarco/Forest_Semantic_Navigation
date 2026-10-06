"""
Compute the traversability costs on a saved GridMap and plot them.

Loads the map written by test_grid_map.py (no pairing, no SegFormer,
no fusion), computes GeometricCost, SemanticCost and their CostFusion,
prints how often CostFusion's per-cell fallback kicks in, and shows:

    1. geometric cost (C_geo)
    2. semantic cost (C_sem)
    3. fused cost (C_total)
    4. the geometric cost components: slope, roughness, step

Because the map is loaded, not rebuilt, this runs in seconds and
always evaluates the SAME map test_grid_map.py produced: rerun it
freely after changing costmap_config.yaml or class_costs.yaml. If
grid_map_config.yaml changed (resolution, map_length) or the retained
classes did, GridMap.load() refuses the file: rebuild the map first.

HEIGHT DATUM: the elevation is relative to the ground under the
starting pose; the costs do not depend on it (slope, roughness and
step are height differences).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from perception.perception_config_loader import PerceptionConfig
from perception.class_reducer import ClassReducer
from mapping.grid_map_config_loader import GridMapConfig
from mapping.grid_map import GridMap
from mapping.grid_map_visualizer import GridMapVisualizer
from costmap.costmap_config_loader import CostmapConfig
from costmap.geometric_cost import GeometricCost
from costmap.semantic_cost import SemanticCost
from costmap.cost_fusion import CostFusion


# Map written by test_grid_map.py; plots are saved next to it.
MAP_PATH = PROJECT_ROOT / "oggi2" / "output" / "grid_map.npz"
OUTPUT_DIR = MAP_PATH.parent

# Saved figures: resolution, and whether to also open them on screen.
FIGURE_DPI = 200
SHOW_PLOTS = True


def plot_layer(visualizer, layer, ax, title, cmap="inferno", vmin=0.0, vmax=1.0):
    """
    Heatmap of a (H, W) layer cropped to the observed cells, reusing
    GridMapVisualizer's crop/extent logic so it lines up exactly with
    the visualizer's own plots.
    """

    row_slice, col_slice = visualizer.resolve_crop(crop_to_observed=True)
    extent = visualizer.compute_extent(row_slice, col_slice)

    im = ax.imshow(
        np.ma.masked_invalid(layer[row_slice, col_slice]),
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label="cost")


def print_diagnostics(grid_map, geo_cost, sem_cost, fused_cost):
    """Valid cells per cost and how CostFusion combined them."""

    elevation = grid_map.get_elevation_layer()

    geo_valid = np.isfinite(geo_cost)
    sem_valid = np.isfinite(sem_cost)

    datum = grid_map.get_height_datum()

    print("\n" + "-" * 60)
    print("Costs")
    print("-" * 60)
    print(
        "Height datum:",
        "none (raw odom heights)" if datum is None
        else f"{datum:+.3f} m odom z (elevation 0 = ground at start)"
    )
    print("Elevation [m] min/max:", np.nanmin(elevation), np.nanmax(elevation))
    print("Elevation observed cells:", int(np.isfinite(elevation).sum()))
    print("C_geo valid cells:       ", int(geo_valid.sum()))
    print("C_sem valid cells:       ", int(sem_valid.sum()))

    print("\nFusion breakdown:")
    print(f"  both available (weighted avg): {int(np.sum(geo_valid & sem_valid))}")
    print(f"  only C_geo (fallback):         {int(np.sum(geo_valid & ~sem_valid))}")
    print(f"  only C_sem (fallback):         {int(np.sum(sem_valid & ~geo_valid))}")

    for name, layer in (("C_geo", geo_cost), ("C_sem", sem_cost), ("C_total", fused_cost)):
        if np.any(np.isfinite(layer)):
            print(f"{name + ' range:':<15}{np.nanmin(layer):.3f} - {np.nanmax(layer):.3f}")
        else:
            print(f"{name + ' range:':<15}no valid cells")


def main():
    perception_config = PerceptionConfig(
        PROJECT_ROOT / "config" / "perception_config.yaml"
    )
    grid_map_config = GridMapConfig(
        PROJECT_ROOT / "config" / "grid_map_config.yaml"
    )
    costmap_config = CostmapConfig(
        PROJECT_ROOT / "config" / "costmap_config.yaml",
        PROJECT_ROOT / "config" / "class_costs.yaml"
    )

    # -------------------------------------------------------------
    # Load the map
    # -------------------------------------------------------------

    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)
    grid_map.load(MAP_PATH)

    # -------------------------------------------------------------
    # Compute the costs
    # -------------------------------------------------------------

    geo_result = GeometricCost(costmap_config).compute(
        grid_map.get_elevation_layer(), resolution=grid_map.resolution
    )

    sem_cost = SemanticCost(costmap_config, class_reducer).compute(
        grid_map.get_semantic_probs_layer()
    )

    fused_cost = CostFusion(costmap_config).compute(geo_result["cost"], sem_cost)

    print_diagnostics(grid_map, geo_result["cost"], sem_cost, fused_cost)

    # -------------------------------------------------------------
    # Plots, one figure each
    # -------------------------------------------------------------

    visualizer = GridMapVisualizer(grid_map)
    figures = {}

    for layer, kind in (
        (geo_result["cost"], "geo"),
        (sem_cost, "sem"),
        (fused_cost, "total"),
    ):
        fig, _ = visualizer.plot_cost_heatmap(layer, kind=kind)
        fig.tight_layout()
        figures[f"cost_{kind}.png"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Geometric cost components")

    for ax, key, title in zip(
        axes,
        ("slope_cost", "roughness_cost", "step_cost"),
        ("Slope cost", "Roughness cost", "Step cost"),
    ):
        plot_layer(visualizer, geo_result[key], ax, title)

    fig.tight_layout()
    figures["cost_geo_components.png"] = fig

    # -------------------------------------------------------------
    # Save (before show(): closing the windows discards the figures)
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()

    for filename, fig in figures.items():
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved {path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()