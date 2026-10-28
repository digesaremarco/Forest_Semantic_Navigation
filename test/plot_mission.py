"""
Regenerate the mission overview (mission_path.png) offline, from a run
folder written by main.py -- no robot, no SegFormer.

Reads from RUN_DIR:
    grid_map.npz            the final map (GridMap.save())
    path.csv                the walked path (~10 Hz + captures)
    planned_paths.csv       the A* plans
    mission_summary.json    home pose and the subtitle numbers

C_total is recomputed from the map exactly as TraversabilityPipeline
does (GeometricCost + SemanticCost + CostFusion), so after changing
costmap_config.yaml or class_costs.yaml the background shows the NEW
costs on the same map. If grid_map_config.yaml or the retained classes
changed since the run, GridMap.load() refuses the file.

Writes RUN_DIR/OUTPUT_NAME (default mission_path.png, overwriting the
one made at the end of the mission).
"""

import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# Run folder written by main.py.
RUN_DIR = PROJECT_ROOT / "spot_runs" / "mission_20261027_111938"

OUTPUT_NAME = "mission_path.png"

# Also open the plot on screen (it is always saved).
SHOW_PLOT = True


def compute_total_cost(grid_map, costmap_config, class_reducer):
    """C_total, same composition as TraversabilityPipeline."""

    from costmap.geometric_cost import GeometricCost
    from costmap.semantic_cost import SemanticCost
    from costmap.cost_fusion import CostFusion

    geo_cost = GeometricCost(costmap_config).compute(
        grid_map.get_elevation_layer(), resolution=grid_map.resolution
    )["cost"]

    sem_cost = SemanticCost(costmap_config, class_reducer).compute(
        grid_map.get_semantic_probs_layer()
    )

    return CostFusion(costmap_config).compute(geo_cost, sem_cost)


def load_cost_map(map_path):
    """(cost_layer, resolution, origin_xy, cell_n), or None if no map."""

    if not map_path.exists():
        print(f"No map at {map_path}: plotting the path only.")
        return None

    from perception.perception_config_loader import PerceptionConfig
    from perception.class_reducer import ClassReducer
    from mapping.grid_map_config_loader import GridMapConfig
    from mapping.grid_map import GridMap
    from costmap.costmap_config_loader import CostmapConfig

    config_dir = PROJECT_ROOT / "config"

    perception_config = PerceptionConfig(config_dir / "perception_config.yaml")
    grid_map_config = GridMapConfig(config_dir / "grid_map_config.yaml")
    costmap_config = CostmapConfig(
        config_dir / "costmap_config.yaml", config_dir / "class_costs.yaml"
    )

    class_reducer = ClassReducer(perception_config)
    grid_map = GridMap(grid_map_config, class_reducer)
    grid_map.load(map_path)

    cost_layer = compute_total_cost(grid_map, costmap_config, class_reducer)

    return cost_layer, grid_map.resolution, grid_map.get_origin()[:2], grid_map.cell_n


def main():
    if not SHOW_PLOT:
        matplotlib.use("Agg")

    from mission.mission_plotter import MissionPlotter

    run_dir = Path(RUN_DIR)

    if not run_dir.exists():
        sys.exit(f"Run folder not found: {run_dir} -- set RUN_DIR at the top of this script.")

    plotter = MissionPlotter()

    path_rows = plotter.load_path_csv(run_dir / "path.csv")
    planned_paths = plotter.load_planned_csv(run_dir / "planned_paths.csv")

    summary_path = run_dir / "mission_summary.json"
    summary = plotter.load_summary(summary_path) if summary_path.exists() else {}

    print(f"Path: {len(path_rows)} poses, {len(planned_paths)} planned path(s)")

    cost_map = load_cost_map(run_dir / "grid_map.npz")

    if cost_map is None:
        cost_layer, resolution, origin, cell_n = None, 0.04, (0.0, 0.0), 1
    else:
        cost_layer, resolution, origin, cell_n = cost_map

    save_path = plotter.plot(
        save_path=run_dir / OUTPUT_NAME,
        cost_layer=cost_layer,
        resolution=resolution,
        origin=origin,
        cell_n=cell_n,
        path_rows=path_rows,
        planned_paths=planned_paths,
        home_pose=summary.get("home_pose"),
        title=f"Mission {run_dir.name}",
        subtitle=plotter.subtitle_from_summary(summary) if summary else None
    )

    print(f"Saved {save_path}")

    if SHOW_PLOT:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 9))
        plt.imshow(mpimg.imread(save_path))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
