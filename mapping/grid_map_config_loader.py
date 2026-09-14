from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GridMapConfig:
    """Loads and provides access to the grid map configuration."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

        if self._config is None:
            raise ValueError(
                f"Configuration file is empty: {self.config_path}"
            )

        self.load_map_config()
        self.load_bounds_config()

    def load_map_config(self):
        map_config = self._config.get("map")

        if map_config is None:
            raise ValueError(
                "Missing 'map' section in grid map configuration."
            )

        self.map_length = map_config.get("map_length")
        self.resolution = map_config.get("resolution")

        if self.map_length is None:
            raise ValueError(
                "Missing 'map.map_length' in grid map configuration."
            )

        if self.resolution is None:
            raise ValueError(
                "Missing 'map.resolution' in grid map configuration."
            )

        self.validate_positive(self.map_length, "map.map_length")
        self.validate_positive(self.resolution, "map.resolution")

        if self.resolution > self.map_length:
            raise ValueError(
                "map.resolution must not be larger than map.map_length."
            )

        self.cell_n = self.compute_cell_n(
            self.map_length,
            self.resolution
        )

    def load_bounds_config(self):
        self.out_of_bounds_warning = self._config.get(
            "out_of_bounds_warning", True
        )

    @staticmethod
    def compute_cell_n(map_length, resolution):
        """
        Compute the number of cells per side of the square grid.

        Kept odd on purpose, so the grid has an exact center cell:
        that center cell is where the first pose received by
        GridMap.update() gets anchored.
        """

        cell_n = int(round(map_length / resolution))

        if cell_n % 2 == 0:
            cell_n += 1

        return cell_n

    @staticmethod
    def validate_positive(value, parameter_name):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{parameter_name} must be a numeric value."
            )

        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be greater than zero."
            )
