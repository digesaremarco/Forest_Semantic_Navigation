from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PlanningConfig:
    """Loads and provides access to the planning configuration."""

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

        self.load_frontier_detection()
        self.load_frontier_refinement()
        self.load_frontier_selection()
        self.load_path_planning()
        self.load_waypoint_generation()
        self.load_path_smoothing()
        self.load_replanning()

    def load_frontier_detection(self):
        section = self._config.get("frontier_detection")

        if section is None:
            raise ValueError(
                "Missing 'frontier_detection' section in "
                f"{self.config_path}."
            )

        self.occupied_threshold = section.get("occupied_threshold")
        self.border_margin_cells = section.get("border_margin_cells")
        self.min_cluster_size = section.get("min_cluster_size")

        if self.occupied_threshold is None:
            raise ValueError(
                "Missing 'frontier_detection.occupied_threshold' in "
                f"{self.config_path}."
            )

        if not (0.0 < self.occupied_threshold < 1.0):
            raise ValueError(
                "frontier_detection.occupied_threshold must be in "
                f"(0, 1), got {self.occupied_threshold}."
            )

        self.border_margin_cells = self.validate_positive_int(
            self.border_margin_cells,
            "frontier_detection.border_margin_cells"
        )

        self.min_cluster_size = self.validate_positive_int(
            self.min_cluster_size,
            "frontier_detection.min_cluster_size"
        )

    def load_frontier_refinement(self):
        section = self._config.get("frontier_refinement")

        if section is None:
            raise ValueError(
                "Missing 'frontier_refinement' section in "
                f"{self.config_path}."
            )

        self.mean_shift_bandwidth_cells = self.validate_positive_int(
            section.get("mean_shift_bandwidth_cells"),
            "frontier_refinement.mean_shift_bandwidth_cells"
        )

        self.max_shift_cells = self.validate_positive_int(
            section.get("max_shift_cells"),
            "frontier_refinement.max_shift_cells"
        )

    def load_frontier_selection(self):
        section = self._config.get("frontier_selection")

        if section is None:
            raise ValueError(
                "Missing 'frontier_selection' section in "
                f"{self.config_path}."
            )

        self.information_gain_radius_cells = self.validate_positive_int(
            section.get("information_gain_radius_cells"),
            "frontier_selection.information_gain_radius_cells"
        )

        self.beta_gain = self.validate_positive_number(
            section.get("beta_gain"),
            "frontier_selection.beta_gain"
        )

        self.beta_cost = self.validate_positive_number(
            section.get("beta_cost"),
            "frontier_selection.beta_cost"
        )

    def load_path_planning(self):
        section = self._config.get("path_planning")

        if section is None:
            raise ValueError(
                f"Missing 'path_planning' section in {self.config_path}."
            )

        self.weight_distance = self.validate_positive_number(
            section.get("weight_distance"),
            "path_planning.weight_distance"
        )

        self.weight_traversability = self.validate_positive_number(
            section.get("weight_traversability"),
            "path_planning.weight_traversability"
        )

        self.nan_cost_penalty = section.get("nan_cost_penalty")

        if self.nan_cost_penalty is None:
            raise ValueError(
                "Missing 'path_planning.nan_cost_penalty' in "
                f"{self.config_path}."
            )

        if not (0.0 <= self.nan_cost_penalty <= 1.0):
            raise ValueError(
                "path_planning.nan_cost_penalty must be in [0, 1], "
                f"got {self.nan_cost_penalty}."
            )

    def load_waypoint_generation(self):
        section = self._config.get("waypoint_generation")

        if section is None:
            raise ValueError(
                "Missing 'waypoint_generation' section in "
                f"{self.config_path}."
            )

        self.waypoint_spacing_m = self.validate_positive_number(
            section.get("waypoint_spacing_m"),
            "waypoint_generation.waypoint_spacing_m"
        )

    def load_path_smoothing(self):
        section = self._config.get("path_smoothing")

        if section is None:
            raise ValueError(
                "Missing 'path_smoothing' section in "
                f"{self.config_path}."
            )

        self.cost_increase_tolerance = section.get("cost_increase_tolerance")

        if self.cost_increase_tolerance is None:
            raise ValueError(
                "Missing 'path_smoothing.cost_increase_tolerance' in "
                f"{self.config_path}."
            )

        if not isinstance(self.cost_increase_tolerance, (int, float)):
            raise ValueError(
                "path_smoothing.cost_increase_tolerance must be a "
                "numeric value."
            )

        if self.cost_increase_tolerance < 0:
            raise ValueError(
                "path_smoothing.cost_increase_tolerance must be >= 0."
            )

        self.cost_increase_tolerance = float(self.cost_increase_tolerance)

    def load_replanning(self):
        section = self._config.get("replanning")

        if section is None:
            raise ValueError(
                f"Missing 'replanning' section in {self.config_path}."
            )

        self.goal_reached_distance_m = self.validate_positive_number(
            section.get("goal_reached_distance_m"),
            "replanning.goal_reached_distance_m"
        )

        self.timeout_s = self.validate_positive_number(
            section.get("timeout_s"),
            "replanning.timeout_s"
        )

    @staticmethod
    def validate_positive_number(value, parameter_name):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{parameter_name} must be a numeric value."
            )

        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be greater than zero."
            )

        return float(value)

    @staticmethod
    def validate_positive_int(value, parameter_name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{parameter_name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be greater than zero."
            )

        return value