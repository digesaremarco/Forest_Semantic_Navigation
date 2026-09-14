from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CostmapConfig:
    """
    Loads and provides access to the full costmap configuration:
    geometric cost thresholds/weights (costmap_config.yaml) and
    per-class semantic costs (class_costs.yaml). Two separate YAML
    files -- one per concern, easy to edit/version independently --
    behind a single loader class.
    """

    def __init__(self, costmap_config_path, class_costs_path):
        self.costmap_config_path = Path(costmap_config_path)
        self.class_costs_path = Path(class_costs_path)

        self._costmap_config = self.read_yaml(self.costmap_config_path)
        self._class_costs_config = self.read_yaml(self.class_costs_path)

        self.load_thresholds()
        self.load_weights()
        self.load_roughness_window()
        self.load_class_costs()

    @staticmethod
    def read_yaml(path):
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(
                f"Configuration file is empty: {path}"
            )

        return config

    def load_thresholds(self):
        thresholds = self._costmap_config.get("thresholds")

        if thresholds is None:
            raise ValueError(
                "Missing 'thresholds' section in "
                f"{self.costmap_config_path}."
            )

        self.theta_max_deg = thresholds.get("theta_max_deg")
        self.sigma_max = thresholds.get("sigma_max")
        self.s_max = thresholds.get("s_max")

        required_fields = {
            "thresholds.theta_max_deg": self.theta_max_deg,
            "thresholds.sigma_max": self.sigma_max,
            "thresholds.s_max": self.s_max
        }

        for field_name, value in required_fields.items():

            if value is None:
                raise ValueError(
                    f"Missing '{field_name}' in "
                    f"{self.costmap_config_path}."
                )

            self.validate_positive(value, field_name)

    def load_weights(self):
        weights = self._costmap_config.get("weights")

        if weights is None:
            raise ValueError(
                f"Missing 'weights' section in {self.costmap_config_path}."
            )

        self.weight_slope = weights.get("slope")
        self.weight_roughness = weights.get("roughness")
        self.weight_step = weights.get("step")

        required_fields = {
            "weights.slope": self.weight_slope,
            "weights.roughness": self.weight_roughness,
            "weights.step": self.weight_step
        }

        for field_name, value in required_fields.items():

            if value is None:
                raise ValueError(
                    f"Missing '{field_name}' in "
                    f"{self.costmap_config_path}."
                )

            self.validate_positive(value, field_name)

        total = (
            self.weight_slope
            + self.weight_roughness
            + self.weight_step
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "weights.slope + weights.roughness + weights.step must "
                f"sum to 1.0, got {total}."
            )

    def load_roughness_window(self):
        self.roughness_window = self._costmap_config.get(
            "roughness_window"
        )

        if self.roughness_window is None:
            raise ValueError(
                "Missing 'roughness_window' in "
                f"{self.costmap_config_path}."
            )

        if not isinstance(self.roughness_window, int):
            raise ValueError(
                "roughness_window must be an integer number of cells."
            )

        if self.roughness_window < 1:
            raise ValueError(
                "roughness_window must be at least 1."
            )

        if self.roughness_window % 2 == 0:
            raise ValueError(
                "roughness_window must be odd, so the cell itself is "
                "the window's center."
            )

    def load_class_costs(self):
        classes = self._class_costs_config.get("classes")

        if classes is None:
            raise ValueError(
                f"Missing 'classes' section in {self.class_costs_path}."
            )

        self.class_costs = {}

        for class_name, cost in classes.items():

            if not isinstance(cost, (int, float)):
                raise ValueError(
                    f"Cost for class '{class_name}' must be numeric, "
                    f"got {type(cost)}."
                )

            if not (0.0 <= cost <= 1.0):
                raise ValueError(
                    f"Cost for class '{class_name}' must be in "
                    f"[0, 1], got {cost}."
                )

            self.class_costs[class_name] = float(cost)

    def get_cost(self, class_name):
        """Return the configured cost for a single class, by name."""

        if class_name not in self.class_costs:
            raise ValueError(
                f"No cost defined for class '{class_name}'. "
                f"Available classes: {list(self.class_costs.keys())}"
            )

        return self.class_costs[class_name]

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