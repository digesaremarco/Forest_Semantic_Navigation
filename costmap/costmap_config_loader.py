from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CostmapConfig:
    """Loads and provides access to the geometric costmap configuration."""

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

        self.load_thresholds()
        self.load_weights()
        self.load_roughness_window()

    def load_thresholds(self):
        thresholds = self._config.get("thresholds")

        if thresholds is None:
            raise ValueError(
                "Missing 'thresholds' section in costmap configuration."
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
                    f"Missing '{field_name}' in costmap configuration."
                )

            self.validate_positive(value, field_name)

    def load_weights(self):
        weights = self._config.get("weights")

        if weights is None:
            raise ValueError(
                "Missing 'weights' section in costmap configuration."
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
                    f"Missing '{field_name}' in costmap configuration."
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
        self.roughness_window = self._config.get("roughness_window")

        if self.roughness_window is None:
            raise ValueError(
                "Missing 'roughness_window' in costmap configuration."
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
