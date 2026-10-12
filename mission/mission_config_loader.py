from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MissionConfig:
    """Loads and provides access to the exploration mission configuration."""

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

        self.load_stop_criteria()
        self.load_return_home()
        self.load_keyframes()
        self.load_output()

    def get_section(self, name):
        section = self._config.get(name)

        if section is None:
            raise ValueError(
                f"Missing '{name}' section in {self.config_path}."
            )

        return section

    def load_stop_criteria(self):
        section = self.get_section("stop_criteria")

        self.coverage_threshold = self.validate_positive_number(
            section.get("coverage_threshold"), "stop_criteria.coverage_threshold"
        )

        if self.coverage_threshold > 1.0:
            raise ValueError(
                "stop_criteria.coverage_threshold is a fraction in (0, 1], "
                f"got {self.coverage_threshold}."
            )

        self.max_exploration_time_s = self.validate_positive_number(
            section.get("max_exploration_time_s"), "stop_criteria.max_exploration_time_s"
        )

        self.min_battery_percent = self.validate_non_negative_number(
            section.get("min_battery_percent"), "stop_criteria.min_battery_percent"
        )

        self.max_consecutive_nav_failures = self.validate_positive_int(
            section.get("max_consecutive_nav_failures"),
            "stop_criteria.max_consecutive_nav_failures"
        )

        self.max_consecutive_acquisition_failures = self.validate_positive_int(
            section.get("max_consecutive_acquisition_failures"),
            "stop_criteria.max_consecutive_acquisition_failures"
        )

    def load_return_home(self):
        section = self.get_section("return_home")

        self.return_home_enabled = self.validate_bool(
            section.get("enabled"), "return_home.enabled"
        )

        self.return_home_max_attempts = self.validate_positive_int(
            section.get("max_attempts"), "return_home.max_attempts"
        )

    def load_keyframes(self):
        section = self.get_section("keyframes")

        self.keyframes_enabled = self.validate_bool(
            section.get("enabled"), "keyframes.enabled"
        )

        self.keyframe_min_translation_m = self.validate_non_negative_number(
            section.get("min_translation_m"), "keyframes.min_translation_m"
        )

        self.keyframe_min_rotation_deg = self.validate_non_negative_number(
            section.get("min_rotation_deg"), "keyframes.min_rotation_deg"
        )

    def load_output(self):
        section = self.get_section("output")

        run_prefix = section.get("run_prefix")

        if not isinstance(run_prefix, str) or run_prefix.strip() == "":
            raise ValueError("output.run_prefix must be a non-empty string.")

        self.run_prefix = run_prefix

        self.save_cycle_plots = self.validate_bool(
            section.get("save_cycle_plots"), "output.save_cycle_plots"
        )

        self.save_map_every_cycle = self.validate_bool(
            section.get("save_map_every_cycle"), "output.save_map_every_cycle"
        )

    def as_dict(self):
        """Raw configuration, for the mission summary."""

        return dict(self._config)

    @staticmethod
    def validate_number(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be a numeric value.")

        return float(value)

    @classmethod
    def validate_positive_number(cls, value, parameter_name):
        value = cls.validate_number(value, parameter_name)

        if value <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

        return value

    @classmethod
    def validate_non_negative_number(cls, value, parameter_name):
        value = cls.validate_number(value, parameter_name)

        if value < 0:
            raise ValueError(f"{parameter_name} must be >= 0.")

        return value

    @staticmethod
    def validate_positive_int(value, parameter_name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

        return value

    @staticmethod
    def validate_bool(value, parameter_name):
        if not isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be true or false.")

        return value
