from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCOMOTION_HINTS = (
    "auto", "trot", "speed_select_trot", "crawl", "speed_select_crawl",
    "amble", "speed_select_amble", "jog", "hop",
)


class NavigationConfig:
    """Loads and provides access to the navigation configuration."""

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

        self.load_mobility()
        self.load_waypoint_following()
        self.load_failure_detection()

    def get_section(self, name):
        section = self._config.get(name)

        if section is None:
            raise ValueError(
                f"Missing '{name}' section in {self.config_path}."
            )

        return section

    def load_mobility(self):
        section = self.get_section("mobility")

        self.max_linear_velocity_mps = self.validate_positive_number(
            section.get("max_linear_velocity_mps"), "mobility.max_linear_velocity_mps"
        )

        self.max_lateral_velocity_mps = self.validate_positive_number(
            section.get("max_lateral_velocity_mps"), "mobility.max_lateral_velocity_mps"
        )

        self.max_angular_velocity_rps = self.validate_positive_number(
            section.get("max_angular_velocity_rps"), "mobility.max_angular_velocity_rps"
        )

        self.locomotion_hint = section.get("locomotion_hint")

        if self.locomotion_hint not in LOCOMOTION_HINTS:
            raise ValueError(
                f"mobility.locomotion_hint must be one of {LOCOMOTION_HINTS}, "
                f"got {self.locomotion_hint!r}."
            )

        self.body_height_m = self.validate_number(
            section.get("body_height_m"), "mobility.body_height_m"
        )

        self.obstacle_avoidance_enabled = self.validate_bool(
            section.get("obstacle_avoidance_enabled"),
            "mobility.obstacle_avoidance_enabled"
        )

        self.obstacle_avoidance_padding_m = self.validate_non_negative_number(
            section.get("obstacle_avoidance_padding_m"),
            "mobility.obstacle_avoidance_padding_m"
        )

        ground_mu_hint = section.get("ground_mu_hint")

        self.ground_mu_hint = (
            None if ground_mu_hint is None
            else self.validate_positive_number(ground_mu_hint, "mobility.ground_mu_hint")
        )

    def load_waypoint_following(self):
        section = self.get_section("waypoint_following")

        self.pass_through_radius_m = self.validate_positive_number(
            section.get("pass_through_radius_m"),
            "waypoint_following.pass_through_radius_m"
        )

        self.goal_tolerance_m = self.validate_positive_number(
            section.get("goal_tolerance_m"),
            "waypoint_following.goal_tolerance_m"
        )

        self.goal_heading_tolerance_deg = self.validate_positive_number(
            section.get("goal_heading_tolerance_deg"),
            "waypoint_following.goal_heading_tolerance_deg"
        )

        self.feedback_period_s = self.validate_positive_number(
            section.get("feedback_period_s"),
            "waypoint_following.feedback_period_s"
        )

    def load_failure_detection(self):
        section = self.get_section("failure_detection")

        self.base_timeout_s = self.validate_positive_number(
            section.get("base_timeout_s"), "failure_detection.base_timeout_s"
        )

        self.nominal_speed_mps = self.validate_positive_number(
            section.get("nominal_speed_mps"), "failure_detection.nominal_speed_mps"
        )

        self.stall_window_s = self.validate_positive_number(
            section.get("stall_window_s"), "failure_detection.stall_window_s"
        )

        self.min_progress_m = self.validate_positive_number(
            section.get("min_progress_m"), "failure_detection.min_progress_m"
        )

        self.blocked_patience_s = self.validate_positive_number(
            section.get("blocked_patience_s"), "failure_detection.blocked_patience_s"
        )

        self.max_path_duration_s = self.validate_positive_number(
            section.get("max_path_duration_s"), "failure_detection.max_path_duration_s"
        )

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
    def validate_bool(value, parameter_name):
        if not isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be true or false.")

        return value
