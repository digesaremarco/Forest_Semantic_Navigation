from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SpotConfig:
    """Loads and provides access to the Spot connection configuration."""

    LEASE_MODES = ("take", "acquire")

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

        self.load_connection()
        self.load_lease()
        self.load_power()

    def get_section(self, name):
        section = self._config.get(name)

        if section is None:
            raise ValueError(
                f"Missing '{name}' section in {self.config_path}."
            )

        return section

    def load_connection(self):
        section = self.get_section("connection")

        self.robot_ip = self.validate_string(
            section.get("robot_ip"), "connection.robot_ip"
        )

        self.username = self.validate_string(
            section.get("username"), "connection.username"
        )

        self.password = self.validate_string(
            section.get("password"), "connection.password"
        )

        self.client_name = self.validate_string(
            section.get("client_name"), "connection.client_name"
        )

        self.time_sync_timeout_s = self.validate_positive_number(
            section.get("time_sync_timeout_s"),
            "connection.time_sync_timeout_s"
        )

    def load_lease(self):
        section = self.get_section("lease")

        self.lease_mode = section.get("mode")

        if self.lease_mode not in self.LEASE_MODES:
            raise ValueError(
                f"lease.mode must be one of {self.LEASE_MODES}, "
                f"got {self.lease_mode!r}."
            )

    def load_power(self):
        section = self.get_section("power")

        self.power_on_timeout_s = self.validate_positive_number(
            section.get("power_on_timeout_s"), "power.power_on_timeout_s"
        )

        self.power_off_timeout_s = self.validate_positive_number(
            section.get("power_off_timeout_s"), "power.power_off_timeout_s"
        )

        self.stand_timeout_s = self.validate_positive_number(
            section.get("stand_timeout_s"), "power.stand_timeout_s"
        )

        self.sit_timeout_s = self.validate_positive_number(
            section.get("sit_timeout_s"), "power.sit_timeout_s"
        )

        self.power_off_on_shutdown = self.validate_bool(
            section.get("power_off_on_shutdown"),
            "power.power_off_on_shutdown"
        )

    @staticmethod
    def validate_string(value, parameter_name):
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(
                f"{parameter_name} must be a non-empty string."
            )

        return value

    @staticmethod
    def validate_positive_number(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{parameter_name} must be a numeric value."
            )

        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be greater than zero."
            )

        return float(value)

    @staticmethod
    def validate_bool(value, parameter_name):
        if not isinstance(value, bool):
            raise ValueError(
                f"{parameter_name} must be true or false."
            )

        return value
