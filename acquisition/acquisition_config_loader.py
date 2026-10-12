from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_CAMERAS = ("frontleft", "frontright", "left", "right", "back")


class AcquisitionConfig:
    """Loads and provides access to the acquisition configuration."""

    MISMATCH_POLICIES = ("warn", "error")

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

        self.load_cameras()
        self.load_image_request()
        self.load_capture()
        self.load_calibration_check()
        self.load_logging()

    def get_section(self, name):
        section = self._config.get(name)

        if section is None:
            raise ValueError(
                f"Missing '{name}' section in {self.config_path}."
            )

        return section

    def load_cameras(self):
        cameras = self._config.get("cameras")

        if not isinstance(cameras, list) or len(cameras) == 0:
            raise ValueError(
                f"'cameras' must be a non-empty list in {self.config_path}."
            )

        unsupported = [name for name in cameras if name not in SUPPORTED_CAMERAS]

        if unsupported:
            raise ValueError(
                f"Unsupported camera(s) {unsupported} in 'cameras'. "
                f"Supported: {SUPPORTED_CAMERAS}."
            )

        if len(set(cameras)) != len(cameras):
            raise ValueError(f"'cameras' contains duplicates: {cameras}.")

        self.camera_names = tuple(cameras)

    def load_image_request(self):
        section = self.get_section("image_request")

        self.jpeg_quality = self.validate_positive_int(
            section.get("jpeg_quality"), "image_request.jpeg_quality"
        )

        if self.jpeg_quality > 100:
            raise ValueError(
                "image_request.jpeg_quality must be in [1, 100], "
                f"got {self.jpeg_quality}."
            )

    def load_capture(self):
        section = self.get_section("capture")

        self.settle_time_s = self.validate_non_negative_number(
            section.get("settle_time_s"), "capture.settle_time_s"
        )

        self.max_visual_depth_dt_s = self.validate_positive_number(
            section.get("max_visual_depth_dt_s"), "capture.max_visual_depth_dt_s"
        )

        self.max_retries = self.validate_positive_int(
            section.get("max_retries"), "capture.max_retries"
        )

        self.retry_delay_s = self.validate_non_negative_number(
            section.get("retry_delay_s"), "capture.retry_delay_s"
        )

    def load_calibration_check(self):
        section = self.get_section("calibration_check")

        self.calibration_check_enabled = self.validate_bool(
            section.get("enabled"), "calibration_check.enabled"
        )

        self.calibration_on_mismatch = section.get("on_mismatch")

        if self.calibration_on_mismatch not in self.MISMATCH_POLICIES:
            raise ValueError(
                "calibration_check.on_mismatch must be one of "
                f"{self.MISMATCH_POLICIES}, got {self.calibration_on_mismatch!r}."
            )

        self.max_translation_error_m = self.validate_positive_number(
            section.get("max_translation_error_m"),
            "calibration_check.max_translation_error_m"
        )

        self.max_rotation_error_deg = self.validate_positive_number(
            section.get("max_rotation_error_deg"),
            "calibration_check.max_rotation_error_deg"
        )

        self.max_intrinsics_error_px = self.validate_positive_number(
            section.get("max_intrinsics_error_px"),
            "calibration_check.max_intrinsics_error_px"
        )

    def load_logging(self):
        section = self.get_section("logging")

        self.save_to_disk = self.validate_bool(
            section.get("save_to_disk"), "logging.save_to_disk"
        )

        output_root = section.get("output_root")

        if not isinstance(output_root, str) or output_root.strip() == "":
            raise ValueError("logging.output_root must be a non-empty string.")

        output_root = Path(output_root)

        self.output_root = (
            output_root if output_root.is_absolute()
            else PROJECT_ROOT / output_root
        )

    @staticmethod
    def visual_source_name(camera_name):
        return f"{camera_name}_fisheye_image"

    @staticmethod
    def depth_source_name(camera_name):
        return f"{camera_name}_depth_in_visual_frame"

    @staticmethod
    def validate_positive_number(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be a numeric value.")

        if value <= 0:
            raise ValueError(f"{parameter_name} must be greater than zero.")

        return float(value)

    @staticmethod
    def validate_non_negative_number(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be a numeric value.")

        if value < 0:
            raise ValueError(f"{parameter_name} must be >= 0.")

        return float(value)

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
