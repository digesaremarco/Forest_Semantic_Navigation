from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CameraConfig:
    """Loads and provides access to the camera configuration."""

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

        self.load_camera_config()

    def load_camera_config(self):
        cameras_config = self._config.get("cameras")

        if cameras_config is None:
            raise ValueError(
                "Missing 'cameras' section in camera configuration."
            )

        self.load_rgb_config(cameras_config)
        self.load_depth_config(cameras_config)

    def load_rgb_config(self, cameras_config):
        rgb_config = cameras_config.get("rgb")

        if rgb_config is None:
            raise ValueError(
                "Missing 'cameras.rgb' section in camera configuration."
            )

        self.rgb_width = rgb_config.get("width")
        self.rgb_height = rgb_config.get("height")

        if self.rgb_width is None:
            raise ValueError(
                "Missing 'cameras.rgb.width' in camera configuration."
            )

        if self.rgb_height is None:
            raise ValueError(
                "Missing 'cameras.rgb.height' in camera configuration."
            )

        self.validate_positive(
            self.rgb_width,
            "cameras.rgb.width"
        )

        self.validate_positive(
            self.rgb_height,
            "cameras.rgb.height"
        )

    def load_depth_config(self, cameras_config):
        depth_config = cameras_config.get("depth")

        if depth_config is None:
            raise ValueError(
                "Missing 'cameras.depth' section in camera configuration."
            )

        self.depth_width = depth_config.get("width")
        self.depth_height = depth_config.get("height")

        if self.depth_width is None:
            raise ValueError(
                "Missing 'cameras.depth.width' in camera configuration."
            )

        if self.depth_height is None:
            raise ValueError(
                "Missing 'cameras.depth.height' in camera configuration."
            )

        self.validate_positive(
            self.depth_width,
            "cameras.depth.width"
        )

        self.validate_positive(
            self.depth_height,
            "cameras.depth.height"
        )

        intrinsics = depth_config.get("intrinsics")

        if intrinsics is None:
            raise ValueError(
                "Missing 'cameras.depth.intrinsics' "
                "in camera configuration."
            )

        self.fx = intrinsics.get("fx")
        self.fy = intrinsics.get("fy")
        self.cx = intrinsics.get("cx")
        self.cy = intrinsics.get("cy")

        if self.fx is None:
            raise ValueError(
                "Missing 'cameras.depth.intrinsics.fx' "
                "in camera configuration."
            )

        if self.fy is None:
            raise ValueError(
                "Missing 'cameras.depth.intrinsics.fy' "
                "in camera configuration."
            )

        if self.cx is None:
            raise ValueError(
                "Missing 'cameras.depth.intrinsics.cx' "
                "in camera configuration."
            )

        if self.cy is None:
            raise ValueError(
                "Missing 'cameras.depth.intrinsics.cy' "
                "in camera configuration."
            )

        self.validate_positive(
            self.fx,
            "cameras.depth.intrinsics.fx"
        )

        self.validate_positive(
            self.fy,
            "cameras.depth.intrinsics.fy"
        )

        if self.cx < 0 or self.cx >= self.depth_width:
            raise ValueError(
                "cameras.depth.intrinsics.cx must be inside "
                "the depth image width."
            )

        if self.cy < 0 or self.cy >= self.depth_height:
            raise ValueError(
                "cameras.depth.intrinsics.cy must be inside "
                "the depth image height."
            )

        self.depth_scale = depth_config.get("depth_scale")

        if self.depth_scale is None:
            raise ValueError(
                "Missing 'cameras.depth.depth_scale' "
                "in camera configuration."
            )

        self.validate_positive(
            self.depth_scale,
            "cameras.depth.depth_scale"
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

    def get_rgb_resolution(self):
        return self.rgb_width, self.rgb_height

    def get_depth_resolution(self):
        return self.depth_width, self.depth_height

    def get_depth_intrinsics(self):
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy
        }

    def get_depth_scale(self):
        return self.depth_scale