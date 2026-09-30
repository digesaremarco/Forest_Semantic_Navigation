"""
Loads the static body -> camera extrinsic transform produced by
fusion/query_camera_extrinsics.py (config/camera_extrinsics.yaml).

Used by FramePosePairer to compose the TRUE camera pose in the odom
frame:

    R_odom_camera = R_odom_body @ R_body_camera
    t_odom_camera = R_odom_body @ t_body_camera + t_odom_body

instead of using the robot body's pose directly as an approximation
of the camera's pose (the placeholder behavior used everywhere so
far, since this file didn't exist yet).
"""

from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fusion.geometry_utils import quaternion_to_rotation_matrix


class CameraExtrinsicsConfig:
    """Loads and provides access to the static body->camera extrinsic."""

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

        self.load_extrinsics()

    def load_extrinsics(self):
        self.camera_frame_name = self._config.get("camera_frame_name")

        if self.camera_frame_name is None:
            raise ValueError(
                f"Missing 'camera_frame_name' in {self.config_path}."
            )

        position = self._config.get("position")
        rotation = self._config.get("rotation")

        if position is None:
            raise ValueError(
                f"Missing 'position' in {self.config_path}."
            )

        if rotation is None:
            raise ValueError(
                f"Missing 'rotation' in {self.config_path}."
            )

        required_position_fields = {"x", "y", "z"}
        required_rotation_fields = {"qx", "qy", "qz", "qw"}

        if not required_position_fields.issubset(position):
            raise ValueError(
                f"'position' in {self.config_path} must have x, y, z."
            )

        if not required_rotation_fields.issubset(rotation):
            raise ValueError(
                f"'rotation' in {self.config_path} must have "
                "qx, qy, qz, qw."
            )

        # Static body -> camera translation.
        self.t = np.array(
            [position["x"], position["y"], position["z"]],
            dtype=np.float32
        )

        # Static body -> camera rotation.
        self.R = quaternion_to_rotation_matrix(
            rotation["qx"], rotation["qy"], rotation["qz"], rotation["qw"]
        )
