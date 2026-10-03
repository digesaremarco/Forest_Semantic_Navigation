"""
Loads the real multi-camera calibration produced on the robot
(nested per-camera / per-source-type YAML, config/camera_intrinsics.
yaml) and exposes, per camera, the intrinsics and the static
body -> camera extrinsic needed everywhere else in the project.

Scope: only the 4 confirmed cameras are loaded by default
(frontleft, frontright, left, right) -- "back" is present in the
real file but intentionally excluded.

Only "visual" and "depth_in_visual" are read. Spot's depth is
always consumed already registered into the visual frame (the
convention used throughout this project, see
fusion/frame_pose_pairing.py), so the raw "depth" source (its own
frame/resolution/extrinsics) is intentionally never used here.

Because depth_in_visual_frame is, by construction, the same camera
as visual (identical frame_name/resolution/intrinsics/extrinsics,
confirmed on the real calibration data), this loader cross-checks
the two sections against each other and fails loudly on any
mismatch, rather than silently trusting one of them -- a mismatch
would mean a different robot/SDK version than assumed everywhere
else in the project.

Only the pinhole model is supported (Spot's fisheye images are
pre-rectified by the robot in every version checked so far, and the
real calibration confirms this for all 5 cameras); a distortion
model reported for any source raises explicitly instead of being
silently ignored.
"""

from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fusion.geometry_utils import quaternion_to_rotation_matrix

DEFAULT_CAMERA_NAMES = ("frontleft", "frontright", "left", "right")

# Fields that must agree, value-for-value, between "visual" and
# "depth_in_visual" for a given camera.
_CROSS_CHECK_FIELDS = (
    "frame_name", "width", "height", "fx", "fy", "cx", "cy"
)


class CameraCalibrationConfig:
    """
    Per-camera intrinsics + static body->camera extrinsics, read
    from the real nested calibration YAML.
    """

    def __init__(self, config_path, camera_names=DEFAULT_CAMERA_NAMES):
        self.config_path = Path(config_path)
        self.camera_names = tuple(camera_names)

        self._raw = self.read_yaml(self.config_path)

        self.cameras = {}

        for camera_name in self.camera_names:
            self.cameras[camera_name] = self.load_camera(camera_name)

    @staticmethod
    def read_yaml(path):
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Configuration file is empty: {path}")

        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration file {path} must contain a mapping of "
                f"camera name -> source type -> calibration."
            )

        return config

    def load_camera(self, camera_name):
        if camera_name not in self._raw:
            raise ValueError(
                f"Camera '{camera_name}' not found in {self.config_path}. "
                f"Available cameras: {list(self._raw.keys())}"
            )

        section = self._raw[camera_name]

        visual = self.get_source(camera_name, section, "visual")
        depth_in_visual = self.get_source(
            camera_name, section, "depth_in_visual"
        )

        self.validate_pinhole(camera_name, "visual", visual)
        self.validate_pinhole(camera_name, "depth_in_visual", depth_in_visual)

        self.cross_check(camera_name, visual, depth_in_visual)

        R_visual, t_visual = self.extract_extrinsics(
            camera_name, "visual", visual
        )
        R_depth, t_depth = self.extract_extrinsics(
            camera_name, "depth_in_visual", depth_in_visual
        )

        if (
            not np.allclose(R_visual, R_depth, atol=1e-6)
            or not np.allclose(t_visual, t_depth, atol=1e-6)
        ):
            raise ValueError(
                f"Camera '{camera_name}': 'visual' and 'depth_in_visual' "
                f"report different body_tform_camera. Expected them to "
                f"be the exact same physical camera."
            )

        depth_scale = depth_in_visual.get("depth_scale")

        if depth_scale is None:
            raise ValueError(
                f"Camera '{camera_name}': missing 'depth_scale' in "
                f"'depth_in_visual'."
            )

        if not isinstance(depth_scale, (int, float)) or depth_scale <= 0:
            raise ValueError(
                f"Camera '{camera_name}': 'depth_scale' must be a "
                f"positive number, got {depth_scale}."
            )

        return {
            "frame_name": visual["frame_name"],
            "width": int(visual["width"]),
            "height": int(visual["height"]),
            "fx": float(visual["fx"]),
            "fy": float(visual["fy"]),
            "cx": float(visual["cx"]),
            "cy": float(visual["cy"]),
            "skew": float(visual.get("skew", 0.0)),
            "depth_scale": float(depth_scale),
            "R": R_visual,
            "t": t_visual,
        }

    @staticmethod
    def get_source(camera_name, section, source_type):
        if source_type not in section:
            raise ValueError(
                f"Camera '{camera_name}' has no '{source_type}' section "
                f"in the calibration file. Available: {list(section.keys())}"
            )

        return section[source_type]

    @staticmethod
    def validate_pinhole(camera_name, source_type, entry):
        model = entry.get("camera_model")

        if model != "pinhole":
            raise ValueError(
                f"Camera '{camera_name}' ({source_type}) reports camera "
                f"model '{model}'. Only 'pinhole' is supported -- Spot's "
                f"fisheye sources are expected to be pre-rectified. If "
                f"this robot/SDK version reports distortion, the "
                f"downstream pinhole-only backprojection in "
                f"PointCloudBuilder needs to be revisited first."
            )

        required_fields = (
            "frame_name", "width", "height", "fx", "fy", "cx", "cy"
        )

        for field_name in required_fields:
            if entry.get(field_name) is None:
                raise ValueError(
                    f"Camera '{camera_name}' ({source_type}) is missing "
                    f"'{field_name}'."
                )

    @staticmethod
    def cross_check(camera_name, visual, depth_in_visual):
        mismatches = []

        for field_name in _CROSS_CHECK_FIELDS:
            v_value = visual.get(field_name)
            d_value = depth_in_visual.get(field_name)

            if isinstance(v_value, float) or isinstance(d_value, float):
                equal = abs(float(v_value) - float(d_value)) <= 1e-6
            else:
                equal = v_value == d_value

            if not equal:
                mismatches.append((field_name, v_value, d_value))

        if mismatches:
            details = ", ".join(
                f"{name}: visual={v!r} vs depth_in_visual={d!r}"
                for name, v, d in mismatches
            )
            raise ValueError(
                f"Camera '{camera_name}': 'visual' and 'depth_in_visual' "
                f"disagree on fields assumed identical ({details}). This "
                f"breaks the project-wide assumption that depth is "
                f"already registered into the visual frame -- check the "
                f"robot/SDK version."
            )

    @staticmethod
    def extract_extrinsics(camera_name, source_type, entry):
        body_tform_camera = entry.get("body_tform_camera")

        if body_tform_camera is None:
            raise ValueError(
                f"Camera '{camera_name}' ({source_type}) is missing "
                f"'body_tform_camera'."
            )

        position = body_tform_camera.get("position")
        rotation = body_tform_camera.get("rotation")

        if position is None or rotation is None:
            raise ValueError(
                f"Camera '{camera_name}' ({source_type}): "
                f"'body_tform_camera' must have 'position' and 'rotation'."
            )

        required_position_fields = {"x", "y", "z"}
        required_rotation_fields = {"qx", "qy", "qz", "qw"}

        if not required_position_fields.issubset(position):
            raise ValueError(
                f"Camera '{camera_name}' ({source_type}): 'position' "
                f"must have x, y, z."
            )

        if not required_rotation_fields.issubset(rotation):
            raise ValueError(
                f"Camera '{camera_name}' ({source_type}): 'rotation' "
                f"must have qx, qy, qz, qw."
            )

        t = np.array(
            [position["x"], position["y"], position["z"]], dtype=np.float32
        )

        R = quaternion_to_rotation_matrix(
            rotation["qx"], rotation["qy"], rotation["qz"], rotation["qw"]
        )

        return R, t

    def get_intrinsics(self, camera_name):
        """Return {fx, fy, cx, cy, width, height, skew} for a camera."""

        camera = self.require_camera(camera_name)

        return {
            "fx": camera["fx"],
            "fy": camera["fy"],
            "cx": camera["cx"],
            "cy": camera["cy"],
            "width": camera["width"],
            "height": camera["height"],
            "skew": camera["skew"],
        }

    def get_intrinsics_matrix(self, camera_name):
        """Return the 3x3 camera matrix K for a camera."""

        camera = self.require_camera(camera_name)

        return np.array([
            [camera["fx"], camera["skew"], camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def get_extrinsics(self, camera_name):
        """Return (R, t): the static body -> camera transform."""

        camera = self.require_camera(camera_name)

        return camera["R"], camera["t"]

    def get_depth_scale(self, camera_name):
        return self.require_camera(camera_name)["depth_scale"]

    def get_frame_name(self, camera_name):
        return self.require_camera(camera_name)["frame_name"]

    def require_camera(self, camera_name):
        if camera_name not in self.cameras:
            raise ValueError(
                f"Camera '{camera_name}' was not loaded. Loaded cameras: "
                f"{list(self.cameras.keys())}"
            )

        return self.cameras[camera_name]

    def list_cameras(self):
        return list(self.cameras.keys())
