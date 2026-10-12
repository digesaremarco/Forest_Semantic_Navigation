"""
Data contracts produced by ImageAcquirer and consumed by the main
loop, ObservationLogger and TraversabilityPipeline.

CameraObservation holds exactly what TraversabilityPipeline.update()
needs for one camera -- update(obs.rgb, obs.depth, obs.R, obs.t,
obs.camera_name) -- plus timing and the original JPEG bytes (saved
as-is by ObservationLogger, so the file on disk is bit-identical to
what was fused, without a second lossy encoding).

AcquisitionSnapshot groups the cameras of ONE get_image() call (one
stop of the stop-and-go loop) with the body pose at that moment.

All poses are (R, t) pairs: R (3, 3), t (3,), float64.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraObservation:
    camera_name: str

    # Spot frame of the camera (frame_name_image_sensor), e.g.
    # "frontleft_fisheye" -- the frame R, t refer to.
    frame_name: str

    # (H, W, 3) uint8, RGB order, NATIVE raster (not rotated).
    rgb: np.ndarray

    # (H, W) uint16, raw depth_in_visual_frame values (meters =
    # value / depth_scale), NATIVE raster, pixel-aligned with rgb.
    depth: np.ndarray

    # vision_tform_camera at the visual image's acquisition time.
    R: np.ndarray
    t: np.ndarray

    # Robot clock [s].
    acquisition_time: float
    depth_acquisition_time: float

    # Original JPEG bytes as returned by the robot (None if the robot
    # returned RAW pixels).
    rgb_encoded: Optional[bytes] = None


@dataclass
class AcquisitionSnapshot:
    # Wall-clock id, "%Y%m%d_%H%M%S_%f" -- same format as the file
    # names written by spot_rgb_depth_log.py.
    snapshot_id: str

    # Local wall clock [s, epoch].
    wall_time: float

    # Robot clock [s] of the reference (first) visual image.
    acquisition_time: float

    cameras: Dict[str, CameraObservation]

    # Body pose at the reference image's acquisition time. odom is
    # only logged (the live pipeline works in vision).
    vision_tform_body: Tuple[np.ndarray, np.ndarray]
    odom_tform_body: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def body_position_xy(self):
        """(x, y) of the body in the vision frame [m]."""

        _, t = self.vision_tform_body

        return float(t[0]), float(t[1])

    def camera_names(self):
        return list(self.cameras.keys())
