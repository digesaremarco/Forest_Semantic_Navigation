"""
Spot image acquisition for the live pipeline: one synchronized
get_image() call for the visual (RGB) and depth_in_visual_frame
sources of every configured camera, decoded into exactly the arrays
TraversabilityPipeline.update() expects, plus each camera's pose.

FORMAT -- identical to the data the model was validated on
(spot_rgb_depth_log.py): visual sources requested as RGB_U8 JPEG at
the configured quality; depth_in_visual_frame as DEPTH_U16 in the
robot's native encoding (raw or PNG, both decoded). The RGB is
converted BGR -> RGB after cv2.imdecode: offline this happened
implicitly (cv2.imwrite, then PIL.Image.open), live the array is
handed over directly, and a missing conversion would silently feed
SegFormer swapped channels. A camera that cannot deliver RGB is a
hard error: the network needs 3 channels.

ORIENTATION: images are returned in the camera's NATIVE raster and
never rotated -- PointCloudBuilder turns the RGB upright for
SegFormer only (UPRIGHT_ROT90_K); K and the extrinsics refer to the
native raster.

POSE: (R, t) of each camera is vision_tform_camera, read from the
transforms_snapshot of THAT camera's visual response, with the
camera frame given by frame_name_image_sensor -- the same frame
camera_intrinsics.yaml's body_tform_camera points to (checked
against get_frame_name()). This replaces the offline composition
odom_tform_body (CSV) @ body_tform_camera (YAML): same chain, but
evaluated by the robot at the image's own acquisition time, so pose
and image are synchronized by construction. depth_in_visual_frame is
registered into the visual frame, so the visual pose is the depth
pose too (frame names checked equal).

CALIBRATION CHECK: verify_calibration() compares, per camera, image
size, pinhole intrinsics and body_tform_camera reported live by the
robot with camera_intrinsics.yaml (which PointCloudBuilder uses for
the backprojection). A mismatch means the YAML is stale or comes
from another robot; per calibration_check.on_mismatch it is printed
or raised.

RETRIES: capture() retries the whole capture (RPC error, missing
source, inconsistent response) up to capture.max_retries times, then
raises AcquisitionError -- the main loop decides what to do with it.
"""

import time
from datetime import datetime

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.exceptions import Error as BosdynError
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.image import build_image_request

from acquisition.acquisition_snapshot import AcquisitionSnapshot, CameraObservation
from spot.frame_utils import rotation_angle_deg, se3_pose_to_rt


PF = image_pb2.Image  # shorthand for the Format / PixelFormat enums


class AcquisitionError(RuntimeError):
    """A capture could not produce a complete, consistent snapshot."""


class ImageAcquirer:

    def __init__(self, acquisition_config, camera_calibration, image_client):
        self.acquisition_config = acquisition_config
        self.camera_calibration = camera_calibration
        self.image_client = image_client

        loaded = set(camera_calibration.list_cameras())

        missing = [
            name for name in acquisition_config.camera_names
            if name not in loaded
        ]

        if missing:
            raise ValueError(
                f"Camera(s) {missing} are acquired but not loaded by "
                f"CameraCalibrationConfig (loaded: {sorted(loaded)}). "
                "Pass camera_names=acquisition_config.camera_names to it."
            )

        self.image_requests = None

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def initialize(self):
        """
        Check that every configured source exists and supports the
        required pixel format, and build the image requests.
        """

        available = {
            source.name: source
            for source in self.image_client.list_image_sources()
        }

        requests = []

        for camera_name in self.acquisition_config.camera_names:
            visual_name = self.acquisition_config.visual_source_name(camera_name)
            depth_name = self.acquisition_config.depth_source_name(camera_name)

            for source_name in (visual_name, depth_name):
                if source_name not in available:
                    raise ValueError(
                        f"Image source '{source_name}' not available on "
                        "this robot."
                    )

            formats = list(getattr(available[visual_name], "pixel_formats", []))

            # An empty list means the robot does not advertise its
            # formats: request RGB anyway, decode_visual() checks it.
            if formats and PF.PIXEL_FORMAT_RGB_U8 not in formats:
                raise ValueError(
                    f"'{visual_name}' does not support RGB_U8 (greyscale "
                    "sensor?). The segmentation model needs RGB."
                )

            requests.append(build_image_request(
                visual_name,
                quality_percent=self.acquisition_config.jpeg_quality,
                image_format=PF.FORMAT_JPEG,
                pixel_format=PF.PIXEL_FORMAT_RGB_U8
            ))

            requests.append(build_image_request(
                depth_name,
                pixel_format=PF.PIXEL_FORMAT_DEPTH_U16
            ))

        self.image_requests = requests

        print(
            f"[ImageAcquirer] Ready: {len(requests)} sources for cameras "
            f"{', '.join(self.acquisition_config.camera_names)} "
            f"(JPEG q={self.acquisition_config.jpeg_quality})."
        )

    # -----------------------------------------------------------------
    # Capture
    # -----------------------------------------------------------------

    def capture(self, settle=True):
        """
        Capture one snapshot of every configured camera.

        Parameters
        ----------
        settle : bool
            Wait capture.settle_time_s first (after a locomotion
            command, so the body has stopped swaying).

        Returns
        -------
        AcquisitionSnapshot

        Raises
        ------
        AcquisitionError
            If every attempt failed.
        """

        if settle and self.acquisition_config.settle_time_s > 0:
            time.sleep(self.acquisition_config.settle_time_s)

        return self.with_retries(self.capture_once, "Capture")

    def capture_once(self):
        responses = self.request_images()

        snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wall_time = time.time()

        cameras = {}

        for camera_name in self.acquisition_config.camera_names:
            visual = self.get_response(
                responses, self.acquisition_config.visual_source_name(camera_name)
            )
            depth = self.get_response(
                responses, self.acquisition_config.depth_source_name(camera_name)
            )

            cameras[camera_name] = self.build_camera_observation(
                camera_name, visual, depth
            )

        reference = self.get_response(
            responses,
            self.acquisition_config.visual_source_name(
                self.acquisition_config.camera_names[0]
            )
        ).shot

        vision_tform_body = self.lookup_pose(
            reference.transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME
        )

        try:
            odom_tform_body = self.lookup_pose(
                reference.transforms_snapshot, ODOM_FRAME_NAME, BODY_FRAME_NAME
            )
        except AcquisitionError:
            odom_tform_body = None

        return AcquisitionSnapshot(
            snapshot_id=snapshot_id,
            wall_time=wall_time,
            acquisition_time=self.timestamp_to_seconds(reference.acquisition_time),
            cameras=cameras,
            vision_tform_body=vision_tform_body,
            odom_tform_body=odom_tform_body
        )

    def build_camera_observation(self, camera_name, visual, depth):
        """Decode and validate one camera's visual + depth responses."""

        frame_name = visual.shot.frame_name_image_sensor
        expected_frame = self.camera_calibration.get_frame_name(camera_name)

        if frame_name != expected_frame:
            raise AcquisitionError(
                f"Camera '{camera_name}': image frame '{frame_name}' differs "
                f"from the calibration frame '{expected_frame}'."
            )

        depth_frame = depth.shot.frame_name_image_sensor

        if depth_frame != frame_name:
            raise AcquisitionError(
                f"Camera '{camera_name}': depth frame '{depth_frame}' differs "
                f"from the visual frame '{frame_name}' -- depth is not "
                "registered into the visual frame."
            )

        rgb, rgb_encoded = self.decode_visual(visual)
        depth_image = self.decode_depth(depth)

        intrinsics = self.camera_calibration.get_intrinsics(camera_name)
        expected_shape = (int(intrinsics["height"]), int(intrinsics["width"]))

        if rgb.shape[:2] != expected_shape or depth_image.shape != expected_shape:
            raise AcquisitionError(
                f"Camera '{camera_name}': rgb {rgb.shape[:2]} / depth "
                f"{depth_image.shape} do not match the calibrated native "
                f"raster {expected_shape} (HxW)."
            )

        visual_time = self.timestamp_to_seconds(visual.shot.acquisition_time)
        depth_time = self.timestamp_to_seconds(depth.shot.acquisition_time)

        if abs(visual_time - depth_time) > self.acquisition_config.max_visual_depth_dt_s:
            raise AcquisitionError(
                f"Camera '{camera_name}': visual and depth acquired "
                f"{abs(visual_time - depth_time) * 1000:.0f} ms apart "
                f"(max {self.acquisition_config.max_visual_depth_dt_s * 1000:.0f} ms)."
            )

        R, t = self.lookup_pose(
            visual.shot.transforms_snapshot, VISION_FRAME_NAME, frame_name
        )

        return CameraObservation(
            camera_name=camera_name,
            frame_name=frame_name,
            rgb=rgb,
            depth=depth_image,
            R=R,
            t=t,
            acquisition_time=visual_time,
            depth_acquisition_time=depth_time,
            rgb_encoded=rgb_encoded
        )

    def request_images(self):
        """One get_image() call; {source_name: ImageResponse}."""

        if self.image_requests is None:
            raise RuntimeError("ImageAcquirer not initialized: call initialize() first.")

        try:
            responses = self.image_client.get_image(self.image_requests)
        except BosdynError as error:
            raise AcquisitionError(f"get_image() failed: {error}") from error

        return {response.source.name: response for response in responses}

    def with_retries(self, function, description):
        last_error = None
        attempts = self.acquisition_config.max_retries

        for attempt in range(1, attempts + 1):
            try:
                return function()
            except AcquisitionError as error:
                last_error = error
                print(f"[ImageAcquirer] {description} attempt {attempt}/{attempts} failed: {error}")

                if attempt < attempts:
                    time.sleep(self.acquisition_config.retry_delay_s)

        raise AcquisitionError(
            f"{description} failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    # -----------------------------------------------------------------
    # Calibration check
    # -----------------------------------------------------------------

    def verify_calibration(self):
        """
        Compare live image size, intrinsics and body_tform_camera with
        camera_intrinsics.yaml, for every configured camera.

        Returns
        -------
        dict or None
            Per-camera report (errors + list of problems), or None if
            the check is disabled.

        Raises
        ------
        ValueError
            On mismatch, if calibration_check.on_mismatch is "error".
        """

        if not self.acquisition_config.calibration_check_enabled:
            print("[ImageAcquirer] Calibration check disabled.")
            return None

        responses = self.with_retries(self.request_images, "Calibration check capture")

        report = {}
        problems = []

        print("\n[ImageAcquirer] Calibration check (live robot vs camera_intrinsics.yaml):")

        for camera_name in self.acquisition_config.camera_names:
            visual = self.get_response(
                responses, self.acquisition_config.visual_source_name(camera_name)
            )

            entry = self.compare_camera(camera_name, visual)
            report[camera_name] = entry

            intrinsics_text = (
                "n/a" if entry["intrinsics_error_px"] is None
                else f"{entry['intrinsics_error_px']:.3f} px"
            )

            print(
                f"  {camera_name:<12} size {entry['live_size']}, "
                f"intrinsics {intrinsics_text}, "
                f"extrinsic {entry['translation_error_m'] * 1000:.1f} mm / "
                f"{entry['rotation_error_deg']:.3f} deg"
                + ("" if not entry["problems"] else "  <-- MISMATCH")
            )

            problems += [f"{camera_name}: {problem}" for problem in entry["problems"]]

        if problems:
            message = (
                "Live calibration differs from camera_intrinsics.yaml:\n  "
                + "\n  ".join(problems)
            )

            if self.acquisition_config.calibration_on_mismatch == "error":
                raise ValueError(message)

            print(f"[ImageAcquirer] WARNING: {message}")
        else:
            print("  All cameras consistent with the YAML.")

        return report

    def compare_camera(self, camera_name, visual):
        config = self.acquisition_config
        calibration = self.camera_calibration

        problems = []

        # Size
        intrinsics = calibration.get_intrinsics(camera_name)
        image = visual.shot.image
        live_size = (int(image.rows), int(image.cols))
        expected_size = (int(intrinsics["height"]), int(intrinsics["width"]))

        if live_size != expected_size:
            problems.append(f"image size {live_size} vs YAML {expected_size} (HxW)")

        # Intrinsics
        intrinsics_error = None

        if visual.source.HasField("pinhole"):
            live = visual.source.pinhole.intrinsics

            intrinsics_error = max(
                abs(live.focal_length.x - intrinsics["fx"]),
                abs(live.focal_length.y - intrinsics["fy"]),
                abs(live.principal_point.x - intrinsics["cx"]),
                abs(live.principal_point.y - intrinsics["cy"]),
            )

            if intrinsics_error > config.max_intrinsics_error_px:
                problems.append(f"intrinsics differ by up to {intrinsics_error:.3f} px")
        else:
            problems.append("live source reports no pinhole model")

        # Extrinsics
        R_yaml, t_yaml = calibration.get_extrinsics(camera_name)

        R_live, t_live = self.lookup_pose(
            visual.shot.transforms_snapshot,
            BODY_FRAME_NAME,
            visual.shot.frame_name_image_sensor
        )

        translation_error = float(
            np.linalg.norm(t_live - np.asarray(t_yaml, dtype=np.float64))
        )
        rotation_error = rotation_angle_deg(R_yaml, R_live)

        if translation_error > config.max_translation_error_m:
            problems.append(f"body_tform_camera translation off by {translation_error * 1000:.1f} mm")

        if rotation_error > config.max_rotation_error_deg:
            problems.append(f"body_tform_camera rotation off by {rotation_error:.3f} deg")

        return {
            "live_size": live_size,
            "intrinsics_error_px": intrinsics_error,
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "problems": problems,
        }

    # -----------------------------------------------------------------
    # Decoding and helpers
    # -----------------------------------------------------------------

    @staticmethod
    def decode_visual(response):
        """(rgb (H, W, 3) uint8 RGB, original JPEG bytes or None)."""

        image = response.shot.image
        data = np.frombuffer(image.data, dtype=np.uint8)

        if image.format == PF.FORMAT_JPEG:
            decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

            if decoded is None:
                raise AcquisitionError(
                    f"'{response.source.name}': JPEG decoding failed."
                )

            if decoded.ndim != 3 or decoded.shape[2] != 3:
                raise AcquisitionError(
                    f"'{response.source.name}': decoded to shape "
                    f"{decoded.shape}, expected 3 channels (greyscale "
                    "image returned?)."
                )

            # cv2 decodes to BGR; the pipeline expects RGB.
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

            return rgb, bytes(image.data)

        if image.format == PF.FORMAT_RAW and image.pixel_format == PF.PIXEL_FORMAT_RGB_U8:
            rgb = data.reshape(image.rows, image.cols, 3).copy()

            return rgb, None

        raise AcquisitionError(
            f"'{response.source.name}': unsupported visual encoding "
            f"{PF.Format.Name(image.format)} / "
            f"{PF.PixelFormat.Name(image.pixel_format)}."
        )

    @staticmethod
    def decode_depth(response):
        """(H, W) uint16 raw depth."""

        image = response.shot.image

        if image.pixel_format != PF.PIXEL_FORMAT_DEPTH_U16:
            raise AcquisitionError(
                f"'{response.source.name}': pixel format "
                f"{PF.PixelFormat.Name(image.pixel_format)}, expected DEPTH_U16."
            )

        if image.format == PF.FORMAT_RAW:
            return np.frombuffer(image.data, dtype=np.uint16).reshape(
                image.rows, image.cols
            ).copy()

        depth = cv2.imdecode(
            np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )

        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise AcquisitionError(
                f"'{response.source.name}': could not decode a 2D uint16 "
                f"depth image ({PF.Format.Name(image.format)})."
            )

        return depth

    @staticmethod
    def get_response(responses, source_name):
        if source_name not in responses:
            raise AcquisitionError(
                f"No response for '{source_name}' "
                f"(got: {sorted(responses.keys())})."
            )

        return responses[source_name]

    @staticmethod
    def lookup_pose(snapshot, parent_frame, child_frame):
        """(R, t) of child_frame in parent_frame from a transforms_snapshot."""

        pose = get_a_tform_b(snapshot, parent_frame, child_frame)

        if pose is None:
            raise AcquisitionError(
                f"transforms_snapshot has no path {parent_frame} -> {child_frame}."
            )

        return se3_pose_to_rt(pose)

    @staticmethod
    def timestamp_to_seconds(timestamp):
        return timestamp.seconds + timestamp.nanos * 1e-9