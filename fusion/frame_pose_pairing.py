"""
Pairs each (RGB, depth) frame with the closest pose recorded in the
odometry log.

RGB<->depth pairing is NOT redone here: Spot's SDK already reprojects
depth into the visual frame at capture time, so the RGB and depth
file for a given acquisition share the exact same timestamp in their
filename (see frame_pairing.py's docstring for the older, more
general nearest-timestamp approach, needed only when that is not the
case). This module only groups files by that shared timestamp.

Frame<->pose pairing, in contrast, IS inherently approximate: the
pose log and the camera are two independent, asynchronously sampled
streams, so the closest pose in time is used, and the resulting time
gap is reported explicitly (via a warning past a configurable
tolerance) rather than hidden.

POSE FRAME: each row of the odometry log stores the robot body's
pose in TWO world frames:

    - "odom"   : kinematic odometry (legs + IMU) only.
    - "vision" : kinematic + visual odometry; usually drifts less.

The two frames have different origins and orientations, so poses
from different frames must never be mixed. The pose_frame argument
selects which one is read for the whole sequence.

CAMERA POSE vs BODY POSE: the odometry log only records the robot
BODY's pose (<pose_frame>_tform_body), not the camera's own pose. By
default, pair() returns that body pose directly, used as an
approximation of the camera pose (the placeholder used everywhere in
this project so far). If a CameraCalibrationConfig + camera_name are
provided (the static body -> camera offset produced on the robot,
see fusion/camera_calibration_loader.py), pair() instead composes it
with each frame's body pose to return the TRUE camera pose in the
selected world frame:

    R_world_camera = R_world_body @ R_body_camera
    t_world_camera = R_world_body @ t_body_camera + t_world_body

Nothing downstream (GridMap, GeometricCost, ...) needs to change
either way -- they just consume whichever R, t this returns.

MULTI-CAMERA FOLDERS: spot_rgb_depth_log.py writes every camera of a
shot into the SAME folders with the SAME timestamp:

    <ts>_frontleft_fisheye_image.jpg   <ts>_frontleft_depth_in_visual_frame.png
    <ts>_right_fisheye_image.jpg       <ts>_right_depth_in_visual_frame.png
    ...

so a timestamp alone does NOT identify a file. When camera_name is
given, only that camera's files are globbed (by their Spot source
name), and two files mapping to the same timestamp raise instead of
silently overwriting each other -- which is what previously let
frontleft frames be processed with the right camera's calibration.

This script is intended for offline field-data testing only. It is
not part of the live Spot pipeline.
"""

import bisect
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fusion.geometry_utils import quaternion_to_rotation_matrix, compose_poses


SUPPORTED_POSE_FRAMES = ("odom", "vision")

# Filename suffixes written by spot_rgb_depth_log.py (Spot source
# names). The leading underscore matters: "*_left_..." must not match
# "<ts>_frontleft_...".
RGB_SUFFIX = "_fisheye_image.jpg"
DEPTH_SUFFIX = "_depth_in_visual_frame.png"


class FramePosePairer:
    """Pairs RGB/depth frame files with the closest logged pose."""

    def __init__(
        self,
        rgb_dir,
        depth_dir,
        pose_csv_path,
        max_pose_dt=1.0,
        camera_calibration=None,
        camera_name=None,
        pose_frame="odom"
    ):
        """
        Parameters
        ----------
        rgb_dir, depth_dir : str or Path
            Directories containing the RGB (.jpg) and depth (.png)
            frames for ONE camera. Files are matched across the two
            directories by the timestamp embedded in their filename
            (YYYYMMDD_HHMMSS_ffffff, 6-digit microseconds).

        pose_csv_path : str or Path
            Path to the odometry log CSV. Expected columns:
            timestamp, then for each of the "odom" and "vision"
            frames: <frame>_x, <frame>_y, <frame>_z, <frame>_qx,
            <frame>_qy, <frame>_qz, <frame>_qw (quaternion,
            scalar-last / ROS convention). Extra columns (e.g.
            acq_time_robot) are ignored. Only the columns of the
            frame selected by pose_frame are required.

        max_pose_dt : float
            Time gap, in seconds, past which a frame<->pose match
            triggers a warning instead of being accepted silently.

        camera_calibration : CameraCalibrationConfig, optional
            The multi-camera calibration (fusion/
            camera_calibration_loader.py). If given, camera_name
            must be given too: pair() then looks up that camera's
            static body -> camera extrinsic and returns the true
            camera pose (composed with each frame's body pose)
            instead of the body pose directly.

        camera_name : str, optional
            Which camera this instance's rgb_dir/depth_dir belong
            to (e.g. "right"), looked up in camera_calibration.
            Required together with camera_calibration -- one
            FramePosePairer instance still handles one camera's
            frames; for several cameras, build one instance per
            camera (same camera_calibration, different
            rgb_dir/depth_dir/camera_name).

        pose_frame : {"odom", "vision"}
            Which world frame to read the body pose in. "vision"
            (kinematic + visual odometry) usually drifts less than
            "odom" (kinematic only). Never mix the two within one
            sequence.
        """

        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.pose_csv_path = Path(pose_csv_path)
        self.max_pose_dt = max_pose_dt
        self.camera_name = camera_name

        if pose_frame not in SUPPORTED_POSE_FRAMES:
            raise ValueError(
                f"pose_frame must be one of {SUPPORTED_POSE_FRAMES}, "
                f"got {pose_frame!r}."
            )

        self.pose_frame = pose_frame

        if (camera_calibration is None) != (camera_name is None):
            raise ValueError(
                "camera_calibration and camera_name must be given "
                "together (both, or neither for the body-pose "
                "placeholder)."
            )

        if camera_calibration is None:
            self.R_body_camera = None
            self.t_body_camera = None
        else:
            self.R_body_camera, self.t_body_camera = (
                camera_calibration.get_extrinsics(camera_name)
            )

        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}"
            )

        if not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}"
            )

        if not self.pose_csv_path.exists():
            raise FileNotFoundError(
                f"Pose log not found: {self.pose_csv_path}"
            )

        # Sorted list of (timestamp_seconds, R, t), filled by
        # load_poses(). Kept sorted so find_closest_pose() can use
        # binary search instead of a linear scan.
        self.poses = []

    def load_poses(self):
        """
        Load and parse every row of the odometry log CSV, reading
        the body pose in the frame selected by self.pose_frame.
        """

        p = self.pose_frame

        position_fields = [f"{p}_x", f"{p}_y", f"{p}_z"]
        quaternion_fields = [f"{p}_qx", f"{p}_qy", f"{p}_qz", f"{p}_qw"]

        required_fields = {
            "timestamp", *position_fields, *quaternion_fields
        }

        with open(
            self.pose_csv_path, "r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file)

            if (
                reader.fieldnames is None
                or not required_fields.issubset(reader.fieldnames)
            ):
                missing = required_fields - set(reader.fieldnames or [])

                raise ValueError(
                    f"Pose log {self.pose_csv_path} is missing "
                    f"required columns for pose_frame='{p}': "
                    f"{sorted(missing)}. Found: {reader.fieldnames}"
                )

            poses = []

            for row in reader:
                timestamp = self.extract_timestamp(row["timestamp"])

                if timestamp is None:
                    raise ValueError(
                        "Could not parse timestamp in pose log: "
                        f"{row['timestamp']}"
                    )

                t = np.array(
                    [float(row[field]) for field in position_fields],
                    dtype=np.float32
                )

                R = quaternion_to_rotation_matrix(
                    *(float(row[field]) for field in quaternion_fields)
                )

                poses.append((timestamp, R, t))

        if len(poses) == 0:
            raise ValueError(
                f"Pose log {self.pose_csv_path} contains no rows."
            )

        poses.sort(key=lambda pose: pose[0])

        self.poses = poses

    def find_frames(self):
        """
        Group RGB and depth files by their shared timestamp.

        Returns
        -------
        list of (timestamp, rgb_path, depth_path), sorted by
        timestamp. Files that don't have a match in the other
        directory are skipped, with a warning.
        """

        if self.camera_name is None:
            # Legacy single-camera folders: every file belongs to the
            # same (unnamed) camera.
            rgb_pattern, depth_pattern = "*.jpg", "*.png"
        else:
            rgb_pattern = f"*_{self.camera_name}{RGB_SUFFIX}"
            depth_pattern = f"*_{self.camera_name}{DEPTH_SUFFIX}"

        rgb_files = sorted(self.rgb_dir.glob(rgb_pattern))
        depth_files = sorted(self.depth_dir.glob(depth_pattern))

        if self.camera_name is not None and not rgb_files:
            others = sorted({
                p.name.split("_", 3)[-1]
                for p in self.rgb_dir.glob("*.jpg")
            })
            raise ValueError(
                f"No RGB files for camera '{self.camera_name}' "
                f"(pattern {rgb_pattern}) in {self.rgb_dir}. "
                f"Sources found there: {others}"
            )

        rgb_by_timestamp = self.index_by_timestamp(rgb_files)
        depth_by_timestamp = self.index_by_timestamp(depth_files)

        common = sorted(
            set(rgb_by_timestamp) & set(depth_by_timestamp)
        )

        missing_depth = sorted(
            set(rgb_by_timestamp) - set(depth_by_timestamp)
        )

        missing_rgb = sorted(
            set(depth_by_timestamp) - set(rgb_by_timestamp)
        )

        if missing_depth:
            print(
                f"[FramePosePairer] WARNING: {len(missing_depth)} "
                "RGB frame(s) have no matching depth file, skipped."
            )

        if missing_rgb:
            print(
                f"[FramePosePairer] WARNING: {len(missing_rgb)} "
                "depth frame(s) have no matching RGB file, skipped."
            )

        return [
            (ts, rgb_by_timestamp[ts], depth_by_timestamp[ts])
            for ts in common
        ]

    def index_by_timestamp(self, file_paths):
        """Map each file's parsed timestamp to its path."""

        index = {}

        for path in file_paths:
            timestamp = self.extract_timestamp(path.name)

            if timestamp is None:
                print(
                    "[FramePosePairer] WARNING: could not parse a "
                    f"timestamp from {path.name}, skipping."
                )
                continue

            # Rounded to microsecond precision so two independently
            # parsed floats for the exact same timestamp string
            # always land on the same dict key.
            key = round(timestamp, 6)

            if key in index:
                raise ValueError(
                    f"Two files share timestamp {path.name[:22]}: "
                    f"{index[key].name} and {path.name}. The folder "
                    "holds several cameras -- pass camera_name so "
                    "only one camera's files are selected."
                )

            index[key] = path

        return index

    def find_closest_pose(self, timestamp):
        """
        Find the pose closest in time to the given timestamp.

        Returns
        -------
        R, t, dt : the pose's rotation matrix, translation, and the
        absolute time gap (seconds) to the given timestamp.
        """

        timestamps = [pose[0] for pose in self.poses]

        idx = bisect.bisect_left(timestamps, timestamp)

        candidates = []

        if idx < len(timestamps):
            candidates.append(idx)

        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(
            candidates,
            key=lambda i: abs(timestamps[i] - timestamp)
        )

        best_timestamp, R, t = self.poses[best_idx]

        return R, t, abs(best_timestamp - timestamp)

    def pair(self):
        """
        Run the full pairing: load poses, group RGB/depth frames,
        and attach the closest pose to each -- composed with the
        camera extrinsics into the true camera pose, if a
        camera_calibration/camera_name was provided at construction
        time.

        Returns
        -------
        list of dict, sorted by timestamp. Each dict has keys:
        "timestamp", "rgb_path", "depth_path", "R", "t", "pose_dt".
        """

        self.load_poses()

        frames = self.find_frames()

        if len(frames) == 0:
            raise ValueError(
                "No matching RGB/depth pairs found in "
                f"{self.rgb_dir} / {self.depth_dir}."
            )

        results = []

        for timestamp, rgb_path, depth_path in frames:
            R_body, t_body, dt = self.find_closest_pose(timestamp)

            if dt > self.max_pose_dt:
                print(
                    f"[FramePosePairer] WARNING: {rgb_path.name} is "
                    f"{dt:.3f}s from its closest pose "
                    f"(> {self.max_pose_dt}s tolerance). Using it "
                    "anyway -- check for gaps in the odometry log."
                )

            if self.R_body_camera is None:
                # Placeholder: body pose used directly as the
                # camera's pose.
                R, t = R_body, t_body
            else:
                # True camera pose: <pose_frame>_tform_body composed
                # with the static body_tform_camera.
                R, t = compose_poses(
                    R_body, t_body,
                    self.R_body_camera, self.t_body_camera
                )

            results.append({
                "timestamp": timestamp,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "R": R,
                "t": t,
                "pose_dt": dt
            })

        max_dt = max(result["pose_dt"] for result in results)

        pose_kind = (
            "body pose (placeholder)"
            if self.R_body_camera is None
            else f"true camera pose ({self.camera_name})"
        )

        print(
            f"[FramePosePairer] Paired {len(results)} frame(s) with "
            f"poses ({pose_kind}, frame: {self.pose_frame}, "
            f"max time gap: {max_dt:.3f}s)."
        )

        return results

    @staticmethod
    def extract_timestamp(text):
        """
        Extract a timestamp (seconds, float) from a Spot filename or
        odometry log timestamp string of the form:

            YYYYMMDD_HHMMSS_ffffff  (6-digit microseconds)

        Returns None if no match is found.
        """

        match = re.search(r"(\d{8})_(\d{6})_(\d{6})", text)

        if match is None:
            return None

        date_str, time_str, us_str = match.groups()

        dt = datetime.strptime(
            f"{date_str}_{time_str}",
            "%Y%m%d_%H%M%S"
        )

        return dt.timestamp() + float(us_str) / 1e6


if __name__ == "__main__":

    from fusion.camera_calibration_loader import CameraCalibrationConfig

    camera_calibration = CameraCalibrationConfig(
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    pairer = FramePosePairer(
        rgb_dir=PROJECT_ROOT / "testset" / "images",
        depth_dir=PROJECT_ROOT / "testset" / "depths",
        pose_csv_path=PROJECT_ROOT / "testset" / "pose" / "odometry_log.csv",
        max_pose_dt=1.0,
        camera_calibration=camera_calibration,
        camera_name="frontleft",
        pose_frame="odom"
    )

    paired_frames = pairer.pair()

    print("\n" + "-" * 60)
    print("Paired frames")
    print("-" * 60)

    for frame in paired_frames:
        print(f"{frame['rgb_path'].name}")
        print(f"  depth      : {frame['depth_path'].name}")
        print(f"  pose dt    : {frame['pose_dt']:.3f} s")
        print(f"  t          : {frame['t']}")
        print()