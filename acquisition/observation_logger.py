"""
Writes every AcquisitionSnapshot to disk, so a live mission can be
replayed offline with the existing tooling and doubles as a field
dataset for the thesis validation.

LAYOUT (one folder per run):

    <run_dir>/
        images_rgb/<camera>/<snapshot_id>_<camera>_fisheye_image.jpg
        depth_data/<camera>/<snapshot_id>_<camera>_depth_in_visual_frame.png
        odometry/odometry_log.csv
        poses/<snapshot_id>.json

images_rgb/, depth_data/ and odometry/ follow spot_rgb_depth_log.py
exactly (same file names, same CSV columns: body pose in odom and
vision at the snapshot), already split into per-camera folders as
test_grid_map.py expects -- so a run can be fed to test_grid_map.py
as DATA_ROOT unchanged.

poses/<snapshot_id>.json additionally stores, per camera, the exact
vision_tform_camera used live (4x4) and the acquisition times: the
ground truth of what was fused, independent of the YAML extrinsics.

RGB is written as the original JPEG bytes returned by the robot (no
re-encoding, bit-identical to what was fused); RAW RGB, if ever
returned, as lossless PNG. Depth as 16-bit PNG.

Every write opens and closes its file, so a crash mid-mission loses
at most the snapshot being written.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from spot.frame_utils import rotation_matrix_to_quaternion, rt_to_matrix


class ObservationLogger:

    ODOMETRY_COLUMNS = (
        "timestamp", "acq_time_robot",
        "odom_x", "odom_y", "odom_z",
        "odom_qx", "odom_qy", "odom_qz", "odom_qw",
        "vision_x", "vision_y", "vision_z",
        "vision_qx", "vision_qy", "vision_qz", "vision_qw",
    )

    def __init__(self, acquisition_config, run_dir):
        self.acquisition_config = acquisition_config
        self.run_dir = Path(run_dir)

        self.rgb_root = self.run_dir / "images_rgb"
        self.depth_root = self.run_dir / "depth_data"
        self.odometry_dir = self.run_dir / "odometry"
        self.poses_dir = self.run_dir / "poses"

        for camera_name in acquisition_config.camera_names:
            (self.rgb_root / camera_name).mkdir(parents=True, exist_ok=True)
            (self.depth_root / camera_name).mkdir(parents=True, exist_ok=True)

        self.odometry_dir.mkdir(parents=True, exist_ok=True)
        self.poses_dir.mkdir(parents=True, exist_ok=True)

        self.odometry_path = self.odometry_dir / "odometry_log.csv"

        if not self.odometry_path.exists():
            with open(self.odometry_path, "w", encoding="utf-8") as file:
                file.write(",".join(self.ODOMETRY_COLUMNS) + "\n")

        self.logged_count = 0

    @staticmethod
    def create_run_dir(output_root, prefix="spot_run"):
        """New timestamped run folder under output_root."""

        run_dir = Path(output_root) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)

        return run_dir

    def log(self, snapshot):
        """
        Write one snapshot (all cameras + odometry row + pose record).

        Returns
        -------
        Path of the snapshot's pose record (JSON).
        """

        camera_entries = {}

        for camera_name, observation in snapshot.cameras.items():
            rgb_path = self.write_rgb(snapshot.snapshot_id, observation)
            depth_path = self.write_depth(snapshot.snapshot_id, observation)

            camera_entries[camera_name] = {
                "frame_name": observation.frame_name,
                "acquisition_time_robot": observation.acquisition_time,
                "depth_acquisition_time_robot": observation.depth_acquisition_time,
                "rgb_file": str(rgb_path.relative_to(self.run_dir)),
                "depth_file": str(depth_path.relative_to(self.run_dir)),
                "vision_tform_camera": rt_to_matrix(observation.R, observation.t).tolist(),
            }

        self.append_odometry_row(snapshot)

        pose_path = self.write_pose_record(snapshot, camera_entries)

        self.logged_count += 1

        return pose_path

    def write_rgb(self, snapshot_id, observation):
        source_name = self.acquisition_config.visual_source_name(observation.camera_name)
        folder = self.rgb_root / observation.camera_name

        if observation.rgb_encoded is not None:
            path = folder / f"{snapshot_id}_{source_name}.jpg"
            path.write_bytes(observation.rgb_encoded)

            return path

        path = folder / f"{snapshot_id}_{source_name}.png"

        if not cv2.imwrite(str(path), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)):
            raise IOError(f"Could not write {path}")

        return path

    def write_depth(self, snapshot_id, observation):
        source_name = self.acquisition_config.depth_source_name(observation.camera_name)
        path = self.depth_root / observation.camera_name / f"{snapshot_id}_{source_name}.png"

        if not cv2.imwrite(str(path), observation.depth):
            raise IOError(f"Could not write {path}")

        return path

    def append_odometry_row(self, snapshot):
        values = [snapshot.snapshot_id, f"{snapshot.acquisition_time:.6f}"]

        for pose in (snapshot.odom_tform_body, snapshot.vision_tform_body):
            if pose is None:
                values += ["nan"] * 7
                continue

            R, t = pose
            quaternion = rotation_matrix_to_quaternion(R)

            values += [f"{value:.6f}" for value in (*t, *quaternion)]

        with open(self.odometry_path, "a", encoding="utf-8") as file:
            file.write(",".join(values) + "\n")

    def write_pose_record(self, snapshot, camera_entries):
        record = {
            "snapshot_id": snapshot.snapshot_id,
            "wall_time": snapshot.wall_time,
            "acquisition_time_robot": snapshot.acquisition_time,
            "vision_tform_body": rt_to_matrix(*snapshot.vision_tform_body).tolist(),
            "odom_tform_body": (
                None if snapshot.odom_tform_body is None
                else rt_to_matrix(*snapshot.odom_tform_body).tolist()
            ),
            "cameras": camera_entries,
        }

        path = self.poses_dir / f"{snapshot.snapshot_id}.json"

        with open(path, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

        return path