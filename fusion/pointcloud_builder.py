"""
Builds a semantically enriched point cloud from an RGB image
and its corresponding depth image.

Pipeline:

    RGB image
        |
        v
    SegFormer inference
        |
        v
    Class reduction
        |
        v
    RGB + depth association
        |
        v
    3D projection
        |
        v
    Semantic point cloud

The generated point cloud contains three parallel arrays:

    points_xyz
        Shape: (N, 3)
        Type: float32
        Coordinates [X, Y, Z] in the depth-camera frame.

    semantic_colors
        Shape: (N, 3)
        Type: uint8
        RGB color associated with the semantic class
        assigned to each point.

    semantic_probs
        Shape: (N, 16)
        Type: float32
        SegFormer probabilities for the 16 retained semantic
        classes.

The three arrays share the same first dimension, so row i
describes the same 3D point.

Semantic colors are obtained from PerceptionConfig rather than
from the original RGB image.

The semantic probabilities are preserved separately and can
later be used to compute the semantic traversability cost.

At this stage the point cloud is expressed in the depth-camera
coordinate frame. Robot/world transformations are handled later
by the pose provider.

CAMERA-AWARE: intrinsics and depth_scale are NOT fixed at
construction time anymore. One PointCloudBuilder is shared across
however many cameras are active (1, 2 or 4), and build() looks up
the right camera's calibration by name on every call, from the
shared CameraCalibrationConfig (fusion/camera_calibration_loader.py).

DEPTH SCALE: Spot reports depth_scale as a DIVISOR --
    depth_meters = raw_value / depth_scale
(see fusion/query_camera_extrinsics.py's own comment: "raw uint16
value / depth_scale = depth in meters"). The real calibration gives
depth_scale = 999.0 for every camera. This module used to multiply
by depth_scale, which was fine with a placeholder value shaped like
a meters-per-unit factor (e.g. 0.001) but silently produces
nonsense (depth inflated ~1000x) with the real value -- fixed to
divide.
"""

import cv2
import numpy as np

from perception.segformer_inference import SegFormerInference
from perception.class_reducer import ClassReducer


class PointCloudBuilder:

    def __init__(self, perception_config, camera_calibration):
        self.perception_config = perception_config
        self.camera_calibration = camera_calibration

        self.segformer = SegFormerInference(
            perception_config
        )

        self.class_reducer = ClassReducer(
            perception_config
        )

        # Intrinsics and depth_scale are no longer fixed here: with
        # several cameras sharing one builder, they are looked up
        # per camera on every build() call (see get_camera_params()).

        # Build the semantic palette corresponding to the filtered classes
        self.semantic_palette = self.build_semantic_palette()

    def build_semantic_palette(self):
        """
        Build the semantic RGB palette corresponding to the
        filtered probability channels.
        """
        return self.class_reducer.get_filtered_class_colors()

    def get_camera_params(self, camera_name):
        """
        Look up one camera's intrinsics + depth_scale from the
        shared CameraCalibrationConfig.

        Returns
        -------
        fx, fy, cx, cy, depth_scale : float
        """

        intrinsics = self.camera_calibration.get_intrinsics(camera_name)
        depth_scale = self.camera_calibration.get_depth_scale(camera_name)

        return (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
            float(depth_scale)
        )

    def build(self, rgb_image, depth_image, camera_name):
        """
        Build a semantic point cloud from an RGB image and
        its corresponding depth image.

        Parameters
        ----------
        rgb_image : numpy.ndarray
            RGB image with shape (H, W, 3).

        depth_image : numpy.ndarray
            Raw depth image with shape (H, W).

        camera_name : str
            Which camera this (rgb_image, depth_image) pair came
            from (e.g. "right"), used to look up that camera's
            intrinsics/depth_scale in camera_calibration.

        Returns
        -------
        dict
            Dictionary containing:

                "points_xyz"
                    (N, 3) float32

                "semantic_colors"
                    (N, 3) uint8

                "semantic_probs"
                    (N, 16) float32
        """

        self.validate_rgb_image(rgb_image)
        self.validate_depth_image(depth_image)

        fx, fy, cx, cy, depth_scale = self.get_camera_params(camera_name)

        depth_height, depth_width = depth_image.shape[:2]

        # SegFormer inference
        probabilities = self.segformer.infer(
            rgb_image
        )

        # probabilities:
        # (24, H_seg, W_seg)

        # Semantic class reduction
        filtered_probabilities = (
            self.class_reducer.filter_probabilities(
                probabilities
            )
        )

        # filtered_probabilities:
        # (16, H_seg, W_seg)

        # Resize semantic probabilities to depth resolution
        if (
            filtered_probabilities.shape[1] != depth_height
            or filtered_probabilities.shape[2] != depth_width
        ):
            filtered_probabilities = self.resize_probabilities(
                filtered_probabilities,
                depth_width,
                depth_height
            )

        # Convert depth to meters.
        #
        # depth_scale is a DIVISOR (raw_value / depth_scale =
        # meters), not a multiplier -- see module docstring.
        depth_meters = (
            depth_image.astype(np.float32)
            / depth_scale
        )

        # Valid depth mask
        valid_depth = (
            np.isfinite(depth_meters)
            & (depth_meters > 0)
            & (depth_meters < 4.0)
        )

        if not np.any(valid_depth):
            return self.empty_pointcloud()

        # Extract valid pixel coordinates
        v, u = np.nonzero(valid_depth)

        z = depth_meters[v, u]

        # Project depth pixels into 3D
        #
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x = (
            (u.astype(np.float32) - cx)
            * z
            / fx
        )

        y = (
            (v.astype(np.float32) - cy)
            * z
            / fy
        )

        points_xyz = np.column_stack(
            (x, y, z)
        ).astype(np.float32)

        # Associate semantic probabilities
        semantic_probs = filtered_probabilities[
            :,
            v,
            u
        ].T.astype(np.float32)

        # Determine the most probable semantic class
        #
        # semantic_probs:
        # (N, 16)
        #
        # semantic_class_ids:
        # (N,)
        semantic_class_ids = np.argmax(
            semantic_probs,
            axis=1
        )

        # Associate semantic RGB colors
        #
        # Instead of using:
        #
        #     rgb_image[v, u]
        #
        # use the RGB color assigned to the predicted
        # semantic class in PerceptionConfig.
        #
        # semantic_palette:
        # (16, 3)
        #
        # semantic_class_ids:
        # (N,)
        #
        # semantic_colors:
        # (N, 3)
        if np.any(
            semantic_class_ids >= len(self.semantic_palette)
        ):
            raise ValueError(
                "Semantic class ID exceeds the configured "
                "semantic color palette."
            )

        semantic_colors = (
            self.semantic_palette[
                semantic_class_ids
            ]
        ).astype(np.uint8)

        # Return semantic point cloud
        return {
            "points_xyz": points_xyz,
            "semantic_colors": semantic_colors,
            "semantic_probs": semantic_probs
        }

    @staticmethod
    def resize_probabilities(
        probabilities,
        width,
        height
    ):
        """
        Resize semantic probability maps to the target
        image resolution.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Shape (C, H, W).

        width : int
            Target width.

        height : int
            Target height.

        Returns
        -------
        numpy.ndarray
            Resized probabilities with shape
            (C, height, width).
        """

        num_classes = probabilities.shape[0]

        resized = np.empty(
            (num_classes, height, width),
            dtype=np.float32
        )

        for class_id in range(num_classes):

            resized[class_id] = cv2.resize(
                probabilities[class_id],
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

        return resized

    @staticmethod
    def empty_pointcloud():
        """
        Return an empty point cloud with the expected structure.
        """

        return {
            "points_xyz": np.empty(
                (0, 3),
                dtype=np.float32
            ),

            "semantic_colors": np.empty(
                (0, 3),
                dtype=np.uint8
            ),

            "semantic_probs": np.empty(
                (0, 16),
                dtype=np.float32
            )
        }

    @staticmethod
    def validate_rgb_image(rgb_image):
        """
        Validate the RGB input image.
        """

        if not isinstance(
            rgb_image,
            np.ndarray
        ):
            raise TypeError(
                "RGB image must be a numpy.ndarray."
            )

        if rgb_image.ndim != 3:
            raise ValueError(
                "RGB image must have shape (H, W, 3)."
            )

        if rgb_image.shape[2] != 3:
            raise ValueError(
                "RGB image must have exactly 3 channels."
            )

    @staticmethod
    def validate_depth_image(depth_image):
        """
        Validate the depth input image.
        """

        if not isinstance(
            depth_image,
            np.ndarray
        ):
            raise TypeError(
                "Depth image must be a numpy.ndarray."
            )

        if depth_image.ndim != 2:
            raise ValueError(
                "Depth image must have shape (H, W)."
            )

        if not np.issubdtype(
            depth_image.dtype,
            np.number
        ):
            raise TypeError(
                "Depth image must contain numeric values."
            )


import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from perception.perception_config_loader import PerceptionConfig
from fusion.camera_calibration_loader import CameraCalibrationConfig


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    perception_config_path = (
        PROJECT_ROOT / "config" / "perception_config.yaml"
    )

    camera_intrinsics_path = (
        PROJECT_ROOT / "config" / "camera_intrinsics.yaml"
    )

    # Which camera this test frame came from -- change this if you
    # are testing a different camera's pair of images.
    camera_name = "right"

    rgb_path = (
        PROJECT_ROOT / "testset" / "paired_images" / "000000.jpg"
    )

    depth_path = (
        PROJECT_ROOT / "testset" / "paired_depths" / "000000.png"
    )

    # -------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------

    perception_config = PerceptionConfig(
        perception_config_path
    )

    camera_calibration = CameraCalibrationConfig(
        camera_intrinsics_path
    )

    # -------------------------------------------------------------
    # Create point cloud builder
    # -------------------------------------------------------------

    builder = PointCloudBuilder(
        perception_config,
        camera_calibration
    )

    # -------------------------------------------------------------
    # Load RGB and depth
    # -------------------------------------------------------------

    rgb_image = np.array(
        Image.open(rgb_path).convert("RGB")
    )

    depth_image = np.array(
        Image.open(depth_path)
    )

    #rgb_image = np.rot90(rgb_image, k=-1)
    #depth_image = np.rot90(depth_image, k=-1)

    print("RGB shape:   ", rgb_image.shape)
    print("Depth shape: ", depth_image.shape)

    # -------------------------------------------------------------
    # Build semantic point cloud
    # -------------------------------------------------------------

    pointcloud = builder.build(
        rgb_image,
        depth_image,
        camera_name
    )

    # -------------------------------------------------------------
    # Extract outputs
    # -------------------------------------------------------------

    points_xyz = pointcloud["points_xyz"]
    semantic_colors = pointcloud["semantic_colors"]
    semantic_probs = pointcloud["semantic_probs"]

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\nPoint cloud generated")

    print("XYZ shape:             ", points_xyz.shape)
    print("Semantic colors shape: ", semantic_colors.shape)
    print("Semantic probs shape:  ", semantic_probs.shape)

    print("\nFirst 5 points:")
    print(points_xyz[:5])

    print("\nFirst 5 semantic colors:")
    print(semantic_colors[:5])

    print("\nFirst 5 semantic probabilities:")
    print(semantic_probs[:5])

    # -------------------------------------------------------------
    # Basic consistency checks
    # -------------------------------------------------------------

    assert points_xyz.shape[0] == semantic_colors.shape[0]
    assert points_xyz.shape[0] == semantic_probs.shape[0]

    assert points_xyz.shape[1] == 3
    assert semantic_colors.shape[1] == 3

    print("\nAll consistency checks passed.")

    # -------------------------------------------------------------
    # Reconstruct segmentation mask
    # -------------------------------------------------------------

    semantic_class_ids = np.argmax(
        semantic_probs,
        axis=1
    )

    # -------------------------------------------------------------
    # Create semantic image
    # -------------------------------------------------------------
    #
    # The point cloud only contains pixels with valid depth.
    # Therefore, we reconstruct an image-sized mask using the
    # original RGB/depth resolution.
    #
    # -------------------------------------------------------------

    height, width = depth_image.shape[:2]

    semantic_image = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # -------------------------------------------------------------
    # Project valid semantic colors back into image coordinates
    # -------------------------------------------------------------

    # Reconstruct the valid-depth mask
    valid_depth = (
        np.isfinite(depth_image)
        & (depth_image > 0)
    )

    valid_v, valid_u = np.where(valid_depth)

    # If the point cloud was generated from exactly the same
    # depth pixels, their order corresponds to these coordinates.
    semantic_image[valid_v, valid_u] = semantic_colors

    # -------------------------------------------------------------
    # Visualize RGB + segmentation + point cloud
    # -------------------------------------------------------------

    fig = plt.figure(figsize=(18, 6))

    # -------------------------------------------------------------
    # Original RGB
    # -------------------------------------------------------------

    ax1 = fig.add_subplot(1, 3, 1)

    ax1.imshow(rgb_image)

    ax1.set_title("RGB Image")
    ax1.axis("off")

    # -------------------------------------------------------------
    # Semantic segmentation
    # -------------------------------------------------------------

    ax2 = fig.add_subplot(1, 3, 2)

    ax2.imshow(semantic_image)

    ax2.set_title("Semantic Segmentation")
    ax2.axis("off")

    # -------------------------------------------------------------
    # Semantic point cloud
    # -------------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3,
        projection="3d"
    )

    if len(points_xyz) > 0:

        ax3.scatter(
            points_xyz[:, 0],
            points_xyz[:, 1],
            points_xyz[:, 2],
            c=semantic_colors / 255.0,
            s=1
        )

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    ax3.set_title("Semantic Point Cloud")

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Print struttura di un singolo punto
    # -------------------------------------------------------------

    if len(points_xyz) > 0:

        point_id = 0

        print("\nStruttura del punto", point_id)
        print("--------------------------------")

        print("XYZ:")
        print("  ", points_xyz[point_id])
        print("  dtype:", points_xyz.dtype)
        print("  shape:", points_xyz[point_id].shape)

        print("\nSemantic color:")
        print("  ", semantic_colors[point_id])
        print("  dtype:", semantic_colors.dtype)
        print("  shape:", semantic_colors[point_id].shape)

        print("\nSemantic probabilities:")
        print("  ", semantic_probs[point_id])
        print("  dtype:", semantic_probs.dtype)
        print("  shape:", semantic_probs[point_id].shape)

        # Classe semantica predetta
        semantic_class_id = np.argmax(
            semantic_probs[point_id]
        )

        print("\nPredicted semantic class ID:")
        print("  ", semantic_class_id)

    else:
        print("\nPoint cloud vuota.")