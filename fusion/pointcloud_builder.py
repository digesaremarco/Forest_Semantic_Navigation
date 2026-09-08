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
"""

import cv2
import numpy as np

from perception.segformer_inference import SegFormerInference
from perception.semantic_cost import ClassReducer


class PointCloudBuilder:

    def __init__(self, perception_config, camera_config):
        self.perception_config = perception_config
        self.camera_config = camera_config

        self.segformer = SegFormerInference(
            perception_config
        )

        self.class_reducer = ClassReducer(
            perception_config
        )

        # Depth camera intrinsics
        intrinsics = camera_config.get_depth_intrinsics()

        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["cx"])
        self.cy = float(intrinsics["cy"])

        # Depth scale
        self.depth_scale = float(
            camera_config.get_depth_scale()
        )

        # Build the semantic palette corresponding to the filtered classes
        self.semantic_palette = self.build_semantic_palette()

    def build_semantic_palette(self):
        """
        Build the semantic RGB palette corresponding to the
        filtered probability channels.
        """
        return self.class_reducer.get_filtered_class_colors()

    def build(self, rgb_image, depth_image):
        """
        Build a semantic point cloud from an RGB image and
        its corresponding depth image.

        Parameters
        ----------
        rgb_image : numpy.ndarray
            RGB image with shape (H, W, 3).

        depth_image : numpy.ndarray
            Raw depth image with shape (H, W).

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

        # Convert depth to meters
        depth_meters = (
            depth_image.astype(np.float32)
            * self.depth_scale
        )

        # Valid depth mask
        valid_depth = (
            np.isfinite(depth_meters)
            & (depth_meters > 0)
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
            (u.astype(np.float32) - self.cx)
            * z
            / self.fx
        )

        y = (
            (v.astype(np.float32) - self.cy)
            * z
            / self.fy
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
