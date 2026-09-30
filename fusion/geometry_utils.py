"""
Small shared geometric helpers used across the pose/frame pairing
and camera extrinsics modules -- kept here instead of duplicated in
each, since both FramePosePairer and CameraExtrinsicsConfig need the
same quaternion convention.
"""

import math

import numpy as np


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert a scalar-last quaternion (ROS convention) to a 3x3
    rotation matrix.
    """

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

    if norm == 0:
        raise ValueError(
            "Cannot convert a zero-norm quaternion to a rotation matrix."
        )

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [
            1 - 2 * (qy ** 2 + qz ** 2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw)
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx ** 2 + qz ** 2),
            2 * (qy * qz - qx * qw)
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx ** 2 + qy ** 2)
        ]
    ], dtype=np.float32)

    return R


def compose_poses(R_a_b, t_a_b, R_b_c, t_b_c):
    """
    Compose two SE(3) poses: given a_tform_b and b_tform_c, return
    a_tform_c (the pose of frame c expressed in frame a).

        R_a_c = R_a_b @ R_b_c
        t_a_c = R_a_b @ t_b_c + t_a_b
    """

    R_a_c = R_a_b @ R_b_c
    t_a_c = R_a_b @ t_b_c + t_a_b

    return R_a_c, t_a_c
