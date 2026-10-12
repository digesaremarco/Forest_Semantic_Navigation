"""
Small geometry helpers shared by spot/ and acquisition/: conversion
of SDK poses (bosdyn.client.math_helpers.SE3Pose) into the (R, t)
numpy convention used throughout fusion/ and mapping/, and back to a
quaternion for logging.

Kept free of any bosdyn import -- se3_pose_to_rt() only calls the
pose's own to_matrix() -- so offline code (ObservationLogger, a
future mock robot for replay) can use it on a machine without the
SDK installed.

Conventions:
    R : numpy.ndarray, shape (3, 3), float64
    t : numpy.ndarray, shape (3,), float64
    quaternion : (qx, qy, qz, qw), same order as Spot's CSV/YAML
"""

import numpy as np


def se3_pose_to_rt(pose):
    """(R, t) from an SDK SE3Pose (anything with to_matrix() -> 4x4)."""

    matrix = np.asarray(pose.to_matrix(), dtype=np.float64)

    return matrix[:3, :3].copy(), matrix[:3, 3].copy()


def rt_to_matrix(R, t):
    """4x4 homogeneous transform from (R, t)."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(R, dtype=np.float64)
    matrix[:3, 3] = np.asarray(t, dtype=np.float64)

    return matrix


def rotation_angle_deg(R_a, R_b):
    """Angle [deg] of the relative rotation between R_a and R_b."""

    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)

    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0

    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def yaw_from_rotation(R):
    """Heading (rotation about world z) [rad] of a body rotation R."""

    R = np.asarray(R, dtype=np.float64)

    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotation_matrix_to_quaternion(R):
    """
    Unit quaternion (qx, qy, qz, qw) of a rotation matrix (Shepperd's
    method, numerically stable for every rotation). Returned with
    qw >= 0; q and -q are the same rotation.
    """

    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)

    if quaternion[3] < 0.0:
        quaternion = -quaternion

    return tuple(float(value) for value in quaternion)