"""
One-off calibration script: read the intrinsics of Spot's five body
cameras (frontleft, frontright, left, right, back) and of their depth
sources, plus the static body -> camera extrinsics, and save everything
to config/camera_intrinsics.yaml.

For every camera three sources are queried:
    - <camera>_fisheye_image           visual (grayscale/RGB) image
    - <camera>_depth_in_visual_frame   depth registered into the visual camera
    - <camera>_depth                   raw depth, own frame and resolution

The intrinsics are the ones Spot's image service reports for each
source. NOTE: Spot's "fisheye" images are already rectified by the
robot, so for most robot software versions the reported model is a plain
pinhole (no distortion). If a distortion model is reported, its
coefficients are saved too.

The intrinsics refer to the image AS RETURNED by the robot, NOT rotated.
Boston Dynamics' examples rotate some images for display (frontleft
about -78 deg, frontright about -102 deg, right 180 deg): if you rotate
images before using them, the intrinsics must be transformed accordingly.

Run this once, on the Jetson (CORE I/O) or any machine connected to the
robot. The robot does not need to stand, move or hold a lease. Re-run
only if a camera is remounted or the robot is recalibrated.

Requires the Boston Dynamics SDK, ideally the same version as the robot
software:
    pip install bosdyn-client==<robot version> bosdyn-api==<robot version>
"""

import os
from pathlib import Path

import yaml

import bosdyn.client
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient


# Connection. From the CORE I/O (payload port) the robot is usually at
# 192.168.50.3; over Spot's WiFi access point it is 192.168.80.3.
# Credentials are read from the standard bosdyn environment variables
# if set, so they don't have to be written in the code.
ROBOT_HOSTNAME = os.environ.get("SPOT_HOSTNAME", "192.168.50.3")
ROBOT_USERNAME = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
ROBOT_PASSWORD = os.environ.get("BOSDYN_CLIENT_PASSWORD", "pwd")

CAMERAS = ["frontleft", "frontright", "left", "right", "back"]

SOURCE_TYPES = {
    "visual": "_fisheye_image",
    "depth_in_visual": "_depth_in_visual_frame",
    "depth": "_depth",
}

SAVE_EXTRINSICS = True

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "config" / "camera_intrinsics.yaml"


def connect_to_robot(hostname, username, password):
    """
    Authenticate with Spot and return a ready-to-use robot object.
    """

    sdk = bosdyn.client.create_standard_sdk("IntrinsicsCalibrationClient")
    robot = sdk.create_robot(hostname)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    return robot


def get_camera_model_name(source):
    """
    Name of the camera model reported for this source ("pinhole",
    "pinhole_brown_conrady", "kannala_brandt"), or None.

    Older SDK versions only have the plain "pinhole" field, newer ones
    group the models in the "camera_models" oneof.
    """

    try:
        return source.WhichOneof("camera_models")
    except ValueError:
        return "pinhole" if source.HasField("pinhole") else None


def extract_intrinsics(source):
    """
    Read the camera model of an ImageSource.

    return: dict with the pinhole parameters, the camera matrix K and,
            if present, the distortion coefficients
    """

    model_name = get_camera_model_name(source)

    if model_name is None:
        return {"camera_model": "none"}

    model_intrinsics = getattr(source, model_name).intrinsics

    # the distortion models wrap the pinhole parameters in a sub-message
    if model_name == "pinhole":
        pinhole = model_intrinsics
    else:
        pinhole = model_intrinsics.pinhole_intrinsics

    fx = float(pinhole.focal_length.x)
    fy = float(pinhole.focal_length.y)
    cx = float(pinhole.principal_point.x)
    cy = float(pinhole.principal_point.y)
    skew = float(pinhole.skew.x)

    data = {
        "camera_model": model_name,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "skew": skew,
        "K": [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
    }

    # distortion coefficients: every scalar field of the model except the
    # pinhole part (read from the descriptor, so zero values are kept too)
    if model_name != "pinhole":
        data["distortion"] = {
            field.name: float(getattr(model_intrinsics, field.name))
            for field in model_intrinsics.DESCRIPTOR.fields
            if field.message_type is None
        }

    return data


def extract_extrinsics(snapshot, frame_name):
    """
    Static body -> camera transform from the image transforms_snapshot.

    return: dict with position and quaternion (scalar-last, like the
            rest of the project)
    """

    try:
        body_tform_camera = get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name)
    except Exception as error:
        available_frames = list(snapshot.child_to_parent_edge_map.keys())
        raise RuntimeError(f"could not compute {BODY_FRAME_NAME}_tform_{frame_name}: {error}\n"
                           f"frames available in this snapshot: {available_frames}") from error

    if body_tform_camera is None:
        available_frames = list(snapshot.child_to_parent_edge_map.keys())
        raise RuntimeError(f"frame '{frame_name}' not connected to '{BODY_FRAME_NAME}' in the "
                           f"snapshot. Frames available: {available_frames}")

    return {
        "position": {
            "x": float(body_tform_camera.position.x),
            "y": float(body_tform_camera.position.y),
            "z": float(body_tform_camera.position.z),
        },
        "rotation": {
            "qx": float(body_tform_camera.rotation.x),
            "qy": float(body_tform_camera.rotation.y),
            "qz": float(body_tform_camera.rotation.z),
            "qw": float(body_tform_camera.rotation.w),
        },
    }


def build_source_names(cameras, source_types):
    """
    return: dict {(camera, source_type): source_name}
    """

    return {
        (camera, source_type): f"{camera}{suffix}"
        for camera in cameras
        for source_type, suffix in source_types.items()
    }


def query_calibration(robot, cameras, source_types, save_extrinsics):
    """
    Capture one image from every available source and read its
    intrinsics (and extrinsics) from the response.

    return: nested dict {camera: {source_type: calibration}}
    """

    image_client = robot.ensure_client(ImageClient.default_service_name)

    available = {source.name for source in image_client.list_image_sources()}
    wanted = build_source_names(cameras, source_types)

    # requesting a non-existing source makes the whole request fail,
    # so only ask for the ones the robot actually has
    missing = sorted(name for name in wanted.values() if name not in available)
    if missing:
        print(f"sources not available on this robot, skipped: {missing}")
        print(f"available sources: {sorted(available)}")

    requested = [name for name in wanted.values() if name in available]
    if not requested:
        raise RuntimeError("none of the requested image sources is available")

    responses = image_client.get_image_from_sources(requested)
    responses_by_name = {response.source.name: response for response in responses}

    calibration = {}

    for (camera, source_type), source_name in wanted.items():
        response = responses_by_name.get(source_name)
        if response is None:
            continue

        source = response.source
        shot = response.shot

        entry = {
            "source_name": source_name,
            "frame_name": shot.frame_name_image_sensor,
            "width": int(source.cols),
            "height": int(source.rows),
        }

        if shot.image.cols != source.cols or shot.image.rows != source.rows:
            print(f"WARNING {source_name}: source declares {source.cols}x{source.rows} but the "
                  f"captured image is {shot.image.cols}x{shot.image.rows}")

        entry.update(extract_intrinsics(source))

        if "depth" in source_type:
            # raw uint16 value / depth_scale = depth in meters
            entry["depth_scale"] = float(source.depth_scale)

        if save_extrinsics:
            entry["body_tform_camera"] = extract_extrinsics(shot.transforms_snapshot,
                                                            shot.frame_name_image_sensor)

        calibration.setdefault(camera, {})[source_type] = entry

    return calibration


def check_depth_in_visual(calibration):
    """
    The depth registered in the visual frame should have exactly the same
    frame, resolution and intrinsics as the visual camera.
    """

    keys = ["frame_name", "width", "height", "fx", "fy", "cx", "cy"]

    print("\ncheck: depth_in_visual_frame vs visual camera")

    for camera, sources in calibration.items():
        if "visual" not in sources or "depth_in_visual" not in sources:
            continue

        visual = sources["visual"]
        depth = sources["depth_in_visual"]

        differences = [
            key for key in keys
            if (abs(visual[key] - depth[key]) > 1e-6 if isinstance(visual.get(key), float)
                else visual.get(key) != depth.get(key))
        ]

        if differences:
            print(f"  {camera}: DIFFERENT in {differences}, use the depth intrinsics for depth")
        else:
            print(f"  {camera}: identical, the visual intrinsics are valid for the depth too")


def print_summary(calibration):
    print("\nsummary")
    for camera, sources in calibration.items():
        for source_type, entry in sources.items():
            if entry.get("camera_model", "none") == "none":
                print(f"  {entry['source_name']:<34} {entry['width']}x{entry['height']}  no camera model")
                continue

            line = (f"  {entry['source_name']:<34} {entry['width']}x{entry['height']}  "
                    f"{entry['camera_model']:<8} fx={entry['fx']:.2f} fy={entry['fy']:.2f} "
                    f"cx={entry['cx']:.2f} cy={entry['cy']:.2f}")
            if "depth_scale" in entry:
                line += f"  depth_scale={entry['depth_scale']:g}"
            print(line)


def save_calibration(calibration, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(calibration, file, default_flow_style=None, sort_keys=False)

    print(f"\ncalibration saved to {output_path}")


if __name__ == "__main__":

    robot = connect_to_robot(ROBOT_HOSTNAME, ROBOT_USERNAME, ROBOT_PASSWORD)

    calibration = query_calibration(robot, CAMERAS, SOURCE_TYPES, SAVE_EXTRINSICS)

    print_summary(calibration)
    check_depth_in_visual(calibration)
    save_calibration(calibration, OUTPUT_PATH)