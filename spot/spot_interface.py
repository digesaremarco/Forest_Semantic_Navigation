"""
Spot SDK wrapper: connection, lease, power, stance and robot-state
queries behind one small interface.

Everything the rest of the project needs from the Boston Dynamics SDK
-- apart from image decoding (acquisition/) and locomotion commands
(navigation/), which receive the clients they need from here -- goes
through this class, so the main loop never talks to the SDK directly
and can later run against a mock exposing the same methods (offline
replay, no robot).

LEASE: taken (LeaseClient.take()) by default, not acquired: when the
operator hands the robot over, the tablet still holds the body lease
and acquire() would fail. A LeaseKeepAlive keeps the lease alive in
the background and returns it on shutdown(). Configurable in
spot_config.yaml ('lease.mode').

E-STOP: deliberately NOT registered by this client. The operator
keeps the E-stop on the tablet; power_on() only checks that the robot
is not E-stopped and fails with an explicit message otherwise.

SHUTDOWN: shutdown() never raises. Every step (sit, power off, lease
return) is attempted independently and any failure is printed, so it
is safe to call from a finally block or an exception handler without
masking the original error. Using the class as a context manager
gives this for free. Sit and power off only happen if THIS client
holds the lease: a read-only session (e.g. acquisition test) never
moves the robot.

POSES: returned as (R, t) numpy pairs -- R (3, 3), t (3,), float64 --
the same convention used throughout fusion/ and mapping/.
"""

import bosdyn.client
from bosdyn.api import robot_state_pb2
from bosdyn.client.frame_helpers import (
    get_odom_tform_body,
    get_vision_tform_body,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandClient,
    blocking_sit,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient

from spot.frame_utils import se3_pose_to_rt, yaw_from_rotation


class SpotInterface:

    def __init__(self, spot_config):
        self.spot_config = spot_config

        self.sdk = None
        self.robot = None

        # Public: acquisition/ and navigation/ are built on top of these.
        self.robot_state_client = None
        self.robot_command_client = None
        self.image_client = None
        self.lease_client = None

        self.lease_keepalive = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    # -----------------------------------------------------------------
    # Connection and lease
    # -----------------------------------------------------------------

    def connect(self):
        """Authenticate, sync time and create the SDK clients."""

        if self.robot is not None:
            return

        config = self.spot_config

        self.sdk = bosdyn.client.create_standard_sdk(config.client_name)
        robot = self.sdk.create_robot(config.robot_ip)

        robot.authenticate(config.username, config.password)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync(timeout_sec=config.time_sync_timeout_s)

        self.robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name
        )
        self.robot_command_client = robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        self.image_client = robot.ensure_client(
            ImageClient.default_service_name
        )
        self.lease_client = robot.ensure_client(
            LeaseClient.default_service_name
        )

        # Set last: is_connected is True only once every client exists.
        self.robot = robot

        print(
            f"[SpotInterface] Connected to {config.robot_ip} as "
            f"'{config.client_name}' (time sync OK)."
        )

    @property
    def is_connected(self):
        return self.robot is not None

    @property
    def has_lease(self):
        return self.lease_keepalive is not None

    def acquire_lease(self):
        """Take (or acquire) the body lease and keep it alive."""

        self.require_connected()

        if self.lease_keepalive is not None:
            return

        if self.spot_config.lease_mode == "take":
            self.lease_client.take()
        else:
            self.lease_client.acquire()

        self.lease_keepalive = LeaseKeepAlive(
            self.lease_client, must_acquire=False, return_at_exit=True
        )

        print(f"[SpotInterface] Body lease obtained ({self.spot_config.lease_mode}).")

    def release_lease(self):
        """Stop the keep-alive and return the lease."""

        if self.lease_keepalive is None:
            return

        keepalive = self.lease_keepalive
        self.lease_keepalive = None

        keepalive.shutdown()

        print("[SpotInterface] Body lease returned.")

    # -----------------------------------------------------------------
    # Power and stance
    # -----------------------------------------------------------------

    def check_estop(self):
        """Raise if the robot is E-stopped."""

        self.require_connected()

        if self.robot.is_estopped():
            raise RuntimeError(
                "Robot is E-stopped. The E-stop is held by the operator "
                "on the tablet: release it there before powering on."
            )

    def is_powered_on(self):
        self.require_connected()

        return self.robot.is_powered_on()

    def power_on(self):
        self.require_lease()
        self.check_estop()

        if self.robot.is_powered_on():
            return

        print("[SpotInterface] Powering on...")

        self.robot.power_on(timeout_sec=self.spot_config.power_on_timeout_s)

        if not self.robot.is_powered_on():
            raise RuntimeError("Robot did not report powered on after power_on().")

        print("[SpotInterface] Powered on.")

    def stand(self):
        self.require_lease()
        self.require_powered()

        blocking_stand(
            self.robot_command_client,
            timeout_sec=self.spot_config.stand_timeout_s
        )

        print("[SpotInterface] Standing.")

    def sit(self):
        self.require_lease()
        self.require_powered()

        blocking_sit(
            self.robot_command_client,
            timeout_sec=self.spot_config.sit_timeout_s
        )

        print("[SpotInterface] Sitting.")

    def power_off(self):
        """Safe power off: the robot sits down first if needed."""

        self.require_lease()

        if not self.robot.is_powered_on():
            return

        self.robot.power_off(
            cut_immediately=False,
            timeout_sec=self.spot_config.power_off_timeout_s
        )

        print("[SpotInterface] Powered off.")

    def shutdown(self):
        """
        Best-effort, never-raising shutdown: sit, optionally power off,
        return the lease. Sit and power off only if this client holds
        the lease.
        """

        if self.robot is None:
            return

        if self.lease_keepalive is not None:
            self.safe_call("sit", self.sit_if_powered)

            if self.spot_config.power_off_on_shutdown:
                self.safe_call("power off", self.power_off)

            self.safe_call("lease return", self.release_lease)

        print("[SpotInterface] Shutdown complete.")

    def sit_if_powered(self):
        if self.robot.is_powered_on():
            self.sit()

    # -----------------------------------------------------------------
    # Robot state
    # -----------------------------------------------------------------

    def get_robot_state(self):
        self.require_connected()

        return self.robot_state_client.get_robot_state()

    def get_vision_tform_body(self, robot_state=None):
        """(R, t) of the body in the vision frame."""

        state = robot_state if robot_state is not None else self.get_robot_state()

        return se3_pose_to_rt(
            get_vision_tform_body(state.kinematic_state.transforms_snapshot)
        )

    def get_odom_tform_body(self, robot_state=None):
        """(R, t) of the body in the odom frame."""

        state = robot_state if robot_state is not None else self.get_robot_state()

        return se3_pose_to_rt(
            get_odom_tform_body(state.kinematic_state.transforms_snapshot)
        )

    def get_body_pose_2d(self, robot_state=None):
        """(x, y, yaw) of the body in the vision frame [m, m, rad]."""

        R, t = self.get_vision_tform_body(robot_state)

        return float(t[0]), float(t[1]), yaw_from_rotation(R)

    def get_body_position_xy(self, robot_state=None):
        """(x, y) of the body in the vision frame [m]."""

        x, y, _ = self.get_body_pose_2d(robot_state)

        return x, y

    def get_battery_percentage(self, robot_state=None):
        """Locomotion battery charge [%], or None if not reported."""

        state = robot_state if robot_state is not None else self.get_robot_state()
        power_state = state.power_state

        if not power_state.HasField("locomotion_charge_percentage"):
            return None

        return float(power_state.locomotion_charge_percentage.value)

    def get_active_faults(self, robot_state=None):
        """
        Active faults, as {"system": [names], "behavior": [names]}.
        Behavior faults (e.g. a fall) block locomotion until cleared.
        """

        state = robot_state if robot_state is not None else self.get_robot_state()

        system = [fault.name for fault in state.system_fault_state.faults]

        behavior = [
            self.behavior_fault_name(fault)
            for fault in state.behavior_fault_state.faults
        ]

        return {"system": system, "behavior": behavior}

    @staticmethod
    def behavior_fault_name(fault):
        try:
            cause = robot_state_pb2.BehaviorFault.Cause.Name(fault.cause)
        except Exception:
            cause = str(fault.cause)

        return f"{cause} (id {fault.behavior_fault_id})"

    # -----------------------------------------------------------------
    # Guards
    # -----------------------------------------------------------------

    def require_connected(self):
        if self.robot is None:
            raise RuntimeError("SpotInterface is not connected: call connect() first.")

    def require_lease(self):
        self.require_connected()

        if self.lease_keepalive is None:
            raise RuntimeError(
                "This client does not hold the body lease: call "
                "acquire_lease() first."
            )

    def require_powered(self):
        if not self.robot.is_powered_on():
            raise RuntimeError("Robot is not powered on.")

    @staticmethod
    def safe_call(description, function):
        try:
            function()
        except Exception as error:
            print(f"[SpotInterface] WARNING: {description} failed during shutdown: {error}")