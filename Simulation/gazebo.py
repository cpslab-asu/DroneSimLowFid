import threading
import typing

import gz.msgs10.pose_v_pb2 as _pose_v
import gz.transport13
import numpy as np
import scipy.spatial as _spat

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
    rot = _spat.transform.Rotation.from_quat(q[0], q[1], q[2], q[3])
    return rot.as_euler("XYZ").astype(np.float64)


class GazeboFlightDynamicsModel:
    MODEL_NAME: typing.ClassVar[str] = "iris_gazebo"
    POSE_TOPIC: typing.ClassVar[str] = "/pose"

    def __init__(self):
        self._node = gz.transport13.Node()
        self._lock = threading.Lock()
        self._state = np.zeros((21,), dtype=np.float64)
        self._last_pose_time: float | None = None

        if not self._node.subscribe(_pose_v.Pose_V, GazeboFlightDynamicsModel.POSE_TOPIC, self._update_pose):
            raise RuntimeError("Could not subscribe to pose topic")

    def _update_pose(self, msg: _pose_v.Pose_V):
        pose_time = msg.header.stamp.sec + msg.header.stamp.nsec / 1e9

        # Exit early if this pose message is earlier than an already processed one
        # i.e. _last_pose_time should only be monotonically increasing
        if self._last_pose_time is not None and pose_time <= self._last_pose_time:
            return

        for pose in msg.pose:
            if pose.name == GazeboFlightDynamicsModel.MODEL_NAME:
                pos = np.array(
                    [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
                )

                quat = np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w
                    ],
                    dtype=np.float64
                )

                with self._lock:
                    # Only update velocity if we have a recieved a prior position
                    if self._last_pose_time is not None: 
                        dt = pose_time - self._last_pose_time

                        # Drone displacement
                        s = pos - self._state[0:3]

                        # Velocity vector
                        self._state[7:10] = s / dt

                        # Rotation from quaternion currently stored in state
                        prev_euler = to_euler(self._state[3:7])

                        # Rotation from quaternion in message
                        curr_euler = to_euler(quat)

                        # Angle of rotation as (3,) NDArray
                        turn: NDArray[np.float64] = curr_euler - prev_euler

                        # Angular velocity vector
                        self._state[10:13] = turn / dt

                    # Position vector
                    self._state[0:3] = pos

                    # Orientation quaternion
                    self._state[3:7] = quat

                    # Update time
                    self._last_pose_time = pose_time

    @property
    def state(self) -> np.typing.NDArray[np.float64]:
        with self._lock:
            return self._state.copy()

    @property
    def pos(self) -> np.typing.NDArray[np.float64]:
        return self.state[0:3]

    @property
    def quat(self) -> np.typing.NDArray[np.float64]:
        return self.state[3:7]

    @property
    def vel(self) -> np.typing.NDArray[np.float64]:
        return self.state[7:10]

    @property
    def omega(self) -> np.typing.NDArray[np.float64]:
        return self.state[10:13]

    @property
    def psi(self) -> float:
        return to_euler(self.quat[0])[0]

    @property
    def theta(self) -> float:
        return to_euler(self.quat)[1]

    @property
    def phi(self) -> float:
        return to_euler(self.quat)[2]

