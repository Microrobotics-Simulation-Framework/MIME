"""How each thing evolves: the registered dynamics entries. Here now: ``body.RigidInertial``, and an apparatus's chain
segments, ``apparatus.RobotArm`` and ``apparatus.Motor``."""

from mime.dynamics.apparatus import Motor, RobotArm
from mime.dynamics.body import RigidInertial

__all__ = ["Motor", "RigidInertial", "RobotArm"]
