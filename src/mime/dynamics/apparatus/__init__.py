"""An apparatus's chain segments: the robot arm and the motor, each with its kinematic rung."""

from mime.dynamics.apparatus._arm import RobotArm
from mime.dynamics.apparatus._motor import Motor

__all__ = ["Motor", "RobotArm"]
