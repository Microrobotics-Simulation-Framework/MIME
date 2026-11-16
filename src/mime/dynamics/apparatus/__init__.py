"""An apparatus's chain segments: the robot arm and the motor, each with its kinematic rung; and the actuators a
description's transmissions name, each a datasheet's rotor inertia with its source."""

from mime.dynamics.apparatus._actuators import Actuator, register_actuator
from mime.dynamics.apparatus._arm import Arm
from mime.dynamics.apparatus._motor import Motor

__all__ = ["Actuator", "Arm", "Motor", "register_actuator"]
