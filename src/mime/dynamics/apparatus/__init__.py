"""An apparatus's chain segments: the robot arm and the motor, each with its kinematic rung; and the actuators a
description's transmissions name, each a datasheet's rotor inertia with its source."""

from mime.dynamics.apparatus._actuators import STEPPERONLINE_17HS19_2004S1, Actuator, actuator, register_actuator, \
    registered_actuators
from mime.dynamics.apparatus._arm import Arm
from mime.dynamics.apparatus._motor import Motor

__all__ = ["STEPPERONLINE_17HS19_2004S1", "Actuator", "Arm", "Motor", "actuator", "register_actuator",
           "registered_actuators"]
