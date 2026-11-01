"""How each thing evolves: the registered dynamics entries. Here now: ``body.RigidInertial``, and an apparatus's chain
segments, ``apparatus.Arm`` and ``apparatus.Motor``, with the actuators a description's transmissions name
(``Actuator``, ``register_actuator``)."""

from mime.dynamics.apparatus import Actuator, Arm, Motor, actuator, register_actuator, registered_actuators
from mime.dynamics.body import RigidInertial

__all__ = ["Actuator", "Arm", "Motor", "RigidInertial", "actuator", "register_actuator", "registered_actuators"]
