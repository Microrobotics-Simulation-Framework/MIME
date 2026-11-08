"""How each thing evolves: the registered dynamics entries. Here now: ``body.RigidInertial``, an apparatus's chain
segments, ``apparatus.Arm`` and ``apparatus.Motor``, with the actuators a description's transmissions name
(``Actuator``, ``register_actuator``), and a medium's flow field, ``flow.FVM`` with its time schemes (``Scheme``)
and the prescribed flow ``flow.Womersley``."""

from mime.dynamics.apparatus import Actuator, Arm, Motor, actuator, register_actuator, registered_actuators
from mime.dynamics.body import RigidInertial
from mime.dynamics.flow import FVM, Scheme, Womersley

__all__ = ["FVM", "Actuator", "Arm", "Motor", "RigidInertial", "Scheme", "Womersley", "actuator", "register_actuator",
           "registered_actuators"]
