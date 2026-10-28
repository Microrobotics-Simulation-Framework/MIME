"""Nodes of magnetic laws: the torque on a permanent moment, and the force of a field's gradient on it."""

from mime.nodes.magnetic._force import MagneticForceInputs, MagneticForceNode
from mime.nodes.magnetic._torque import MagneticTorqueInputs, MagneticTorqueNode

__all__ = ["MagneticForceInputs", "MagneticForceNode", "MagneticTorqueInputs", "MagneticTorqueNode"]
