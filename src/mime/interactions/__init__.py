"""Interactions: one physical law among named things, one module per family. Here now: ``Magnetic.Torque``,
``Gravity`` and ``Contact`` (by projection)."""

from mime.interactions.contact import Contact
from mime.interactions.gravity import Gravity
from mime.interactions.magnetic import Magnetic, MagneticTorque

__all__ = ["Contact", "Gravity", "Magnetic", "MagneticTorque"]
