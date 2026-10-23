"""Interactions: one physical law among named things, one module per family. Here now: ``Magnetic.Torque``,
``Gravity``, ``Contact`` (by projection) and ``Drag`` (its surrogate rung, with ``Lubrication``)."""

from mime.interactions.contact import Contact
from mime.interactions.gravity import Gravity
from mime.interactions.hydrodynamic import Drag, Lubrication
from mime.interactions.magnetic import Magnetic, MagneticTorque

__all__ = ["Contact", "Drag", "Gravity", "Lubrication", "Magnetic", "MagneticTorque"]
