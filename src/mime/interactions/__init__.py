"""Interactions: one physical law among named things, one module per family. Here now: ``Magnetic.Torque`` and
``Magnetic.Force``, ``Gravity``, ``Contact`` (by projection of the body's hull) and ``Drag`` (its surrogate rung, with
``Lubrication``, and its MRS rung); and ``KEPT``, the earlier versions kept registered."""

from collections.abc import Mapping
from types import MappingProxyType

from mime.interactions._previous import CentreContact
from mime.interactions.contact import Contact
from mime.interactions.gravity import Gravity
from mime.interactions.hydrodynamic import Drag, Lubrication
from mime.interactions.magnetic import Magnetic, MagneticForce, MagneticTorque
from mime.registry import Interaction

#: The versions of the interactions kept registered after a later version changed what they compute, by name and
#: version, each with the version that superseded it: a document naming one loads it, and a script that must reproduce
#: one (7j's frozen register's, script A's and A0's comparisons) builds it from here. None is exported for a new
#: experiment.
KEPT: Mapping[tuple[str, str], tuple[type[Interaction], str]] = MappingProxyType({
    ("contact", "0.1.0"): (CentreContact, Contact.version),
})

__all__ = ["KEPT", "Contact", "Drag", "Gravity", "Lubrication", "Magnetic", "MagneticForce", "MagneticTorque"]
