"""Interactions: one physical law among named things, one module per family. Here now: ``Magnetic.Torque`` and
``Magnetic.Force``, ``Gravity``, ``Contact`` (by projection of the body's hull) and ``Drag`` (its surrogate rung, with
``Lubrication``, and its MRS rung, with ``ForceSampling`` and ``NearField``, its settings' groups, and
``roughness_cutoff``, where its lubrication hands a gap to contact), the flow's ``BodyForce`` and the laws at its
openings, ``Inflow`` and ``Windkessel``; and ``KEPT``, the earlier versions kept registered."""

from collections.abc import Mapping
from types import MappingProxyType

from mime.interactions._previous import CentreContact, CentreMagneticForce, CentreMagneticTorque, UnrolledContact
from mime.interactions.contact import Contact
from mime.interactions.flow import BodyForce, Inflow, Windkessel
from mime.interactions.gravity import Gravity
from mime.interactions.hydrodynamic import Drag, ForceSampling, Lubrication, NearField, PointForceDrag, UncoupledDrag
from mime.interactions.magnetic import Magnetic, MagneticForce, MagneticTorque
from mime.nodes.hydro.mrs._handover import roughness_cutoff
from mime.registry import Interaction

#: The versions of the interactions kept registered after a later version changed what they compute, by name and
#: version, each with the version that superseded it: a document naming one loads it, and a script that must reproduce
#: one (7j's frozen register's, script A's and A0's comparisons) builds it from here. None is exported for a new
#: experiment.
KEPT: Mapping[tuple[str, str], tuple[type[Interaction], str]] = MappingProxyType({
    ("contact", "0.1.0"): (CentreContact, Contact.version),
    ("contact", "0.4.0"): (UnrolledContact, Contact.version),
    ("drag", "0.3.0"): (PointForceDrag, Drag.version),
    ("drag", "0.4.0"): (UncoupledDrag, Drag.version),
    ("magnetic.torque", "0.1.0"): (CentreMagneticTorque, MagneticTorque.version),
    ("magnetic.force", "0.1.0"): (CentreMagneticForce, MagneticForce.version),
})

__all__ = ["KEPT", "BodyForce", "Contact", "Drag", "ForceSampling", "Gravity", "Inflow", "Lubrication", "Magnetic",
           "MagneticForce", "MagneticTorque", "NearField", "Windkessel", "roughness_cutoff"]
