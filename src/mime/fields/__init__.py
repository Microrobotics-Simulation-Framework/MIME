"""Field providers: what makes a field available at points. Here now: ``UniformRotating``; ``PointDipole``, a magnet
part's field; ``CalibratedMultipole``, an electromagnet array's, on the contract every ``CoilArray`` keeps, with
``Actuation`` and ``allocate``; and ``KEPT``, the earlier versions kept registered."""

from collections.abc import Mapping
from types import MappingProxyType

from mime.fields._coil_array import CALIBRATIONS, Actuation, Allocation, CalibratedMultipole, CoilArray, allocate
from mime.fields._point_dipole import PointDipole
from mime.fields._previous import ClockRotating
from mime.fields._uniform_rotating import UniformRotating
from mime.registry import FieldProvider

#: The versions of the field providers kept registered after a later version changed what they compute, by name and
#: version, each with the version that superseded it: a document naming one loads it, and a script that must reproduce
#: one builds it from here. None is exported for a new experiment.
KEPT: Mapping[tuple[str, str], tuple[type[FieldProvider], str]] = MappingProxyType({
    ("uniform_rotating", "0.1.0"): (ClockRotating, UniformRotating.version),
})

__all__ = ["CALIBRATIONS", "KEPT", "Actuation", "Allocation", "CalibratedMultipole", "CoilArray", "PointDipole",
           "UniformRotating", "allocate"]
