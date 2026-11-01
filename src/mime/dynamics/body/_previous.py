"""The versions of the body dynamics kept registered after a change of law, so that a folder saved with one still loads
and reproduces, and so that the frozen register's scripts keep the law they were run with: ``rigid_inertial`` 0.2.0,
the gyroscopic term explicit (MIME-ANO-129). Each is its successor in every respect but one. A script that writes
``RigidInertial()`` gets the current version; a load of a document that names an old one gets it from here. None is
exported for a new experiment: the explicit term gains kinetic energy for a body of unequal moments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mime.dynamics.body._rigid_inertial import RigidInertial
from mime.nodes.body import RigidInertialNode
from mime.registry import Dynamics, register_dynamics

EXPLICIT_LIMITS = (
    "the gyroscopic term is taken explicitly, so a torque-free body of three different moments gains kinetic energy at "
    "every step, first order in the step (2.6 times in a second at 1 ms and 200 rad/s), and a rig with nothing to damp "
    "it goes non-finite; a body of equal moments is unaffected, bit for bit (MIME-ANO-129)",
)


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class RigidInertialExplicit(RigidInertial):
    """``rigid_inertial`` 0.2.0: the step of 0.3.0 with the gyroscopic term explicit, ``-omega x I omega`` in the load
    and nothing of its Jacobian in the matrix. It reproduces, bit for bit, what 0.2.0 computed: for equal moments it
    is 0.3.0's step, and for unequal moments it gains energy (MIME-ANO-129)."""

    version = "0.2.0"
    limits = EXPLICIT_LIMITS
    builds = {"": (RigidInertialNode,)}


#: Each kept version by its entry's name and version, with the version that superseded it.
KEPT: Mapping[tuple[str, str], tuple[type[Dynamics], str]] = MappingProxyType({
    ("rigid_inertial", "0.2.0"): (RigidInertialExplicit, RigidInertial.version),
})

__all__ = ["KEPT", "RigidInertialExplicit"]
