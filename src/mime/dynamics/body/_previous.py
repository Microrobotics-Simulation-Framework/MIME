"""The versions of the body dynamics kept registered after a change of law, so that a folder saved with one still loads
and reproduces, and so that the frozen register's scripts keep the law they were run with: ``rigid_inertial`` 0.3.0,
the inertia as three principal moments about the body's axes, and 0.2.0, the gyroscopic term explicit as well
(MIME-ANO-129). Each is its successor in every respect but one. A script that writes ``RigidInertial()`` gets the
current version; a load of a document that names an old one gets it from here. None is exported for a new experiment:
0.3.0 has no place for products of inertia, and 0.2.0's explicit term gains kinetic energy for a body of unequal
moments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import override

import numpy as np

from mime.dynamics.body._rigid_inertial import LIMITS, RigidInertial, tensor_of
from mime.nodes.body import RigidInertialMidpointNode, RigidInertialNode
from mime.registry import Dynamics, Problem, register_dynamics
from mime.units import Magnitude

EXPLICIT_LIMITS = (
    "the gyroscopic term is taken explicitly, so a torque-free body of three different moments gains kinetic energy at "
    "every step, first order in the step (2.6 times in a second at 1 ms and 200 rad/s), and a rig with nothing to damp "
    "it goes non-finite; a body of equal moments is unaffected, bit for bit (MIME-ANO-129)",
)
#: The products of inertia 0.3.0 took as none, as a share of the largest moment: below it, its part resolver used the
#: diagonal without a word; above it, 0.3.0 refused the tensor or, in exploratory, dropped the products with a stamp.
PRODUCTS_TAKEN_AS_NONE = 1e-12


def products_share(tensor: np.ndarray) -> float:
    """The largest product of inertia of ``tensor`` as a share of its largest moment (the product itself where every
    moment is zero)."""
    largest = float(np.max(np.abs(np.diag(tensor))))
    across = float(np.max(np.abs(tensor - np.diag(np.diag(tensor)))))
    return across / largest if largest > 0.0 else across


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class RigidInertialMoments(RigidInertial):
    """``rigid_inertial`` 0.3.0: the inertia as the three principal moments about the body's axes, the gyroscopic term
    linearly implicit at the midpoint (``RigidInertialMidpointNode``). It takes the diagonal of the body's inertia and
    refuses a tensor whose products 0.3.0 did not take as none (:data:`PRODUCTS_TAKEN_AS_NONE`): this version has no
    place for them, and exploratory's waiver that dropped them went with 0.4.0."""

    version = "0.3.0"
    limits = LIMITS
    builds = {"": (RigidInertialMidpointNode,)}

    @override
    def node_inertia(self, inertia: tuple[float, ...]) -> tuple[float, ...]:
        """The three moments the node takes: the diagonal of the body's inertia."""
        return tuple(float(c) for c in np.diag(tensor_of(inertia)))

    @override
    def value_problems(self, values: Mapping[str, Magnitude]) -> list[Problem]:
        found = super().value_problems(values)
        if found or "inertia" not in values:
            return found
        held = values["inertia"]
        share = products_share(tensor_of(held if isinstance(held, tuple) else (held,)))
        if share <= PRODUCTS_TAKEN_AS_NONE:
            return []
        return [Problem("value.not-positive",
                        f"rigid_inertial {self.version} takes the principal moments about the body's axes, and the "
                        f"inertia has products of inertia, the largest {100.0 * share:.3g}% of the largest moment",
                        "build the body with the current version, which carries the full tensor, "
                        "Body(..., dynamics=RigidInertial()); or, to keep this version, declare the principal moments "
                        "about the body's axes: exp.declare('<body>.inertia', kg_m2((Ix, Iy, Iz)), reason=...)")]


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class RigidInertialExplicit(RigidInertialMoments):
    """``rigid_inertial`` 0.2.0: the step of 0.3.0 with the gyroscopic term explicit, ``-omega x I omega`` in the load
    and nothing of its Jacobian in the matrix. It reproduces, bit for bit, what 0.2.0 computed: for equal moments it
    is 0.3.0's step, and for unequal moments it gains energy (MIME-ANO-129)."""

    version = "0.2.0"
    limits = EXPLICIT_LIMITS
    builds = {"": (RigidInertialNode,)}


#: Each kept version by its entry's name and version, with the version that superseded it.
KEPT: Mapping[tuple[str, str], tuple[type[Dynamics], str]] = MappingProxyType({
    ("rigid_inertial", "0.2.0"): (RigidInertialExplicit, RigidInertialMoments.version),
    ("rigid_inertial", "0.3.0"): (RigidInertialMoments, RigidInertial.version),
})

__all__ = ["KEPT", "PRODUCTS_TAKEN_AS_NONE", "RigidInertialExplicit", "RigidInertialMoments", "products_share"]
