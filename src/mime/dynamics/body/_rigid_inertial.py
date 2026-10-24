from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.gradients import GradientStatus
from mime.nodes.body import SURFACE, RigidInertialNode
from mime.registry import BuildContext, Dynamics, Port, register_dynamics
from mime.things import ThingKind
from mime.units import DIMENSIONLESS, FREQUENCY, LENGTH, VELOCITY, Magnitude
from mime.vocabulary import (
    APPLIED_WRENCH,
    BACKGROUND_LOAD,
    CONFINEMENT,
    POSE,
    RESISTANCE,
    TWIST,
    Capability,
)

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

#: How far a declared orientation's norm may be from one.
_UNIT = 1e-6


def _components(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class RigidInertial(Dynamics):
    """A rigid body with mass and inertia, its state the pose and twist of its centre of mass in the world frame.
    It sums every applied wrench and background load, integrates every delivered resistance implicitly,
    ``(M/dt + R) dV = F - R V``, ``V' = V + dV`` (there is no explicit setting), and projects a confinement after the
    step.

    Version 0.2.0 solves the step for the change in velocity, which 0.1.0 solved as ``(M/dt + R) V' = (M/dt) V + F``:
    the same equation, so float64 results move by rounding, and in float32 a small load is no longer rounded against
    the inertial term (MIME-ANO-113). And it holds its confinement inside the solve where the body starts on its
    surface moving outward, which 0.1.0 left to the projection after the step (an error first order in ``dt`` with a
    coupled resistance; ``RigidInertialNode``). A document naming 0.1.0 is refused at load (``load.version``).
    It reads its body's ``mass`` and ``inertia`` (principal moments about the body axes) and needs an initial
    ``position``, ``orientation`` (a unit quaternion ``(w, x, y, z)``, body to world), ``velocity`` and
    ``angular_velocity``."""

    name = "rigid_inertial"
    version = "0.2.0"
    differentiability = GradientStatus.EXACT
    references = ("PeraireWidnall2009",)
    builds = {"": (RigidInertialNode,)}
    applies_to = frozenset({ThingKind.BODY})
    flags = frozenset({Capability.RIGID, Capability.INERTIAL, Capability.RESISTANCE_IMPLICIT,
                       Capability.PROJECTS_CONFINEMENT})
    publishes = {POSE.name: Port("pose"), TWIST.name: Port("twist")}
    accepts = {APPLIED_WRENCH.name: Port("applied_wrench"), RESISTANCE.name: Port("resistance"),
               BACKGROUND_LOAD.name: Port("background_load"), CONFINEMENT.name: Port("confinement")}
    initial_state = {"position": LENGTH, "orientation": DIMENSIONLESS, "velocity": VELOCITY,
                     "angular_velocity": FREQUENCY}
    values = {"mass": "mass", "inertia": "inertia"}

    @override
    def initial_problems(self, initial: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        for name, size in (("position", 3), ("orientation", 4), ("velocity", 3), ("angular_velocity", 3)):
            if len(_components(initial[name])) != size:
                found.append(f"{name} has {size} components, not {initial[name]!r}")
        orientation = _components(initial["orientation"])
        if len(orientation) == 4 and abs(math.sqrt(sum(c * c for c in orientation)) - 1.0) > _UNIT:
            found.append(f"orientation {orientation} is not a unit quaternion (w, x, y, z)")
        return found

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        initial = {name: _components(value) for name, value in ctx.initial.items()}
        pose = (*initial["position"], *initial["orientation"])
        twist = (*initial["velocity"], *initial["angular_velocity"])
        mass, inertia = ctx.scalar_value("mass"), ctx.vector_value("inertia")
        if not ctx.confinements:
            b.add_node(RigidInertialNode(ctx.node_name(), ctx.timestep, mass=mass, inertia=inertia,
                                         initial_pose=pose, initial_twist=twist))
            return
        (held,) = ctx.confinements
        b.add_node(RigidInertialNode(ctx.node_name(), ctx.timestep, mass=mass, inertia=inertia, initial_pose=pose,
                                     initial_twist=twist, surface_shape=held.shape, extent=held.extent,
                                     margin=held.margin, velocity_rule=held.rule,
                                     **{f"{SURFACE}{name}": value for name, value in held.parameters.items()}))
