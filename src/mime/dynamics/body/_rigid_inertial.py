from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np

from mime.gradients import GradientStatus
from mime.nodes.body import HULL, SURFACE, RigidInertialNode, RigidInertialTensorNode
from mime.registry import BuildContext, Dynamics, Port, Problem, register_dynamics
from mime.things import ThingKind
from mime.units import ANGULAR_VELOCITY, DIMENSIONLESS, LENGTH, VELOCITY, Magnitude
from mime.vocabulary import (
    APPLIED_WRENCH,
    BACKGROUND_LOAD,
    CONFINEMENT,
    MASS_MATRIX,
    POSE,
    RESISTANCE,
    RIGID_BODY,
    STEP_ROW,
    TWIST,
    Capability,
)

if TYPE_CHECKING:
    from mime.build import DynamicsBuilder, ResolvedRegistration

#: How far a declared orientation's norm may be from one.
_UNIT = 1e-6
#: The rotation rate times the step above which the gyroscopic term is no longer resolved to the accuracy this entry
#: states: below it the torque-free energy stays within 2e-3 of its start (measured on the node alone, 206 rad/s,
#: moments (1.0, 1.2, 0.8) e-10, ``tests/oracles/test_gyroscopic_energy.py``: 1.4e-3 at 0.5), and the error grows as
#: its square (5.5e-3 at 1, 0.10 at 4, 1.2 at 10). A declaration from that measurement on one body, not a fit and not a
#: bound for every body; nothing refuses a run past it.
RATE_TIMES_STEP = 0.5
LIMITS = (
    f"the gyroscopic term is resolved only while the rotation rate times the step, |omega| dt, is below about "
    f"{RATE_TIMES_STEP:g}, where a torque-free body of unequal moments keeps its kinetic energy to 2e-3; the error "
    f"grows as its square, and a rate times the step of order one runs finite and wrong, with nothing to refuse it "
    f"(MIME-ANO-129)",
    "the direction of the angular momentum in the world is first order in the step, bounded: about 2 degrees at a 1 ms "
    "step and 200 rad/s, while its magnitude and the kinetic energy are second order (MIME-ANO-129)",
    "the step's matrix, M/dt + R - K/2, is not symmetric for a body of unequal moments: a solve that needs it "
    "symmetric positive definite does not hold for it (MIME-ANO-129)",
)


#: How far a declared inertia tensor may be from symmetric, as a share of its largest entry: a tensor written out to
#: twelve digits, entry by entry. A declaration, not a tolerance on the body.
SYMMETRY = 1e-12


def _components(value: Magnitude) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


def tensor_of(inertia: Sequence[float]) -> np.ndarray:
    """The 3 x 3 of an ``inertia`` value, in float64: the diagonal of three principal moments, or nine numbers
    row-major."""
    held = np.asarray(inertia, dtype=np.float64)
    return np.diag(held) if held.size == 3 else held.reshape(3, 3)


def inertia_problems(components: tuple[float, ...]) -> list[Problem]:
    """What is wrong with an ``inertia`` value as the step takes it: three moments or nine numbers, finite, symmetric
    to :data:`SYMMETRY` of its largest entry, and positive definite (the step divides by it)."""
    if len(components) not in (3, 9) or not all(math.isfinite(c) for c in components):
        return [Problem("value.not-positive", f"the inertia {components} is neither three finite moments nor a 3 x 3 "
                                              f"tensor's nine finite numbers",
                        "declare the principal moments about the body's axes, "
                        "exp.declare('<body>.inertia', kg_m2((Ix, Iy, Iz)), reason=...), or the tensor about its "
                        "centre of mass in its frame, nine numbers row-major, "
                        "exp.declare('<body>.inertia', kg_m2((Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz)), "
                        "reason=...)")]
    tensor = tensor_of(components)
    asymmetry = float(np.max(np.abs(tensor - tensor.T)))
    if asymmetry > SYMMETRY * float(np.max(np.abs(tensor))):
        return [Problem("value.not-positive", f"the inertia tensor {components} is not symmetric: J[i][j] and J[j][i] "
                                              f"differ by up to {asymmetry:.3g} kg m^2",
                        "declare the tensor with each product of inertia the same in its two places, J[i][j] equal "
                        "to J[j][i]: exp.declare('<body>.inertia', kg_m2((Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, "
                        "Izz)), reason=...)")]
    least = float(np.min(np.linalg.eigvalsh(0.5 * (tensor + tensor.T))))
    if least > 0.0:
        return []
    what = "moments of inertia are not all above zero" if len(components) == 3 else \
        f"inertia tensor is not positive definite, its least principal moment {least:.3g} kg m^2"
    return [Problem("value.not-positive",
                    f"the {what}, {components}: the step divides by the inertia, and a body without a positive moment "
                    f"about every axis has no rotation to integrate",
                    "declare the principal moments about the body's axes, each above zero, "
                    "exp.declare('<body>.inertia', kg_m2((Ix, Iy, Iz)), reason=...), or a positive definite tensor, "
                    "nine numbers row-major, exp.declare('<body>.inertia', kg_m2((Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, "
                    "Iyz, Izz)), reason=...)")]


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class RigidInertial(Dynamics):
    """A rigid body with mass and inertia, its state the pose and twist of its centre of mass in the world frame.
    Version 0.4.0 takes the full inertia tensor about the centre of mass in the body's frame, carried as it is
    (``RigidInertialTensorNode``): ``inertia`` is three principal moments about the body's axes or the symmetric 3 x 3
    tensor's nine numbers, row-major, and a body whose parts give products of inertia runs with them, where 0.3.0
    refused them or, in exploratory, dropped them. A body with no products moves as 0.3.0's did, bit for bit per
    compiled program. 0.3.0 and 0.2.0 stay registered (``mime.dynamics.body.KEPT``).
    It sums every applied wrench and background load, integrates every delivered resistance implicitly,
    ``(M/dt + R) dV = F - R V``, ``V' = V + dV`` (there is no explicit setting), takes the gyroscopic term linearly
    implicit at the midpoint (it keeps a torque-free body's kinetic energy and the magnitude of its angular momentum
    to second order; MIME-ANO-129), and holds a confinement: its centre by its extent, projected after the step, or
    its hull by impulses through the step's system and a projection after it (``Contact`` 0.5.0, and 0.4.0 kept;
    ``RigidInertialNode``), which needs nothing of a body's dynamics but this capability, so both versions hold both.

    Version 0.3.0 takes the gyroscopic term linearly implicit at the midpoint, where 0.2.0 took it explicit: the
    explicit term gained kinetic energy at every step for a body of three different moments, 2.6 times in a second at
    1 ms, and a rig with nothing to damp it went non-finite (MIME-ANO-129). Results of a rotating body of unequal
    moments move; those of an isotropic body do not, bit for bit. 0.2.0 stays registered
    (``mime.dynamics.body.KEPT``), so that a folder saved with it loads and reproduces, and so that the frozen
    register's scripts keep the law they were run with. A moment of inertia that is not above zero is refused at
    build (``value.not-positive``).

    Version 0.2.0 solves the step for the change in velocity, which 0.1.0 solved as ``(M/dt + R) V' = (M/dt) V + F``:
    the same equation, so float64 results move by rounding, and in float32 a small load is no longer rounded against
    the inertial term (MIME-ANO-113). And it holds its confinement inside the solve where the body starts on its
    surface moving outward, which 0.1.0 left to the projection after the step (an error first order in ``dt`` with a
    coupled resistance; ``RigidInertialNode``). A document naming 0.1.0 is refused at load (``load.version``).
    It reads its body's ``mass`` and ``inertia`` (principal moments about the body axes, or the tensor's nine numbers)
    and needs an initial
    ``position``, ``orientation`` (a unit quaternion ``(w, x, y, z)``, body to world), ``velocity`` and
    ``angular_velocity``."""

    name = "rigid_inertial"
    version = "0.4.0"
    limits = LIMITS
    differentiability = GradientStatus.EXACT
    references = ("PeraireWidnall2009", "Sola2017")
    builds = {"": (RigidInertialTensorNode,)}
    applies_to = frozenset({ThingKind.BODY})
    flags = frozenset({Capability.RIGID, Capability.INERTIAL, Capability.RESISTANCE_IMPLICIT,
                       Capability.PROJECTS_CONFINEMENT, Capability.HOLDS_HULL})
    publishes = {POSE.name: Port("pose"), TWIST.name: Port("twist"), MASS_MATRIX.name: Port("mass_matrix"),
                 STEP_ROW.name: Port("step_row")}
    port_kinds = (RIGID_BODY.name,)
    accepts = {APPLIED_WRENCH.name: Port("applied_wrench"), RESISTANCE.name: Port("resistance"),
               BACKGROUND_LOAD.name: Port("background_load"), CONFINEMENT.name: Port("confinement")}
    initial_state = {"position": LENGTH, "orientation": DIMENSIONLESS, "velocity": VELOCITY,
                     "angular_velocity": ANGULAR_VELOCITY}
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

    def node_inertia(self, inertia: tuple[float, ...]) -> tuple[float, ...]:
        """The ``inertia`` the node takes from the body's resolved value: the tensor's nine numbers, row-major, the
        diagonal's where three moments were declared."""
        return tuple(float(c) for c in tensor_of(inertia).reshape(9))

    @override
    def value_problems(self, values: Mapping[str, Magnitude]) -> list[Problem]:
        return inertia_problems(_components(values["inertia"])) if "inertia" in values else []

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        node = cast(type[RigidInertialNode], type(self).builds[""][0])
        initial = {name: _components(value) for name, value in ctx.initial.items()}
        pose = (*initial["position"], *initial["orientation"])
        twist = (*initial["velocity"], *initial["angular_velocity"])
        mass, inertia = ctx.scalar_value("mass"), self.node_inertia(ctx.vector_value("inertia"))
        published: dict[str, Any] = {**({"publishes_mass_matrix": True} if ctx.mass_read else {}),
                                     **({"publishes_step_row": True} if ctx.row_read else {})}
        if not ctx.confinements:
            b.add_node(node(ctx.node_name(), ctx.timestep, mass=mass, inertia=inertia, initial_pose=pose,
                            initial_twist=twist, **published))
            return
        (held,) = ctx.confinements
        hull: dict[str, Any] = {}
        if held.hull is not None:
            kind = held.hull.kind
            settings = {setting.name: held.hull.settings.get(setting.name, getattr(kind, setting.name))
                        for setting in dataclasses.fields(kind)}
            hull = {"hull_kind": kind.name, "hull_kind_version": kind.version, "hull_patches": tuple(kind.patches),
                    "hull_counts": held.counts, "hull_centre": held.hull.centre,
                    "hull_rotation": tuple(c for row in held.hull.rotation for c in row),
                    **{f"{HULL}{name}": value for name, value in settings.items()},
                    **({"contact_loops": True} if held.loops else {})}
        surface: dict[str, Any] = {f"{SURFACE}{name}": value for name, value in held.parameters.items()}
        b.add_node(node(ctx.node_name(), ctx.timestep, mass=mass, inertia=inertia, initial_pose=pose,
                        initial_twist=twist, surface_shape=held.shape, extent=held.extent, margin=held.margin,
                        velocity_rule=held.rule, **hull, **surface, **published))
