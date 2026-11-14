"""A test-only controller that calls ``mime.control.exact_allocation`` inside the compiled step, and the experiment it
runs in. Registered under the ``oracle`` namespace in this process only (the shipped registry holds no controller that
calls the law), so that the switch scan and the solve walk reach the law's sites through one configuration.

The experiment is camera spike R2's STATE arm without the camera: a 1 mm sphere of 4e-6 kg with a fixed moment of
2.8e-4 A m^2 at the centre of Navion_2's declared region, the force alone (no torque, so that the moment holds its
direction), no gravity, no medium, dt = 1 ms. The controller reads the sphere's true pose every 5 ms, one step late,
and pulls it towards a point 3 mm off its start by a PD force, allocated to the three coils within 25 A at the sphere's
point through the law, its rows the force per unit current on the moment (N/A) over a scale of 1 mN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from mime import Apparatus, Body, Experiment, State
from mime.control import allocation_switch, exact_allocation
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.gradients import GradientStatus
from mime.interactions import Magnetic
from mime.nodes.actuation import placed_unit_fields
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import A_m2, kg, kg_m2, m, m_per_s, m_per_s2, ms, rad_per_s
from mime.vocabulary import register_namespace

jnp: Any = jax.numpy
register_namespace("oracle", "0.0.1")

#: The owner the controller declares and passes to the law (a test module's owner carries its module's name).
OWNER = "allocation_cases.active_set"
MOMENT = (0.0, 0.0, 2.8e-4)
MASS = 4e-6
LIMIT = 25.0
FORCE_SCALE = 1e-3
OFFSET = (3e-3, -1.5e-3, 0.0)


def _start() -> tuple[float, float, float]:
    region = calibration_of(asset("navion_2")).region
    centre = region.mean(axis=1)
    return (float(centre[0]), float(centre[1]), float(centre[2]))


@register_controller
@dataclass(frozen=True, kw_only=True)
class AllocatingPD(Controller):
    """A PD force towards a point, allocated to Navion_2's coils within ``limit`` (A) by the exact allocation."""

    name = "oracle.allocating_pd"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(allocation_switch(OWNER, {"currents": 0}),))
    builds = {"": ()}
    reads = {"pose": Signal("m; 1", (7,))}
    writes = {"currents": Signal("A", (3,))}
    kp: float = 2500.0
    kd: float = 80.0
    limit: float = LIMIT

    def law(self, ctx: Any) -> ControlLaw:
        held = calibration_of(asset("navion_2"))
        dtype = jnp.zeros(()).dtype
        positions = jnp.asarray(held.positions, dtype=dtype)
        vectors = jnp.asarray(np.asarray(held.coefficients)[..., None] * np.asarray(held.directions), dtype=dtype)
        kp, kd, limit = ctx.scalar("kp"), ctx.scalar("kd"), ctx.scalar("limit")
        period = ctx.timestep
        start = jnp.asarray(_start(), dtype=dtype)
        goal = start + jnp.asarray(OFFSET, dtype=dtype)
        moment = jnp.asarray(MOMENT, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            point = reads["pose"][:3]
            wanted = MASS * (kp * (goal - point) - kd * (point - state["previous"]) / period)
            _, gradients = placed_unit_fields(point, jnp.zeros(3, dtype=dtype),
                                              jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype), positions, vectors)
            rows = jnp.einsum("i,ijk->jk", moment, gradients)
            found = exact_allocation(rows, wanted, lower=-limit, upper=limit,
                                     scales=jnp.full((3,), FORCE_SCALE, dtype=dtype), owner=OWNER)
            return {"previous": point}, {"currents": found.commands}

        return ControlLaw(state={"previous": start}, held={"currents": jnp.zeros((3,), dtype=dtype)}, step=step)


def allocating_sphere(name: str = "allocating_sphere", *, rate=ms(5), latency=ms(1)) -> Experiment:
    """The sphere in Navion_2's field, its coils' currents written by :class:`AllocatingPD` inside the compiled step."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    exp.add_apparatus(Apparatus("coils", fields={"magnetic": CalibratedMultipole(calibration=asset("navion_2"))}))
    exp.add_body(Body("sphere", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=m(1e-3)))},
                      initial=State(position=m(_start()), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("sphere.mass", kg(MASS)), ("sphere.inertia", kg_m2((1.6e-12,) * 3)),
                        ("sphere.magnetic_moment", A_m2(MOMENT)), ("sphere.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre sphere with a magnet at its centre")
    exp.attach(Magnetic.Force(), body="sphere", field="coils.magnetic")
    exp.set_controller(AllocatingPD(), reads={"pose": "sphere.pose"}, writes={"currents": "coils.currents"},
                       mode="graph", rate=rate, latency=latency)
    return exp


@dataclass(frozen=True)
class GuardPoint:
    """An allocation at which some of the law's domain guards have their conditions false: its rows, wanted values
    and symmetric limit."""

    rows: tuple[tuple[float, ...], ...]
    wanted: tuple[float, ...]
    limit: float


#: The points the law's guards are reached false at, each guard by one or both: a command with no effect (the column
#: scale's guard, and every free subset holding it, rank deficient); and rows of rank one against three commands (every
#: subset of two free commands or more rank deficient, in columns and, past two, in rows).
GUARD_POINTS = {
    "a command with no effect": GuardPoint(((1.0, 0.0, 2.0), (0.5, 0.0, -1.0)), (10.0, 3.0), 1.0),
    "rows of rank one": GuardPoint(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0)), (1.0, 2.0), 1.0),
}


def _at(point: GuardPoint) -> tuple[Any, Any]:
    dtype = jnp.zeros(()).dtype
    return jnp.asarray(point.rows, dtype=dtype), jnp.asarray((*point.wanted, point.limit), dtype=dtype)


def _law_at(rows: Any, p: Any) -> dict[str, Any]:
    found = exact_allocation(rows, p[:-1], lower=-p[-1], upper=p[-1], scales=jnp.ones(rows.shape[0], rows.dtype),
                             owner=OWNER)
    return {"commands": found.commands, "objective": found.objective}


def guard_point_value(point: GuardPoint) -> dict[str, Any]:
    """The law's outputs at ``point``, evaluated afresh (no compiled cache), in the run's dtype."""
    rows, p = _at(point)
    found = exact_allocation(rows, p[:-1], lower=-p[-1], upper=p[-1], scales=jnp.ones(rows.shape[0], rows.dtype),
                             owner=OWNER)
    return {"commands": found.commands, "active": found.active, "objective": found.objective}


def guard_point_derivative(point: GuardPoint) -> Any:
    """Their derivative in the wanted values and the limit, traced afresh."""
    rows, p = _at(point)
    return jax.jacfwd(lambda q: _law_at(rows, q))(p)


__all__ = ["GUARD_POINTS", "LIMIT", "OWNER", "AllocatingPD", "GuardPoint", "allocating_sphere", "guard_point_derivative",
           "guard_point_value"]
