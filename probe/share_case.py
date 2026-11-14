"""AP2's step share: the exact allocation's share of a full base step, through the real node, in two cases.

``navion``: the done case, ``tests/allocation_cases.allocating_sphere`` (Navion_2, 3 coils, a GRAPH controller every
5 ms, 1 ms late, allocating a PD force through the shipped law within 25 A); with ``allocating=False`` the same
controller makes the same rows and wanted force and writes zero currents from them (no allocation), the baseline.

``cardiomag``: the multipole oracle's sphere in CardioMag (8 coils, the force alone, dt 1 ms), as in the first step's
probe (run 43569142070), its drive replaced by a GRAPH controller every base step that makes ``allocate()``'s 8 rows
at the sphere's point (the field's three over 10 mT and the gradient's five over 1 T/m, given as the law's scales) and
allocates all 8 coils within +-30 A through the shipped law, to a field of 15 mT turning at 2 Hz in the x-y plane and
a gradient of 0.2 T/m times diag(1, -1/2, -1/2); with ``allocating=False`` the same rows and wanted values made and
no allocation."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE.parent / "tests" / "oracles"))

import allocation_cases  # noqa: E402
from allocation_cases import FORCE_SCALE, MASS, MOMENT, OFFSET, AllocatingPD, allocating_sphere  # noqa: E402
from mime.build._declarations import ControllerDeclaration, ThingDeclaration  # noqa: E402
from mime.control import allocation_switch, exact_allocation  # noqa: E402
from mime.data import asset  # noqa: E402
from mime.fields._coil_array import calibration_of  # noqa: E402
from mime.gradients import GradientStatus  # noqa: E402
from mime.nodes.actuation import placed_unit_fields  # noqa: E402
from mime.registry import ControlLaw, Controller, GradientRecord, Signal, register_controller  # noqa: E402
from mime.units import ms  # noqa: E402
from mime.vocabulary import register_namespace  # noqa: E402
from test_calibrated_multipole import sphere_in_the_array  # noqa: E402

jnp: Any = jax.numpy
register_namespace("probe", "0.0.1")
OWNER = "share_case.active_set"
_RECORD = GradientRecord(where="its step: the rows alone", reason="AP2's step-share probe", route="none: a probe")


@register_controller
@dataclass(frozen=True, kw_only=True)
class MakingPD(AllocatingPD):
    """:class:`allocation_cases.AllocatingPD` with its allocation removed: the same rows and wanted force, and zero
    currents made from them."""

    name = "probe.making_pd"
    version = "0.1.0"
    differentiability = GradientStatus.UNVERIFIED
    gradient = _RECORD

    def law(self, ctx: Any) -> ControlLaw:
        held = calibration_of(asset("navion_2"))
        dtype = jnp.zeros(()).dtype
        positions = jnp.asarray(held.positions, dtype=dtype)
        vectors = jnp.asarray(np.asarray(held.coefficients)[..., None] * np.asarray(held.directions), dtype=dtype)
        kp, kd, limit = ctx.scalar("kp"), ctx.scalar("kd"), ctx.scalar("limit")
        period = ctx.timestep
        start = jnp.asarray(allocation_cases._start(), dtype=dtype)
        goal = start + jnp.asarray(OFFSET, dtype=dtype)
        moment = jnp.asarray(MOMENT, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            point = reads["pose"][:3]
            wanted = MASS * (kp * (goal - point) - kd * (point - state["previous"]) / period)
            _, gradients = placed_unit_fields(point, jnp.zeros(3, dtype=dtype),
                                              jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype), positions, vectors)
            rows = jnp.einsum("i,ijk->jk", moment, gradients)
            made = 0.0 * (jnp.sum(rows / FORCE_SCALE, axis=0) + jnp.sum(wanted) + limit)
            return {"previous": point}, {"currents": made}

        return ControlLaw(state={"previous": start}, held={"currents": jnp.zeros((3,), dtype=dtype)}, step=step)


def navion(allocating: bool = True) -> Any:
    """``allocation_cases.allocating_sphere`` itself with the law; without it, the same experiment with
    :class:`MakingPD` (its controller declared once: a second declaration is refused)."""
    if allocating:
        return allocating_sphere("share_navion")
    exp = allocating_sphere("share_navion_none")
    exp._declarations[:] = [d for d in exp._declarations if not isinstance(d, ControllerDeclaration)]
    exp.set_controller(MakingPD(), reads={"pose": "sphere.pose"}, writes={"currents": "coils.currents"},
                       mode="graph", rate=ms(5), latency=ms(1))
    return exp


SCALES = (1e-2, 1.0)  # T, T/m: allocate()'s rows' scales given to the law
OMEGA = 2 * np.pi * 2.0
SHAPE = np.diag([1.0, -0.5, -0.5])


def _gradient5(g: Any) -> Any:
    return jnp.stack([g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2]])


def _cardiomag_law(allocating: bool) -> Any:
    held = calibration_of(asset("cardiomag"))
    positions = np.asarray(held.positions)
    vectors = np.asarray(held.coefficients)[..., None] * np.asarray(held.directions)

    def law(ctx: Any) -> ControlLaw:
        limit, strength, pull = ctx.scalar("limit"), ctx.scalar("strength"), ctx.scalar("pull")
        dt = ctx.timestep
        dtype = jnp.zeros(()).dtype
        pos, vec = jnp.asarray(positions, dtype=dtype), jnp.asarray(vectors, dtype=dtype)
        scales = jnp.asarray([SCALES[0]] * 3 + [SCALES[1]] * 5, dtype=dtype)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            t = state["t"]
            fields, gradients = placed_unit_fields(reads["pose"][:3], jnp.zeros(3, dtype=dtype),
                                                   jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype), pos, vec)
            rows = jnp.concatenate([fields, _gradient5(gradients)])
            wanted = jnp.concatenate([strength * jnp.stack([jnp.cos(OMEGA * t), jnp.sin(OMEGA * t), 0.0 * t]),
                                      _gradient5(pull * jnp.asarray(SHAPE, dtype=dtype))])
            bound = jnp.full((8,), 1.0, dtype=dtype) * limit
            if allocating:
                currents = exact_allocation(rows, wanted, lower=-bound, upper=bound, scales=scales,
                                            owner=OWNER).commands
            else:
                currents = 0.0 * (jnp.sum(rows / scales[:, None], axis=0) + jnp.sum(wanted / scales) + bound)
            return {"t": t + dt}, {"currents": currents}

        return ControlLaw(state={"t": jnp.zeros((), dtype=dtype)}, held={"currents": jnp.zeros((8,), dtype=dtype)},
                          step=step)

    return law


def _cardiomag_controller(allocating: bool) -> type[Controller]:
    suffix = "law" if allocating else "none"

    @register_controller
    @dataclass(frozen=True, kw_only=True)
    class Allocating(Controller):
        name = f"probe.cardiomag_{suffix}"
        version = "0.1.0"
        differentiability = GradientStatus.PIECEWISE_KINK if allocating else GradientStatus.UNVERIFIED
        gradient = GradientRecord(switches=(allocation_switch(OWNER, {"currents": 0}),)) if allocating else _RECORD
        builds = {"": ()}
        reads = {"pose": Signal("m; 1", (7,))}
        writes = {"currents": Signal("A", (8,))}
        limit: float = 30.0
        strength: float = 0.015
        pull: float = 0.2

        def law(self, ctx: Any) -> ControlLaw:
            return _cardiomag_law(allocating)(ctx)

    return Allocating


_CONTROLLERS = {flag: _cardiomag_controller(flag) for flag in (True, False)}


def cardiomag(allocating: bool = True) -> Any:
    exp = sphere_in_the_array("cardiomag", 3, torque=False)
    exp._declarations[:] = [ThingDeclaration(dataclasses.replace(d.thing, commands={}))
                            if isinstance(d, ThingDeclaration) and d.thing.name == "coils" else d
                            for d in exp._declarations]
    exp.set_controller(_CONTROLLERS[allocating](), reads={"pose": "sphere.pose"}, writes={"currents": "coils.currents"},
                       mode="graph")
    return exp
