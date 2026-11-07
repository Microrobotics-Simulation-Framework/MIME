"""The angle from a field to a body's moment in the field's plane of rotation, and the body's spin about the field's
axis, kept in the observing node's own state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import (
    MimeNode,
    NodeFields,
    Symbol,
    SymbolKind,
    as_run_array,
    jnp,
    node_fields,
    node_input,
    quaternion,
    switch,
    typed_view,
)

EQUATIONS = (r"\phi = \operatorname{atan2}\left( \mathbf{a} \cdot (\mathbf{m}_{\perp} \times \mathbf{B}_{\perp}), "
             r"\mathbf{m}_{\perp} \cdot \mathbf{B}_{\perp} \right) + 2 \pi k \, \nu h, \quad "
             r"\nu = \frac{\boldsymbol{\Omega} \cdot \mathbf{a}}{2 \pi}, \quad "
             r"\bar{\phi} = \frac{\sum_{j : t_{j} > t_{a}} \phi_{j} h}{T}, \quad "
             r"\bar{\nu} = \frac{\sum_{j : t_{j} > t_{a}} \nu_{j} h}{T}, \quad T = \sum_{j : t_{j} > t_{a}} h")

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.body._phase_lag.PhaseLagNode.advance"


class PhaseLagState(TypedDict, closed=True):
    lag: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    spin: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    mean_lag: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    mean_spin: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    lagged: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    turned: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    observed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


class PhaseLagInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    twist: NotRequired[ReadOnly[Float[Array, "6"]]]
    flux_density: NotRequired[ReadOnly[Float[Array, "3"]]]


def _across(v: Array, axis: Array) -> Array:
    return v - jnp.dot(v, axis) * axis


class PhaseLagNode(MimeNode):
    """``lag``, the angle about ``about`` from the body's moment to the field (radians), each projected onto the plane
    of rotation (positive where the field leads), net of the declared lag: the moment as the laws that turn the body
    read it, ``behind`` steps before the step's end, which is the moment after the step turned back by the body's spin
    about the axis over those steps (``2 pi`` times ``behind`` times the spin times the step). ``spin``, the body's
    angular velocity about the axis in turns per second (Hz), so that it reads against a field's frequency as it is.
    ``mean_lag`` and ``mean_spin``, their means over the steps that end after ``after`` (``lagged`` and ``turned`` their
    integrals over them, the latter the body's turns about the axis, and ``observed`` their length). Its time is its
    step count times its timestep; the time observed is zero or at least one timestep, so a mean is taken from half a
    timestep on. Where the moment or the field has no part in the plane the angle reads zero."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-121",
        description="The angle from a field to a body's moment in the plane the field turns in, net of the lag the "
                    "build declares for the laws that turn the body, and the body's spin about the field's axis, each "
                    "step and their means over the steps that end after a start time, kept in the node's own state.",
        governing_equations=EQUATIONS,
        assumptions=("The field and the moment are each taken in the plane perpendicular to the axis: their angle is "
                     "the angle about the axis between their projections",
                     "The moment the laws read, behind steps before the end of the step, is the moment after the step "
                     "turned back about the axis by the body's spin times that many steps: exact where the body turns "
                     "about the axis at the spin it ends the step with, as the rigid body turns its pose by its new "
                     "angular velocity",
                     "The time of a step is its count times the timestep, and a step counts whole once its end lies "
                     "after the start time"),
        limitations=("The window's edge is a switch: the means move in jumps of a step as the start time moves, and "
                     "each is zero until some time has been observed",
                     "Where the body turns about another axis as well (it tilts), the turn back about the axis is "
                     "the part of its rotation about that axis, not the whole of it",
                     "The angle is undefined where the moment or the field has no part in the plane (a field of zero "
                     "amplitude): it reads zero there"),
    )
    symbols = (
        Symbol(r"\phi", SymbolKind.STATE, "lag", "1", _ADVANCE),
        Symbol(r"\nu", SymbolKind.STATE, "spin", "Hz", _ADVANCE),
        Symbol(r"\bar{\phi}", SymbolKind.STATE, "mean_lag", "1", _ADVANCE),
        Symbol(r"\bar{\nu}", SymbolKind.STATE, "mean_spin", "Hz", _ADVANCE),
        Symbol("T", SymbolKind.STATE, "observed", "s", _ADVANCE),
        Symbol(r"\mathbf{B}", SymbolKind.INPUT, "flux_density", "T", _ADVANCE),
        Symbol(r"\boldsymbol{\Omega}", SymbolKind.INPUT, "twist", "m/s; rad/s", _ADVANCE),
        Symbol(r"\mathbf{m}", SymbolKind.PARAMETER, "moment", "A m^2", _ADVANCE),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "about", "1", _ADVANCE),
        Symbol("k", SymbolKind.PARAMETER, "behind", "1", _ADVANCE),
        Symbol("t_{a}", SymbolKind.PARAMETER, "after", "s", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, moment: Sequence[float], about: Sequence[float],
                 behind: float = 0.0, after: float = 0.0, declared: str = "") -> None:
        """``behind``, the steps the laws read the pose behind this node's read, is declared by the build from its
        lag accounting (:meth:`netted`) and ``declared`` says how; a node made on its own is net of no lag."""
        super().__init__(name, timestep, moment=tuple(float(c) for c in moment), about=tuple(float(c) for c in about),
                         behind=float(behind), after=float(after), declared=declared)

    def netted(self, behind: int, declared: str) -> PhaseLagNode:
        """This node made net of ``behind`` steps, the lag the build declares, with ``declared`` its account of it."""
        own = self.constants()
        return PhaseLagNode(self.name, self.delta_t, moment=own["moment"], about=own["about"], behind=float(behind),
                            after=own["after"], declared=declared)

    def built_lines(self, start: Mapping[str, Any] | None, closest_approach: float | None) -> list[str]:
        """The lag the build removed, and from what it read it."""
        del start, closest_approach
        own = self.constants()
        return [f"the lag is {own.get('declared') or 'not declared'}"]

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "moment": ParamSpec(units="A m^2"), "about": ParamSpec(units="1"),
                "behind": ParamSpec(units="1"), "after": ParamSpec(units="s")}

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(0.0)
        return node_fields(PhaseLagState(lag=zero, spin=zero, mean_lag=zero, mean_spin=zero, lagged=zero, turned=zero,
                                         observed=zero, steps=jnp.asarray(0, dtype=jnp.int32)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,)), "twist": node_input("m/s; rad/s", (6,)),
                "flux_density": node_input("T", (3,))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(PhaseLagState, state), typed_view(PhaseLagInputs, boundary_inputs)
        dtype = held["lag"].dtype
        pose = given.get("pose", jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=dtype))
        twist = given.get("twist", jnp.zeros(6, dtype=dtype))
        field = given.get("flux_density", jnp.zeros(3, dtype=dtype))
        axis = jnp.asarray(p["about"], dtype=dtype)
        moment = _across(quaternion.rotate(pose[3:], jnp.asarray(p["moment"], dtype=dtype)), axis)
        across = _across(field, axis)
        spin = jnp.dot(twist[3:], axis) / (2 * jnp.pi)
        sine, cosine = jnp.dot(axis, jnp.cross(moment, across)), jnp.dot(moment, across)
        some = sine * sine + cosine * cosine > 0
        angle = jnp.arctan2(switch.domain_guard(some, sine, 0.0), switch.domain_guard(some, cosine, 1.0))
        ahead = angle + 2 * jnp.pi * as_run_array(p["behind"]) * spin * dt
        lag = jnp.arctan2(jnp.sin(ahead), jnp.cos(ahead))
        steps = held["steps"] + 1
        elapsed = steps.astype(dtype) * dt
        counted = elapsed > p["after"]

        def after_by() -> Array:
            return elapsed - p["after"]

        def rounded() -> Array:
            return switch.rounding(elapsed)

        lagged = held["lagged"] + switch.where(counted, lag * dt, 0.0, owner="phase_lag.window", resolution=rounded,
                                               indicator=after_by)
        turned = held["turned"] + switch.where(counted, spin * dt, 0.0, owner="phase_lag.window", resolution=rounded,
                                               indicator=after_by)
        observed = held["observed"] + switch.where(counted, dt, 0.0, owner="phase_lag.window", resolution=rounded,
                                                   indicator=after_by)
        seen = observed > dt / 2
        length = switch.domain_guard(seen, observed, 1.0)
        mean_lag = switch.where(seen, lagged / length, 0.0, owner="phase_lag.mean", resolution=rounded,
                                indicator=lambda: observed - dt / 2)
        mean_spin = switch.where(seen, turned / length, 0.0, owner="phase_lag.mean", resolution=rounded,
                                 indicator=lambda: observed - dt / 2)
        return node_fields(PhaseLagState(lag=lag, spin=spin, mean_lag=mean_lag, mean_spin=mean_spin, lagged=lagged,
                                         turned=turned, observed=observed, steps=steps))
