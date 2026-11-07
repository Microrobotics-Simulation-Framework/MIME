"""A uniform magnetic field rotating about a fixed axis at a commanded frequency and amplitude."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_input, typed_view
from mime.nodes.actuation._turns import advance_phase

EQUATIONS = (r"\mathbf{B} = \frac{B_{0}}{\left\| \mathbf{d} - (\mathbf{d} \cdot \mathbf{a}) \mathbf{a} \right\|} "
             r"\left( \cos(2 \pi f n h) \left( \mathbf{d} - (\mathbf{d} \cdot \mathbf{a}) \mathbf{a} \right) + "
             r"\sin(2 \pi f n h) \, \mathbf{a} \times \mathbf{d} \right), \quad G = 0")

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.actuation._uniform_field.UniformRotatingNode.advance"
_FIELD = "mime.nodes.actuation._uniform_field.UniformRotatingNode.field_at"


class UniformRotatingInputs(TypedDict, closed=True):
    frequency: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    amplitude: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape


class UniformRotatingState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``steps``, and ``flux_density_<slot>`` and ``gradient_<slot>`` for each reader slot."""

    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


def _unit(v: Array) -> Array:
    return v / jnp.sqrt(jnp.sum(v * v))


class UniformRotatingNode(MimeNode):
    """``B(t) = amplitude (cos(2 pi f t) e1 + sin(2 pi f t) axis x e1)``: the same at every point, turning right-handed
    about ``rotation_axis`` at the commanded frequency ``f`` (``e1``, ``initial_direction`` projected onto the plane of
    rotation and normalised, is where it points at ``t = 0``). Its time is its step count times its timestep, never a
    sum of timesteps, which drifted in float32 in 0.2. Each reader slot reads the field and its gradient, zero."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-116",
        description="A uniform magnetic field turning right-handed about a fixed axis at a commanded frequency and "
                    "amplitude: the same at every point, with a zero gradient.",
        governing_equations=EQUATIONS,
        assumptions=("The field is uniform: it is the same at every point of every reader slot, and its gradient is "
                     "zero",
                     "The rotation axis in the equation is a unit vector: the node normalises `rotation_axis` before "
                     "it reads it",
                     "Time is the step count times the timestep, so the field's phase is exact however long the run "
                     "and is not a sum of timesteps",
                     "The frequency and amplitude commanded at a step are the ones that step's field is made with: "
                     "the phase is the present frequency times the time since the start, not the integral of a "
                     "frequency that changes"),
    )
    symbols = (
        Symbol(r"\mathbf{B}", SymbolKind.STATE, "flux_density_{slot}", "T", _ADVANCE),
        Symbol("G", SymbolKind.STATE, "gradient_{slot}", "T/m", _ADVANCE),
        Symbol("B_{0}", SymbolKind.INPUT, "amplitude", "T", _ADVANCE),
        Symbol("f", SymbolKind.INPUT, "frequency", "Hz", _ADVANCE),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "rotation_axis", "1", _FIELD),
        Symbol(r"\mathbf{d}", SymbolKind.PARAMETER, "initial_direction", "1", _FIELD),
        Symbol("n", SymbolKind.STATE, "steps", "1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, slots: Sequence[str], rotation_axis: Sequence[float],
                 initial_direction: Sequence[float]) -> None:
        axis = tuple(float(c) for c in rotation_axis)
        start = tuple(float(c) for c in initial_direction)
        along = sum(a * s for a, s in zip(axis, start, strict=True)) / math.sqrt(sum(a * a for a in axis))
        if math.sqrt(sum(s * s for s in start)) * (1 - 1e-6) <= abs(along):
            raise ValueError(f"initial_direction {start} lies along the rotation axis {axis}: the field's direction at "
                             f"t = 0 must have a part in the plane of rotation")
        super().__init__(name, timestep, slots=tuple(slots), rotation_axis=axis, initial_direction=start)

    def _slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants()["slots"])

    def _fields(self, steps: Array, field: Array) -> NodeFields:
        out: NodeFields = {"steps": steps}
        for slot in self._slots():
            out[f"flux_density_{slot}"] = field
            out[f"gradient_{slot}"] = jnp.zeros((3, 3), dtype=field.dtype)
        return out

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "rotation_axis": ParamSpec(units="1"),
                "initial_direction": ParamSpec(units="1")}

    @override
    def initial_state(self) -> NodeFields:
        return self._fields(jnp.asarray(0, dtype=jnp.int32), as_run_array(jnp.zeros(3)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"frequency": node_input("Hz", ()), "amplitude": node_input("T", ())}

    def field_at(self, p: Mapping[str, Any], frequency: Array, amplitude: Array, time: Array) -> Array:
        axis = _unit(jnp.asarray(p["rotation_axis"], dtype=time.dtype))
        start = jnp.asarray(p["initial_direction"], dtype=time.dtype)
        first = _unit(start - jnp.dot(start, axis) * axis)
        phase = 2 * jnp.pi * frequency * time
        return amplitude * (jnp.cos(phase) * first + jnp.sin(phase) * jnp.cross(axis, first))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(UniformRotatingState, state), typed_view(UniformRotatingInputs,
                                                                                boundary_inputs)
        steps = held["steps"] + 1
        zero = as_run_array(0.0)
        time = steps.astype(zero.dtype) * dt
        return self._fields(steps, self.field_at(p, given.get("frequency", zero), given.get("amplitude", zero), time))


INTEGRATING_EQUATIONS = (
    r"\mathbf{B} = \frac{B_{0}}{\left\| \mathbf{d} - (\mathbf{d} \cdot \mathbf{a}) \mathbf{a} \right\|} "
    r"\left( \cos(2 \pi \Theta_{n}) \left( \mathbf{d} - (\mathbf{d} \cdot \mathbf{a}) \mathbf{a} \right) + "
    r"\sin(2 \pi \Theta_{n}) \, \mathbf{a} \times \mathbf{d} \right), "
    r"\quad \Theta_{n} + \theta_{n} = \operatorname{frac}\left( \Theta_{n-1} + \theta_{n-1} + f_{n} h \right), \quad G = 0")

#: Where the integrating node's symbol rows' function is.
_INTEGRATE = "mime.nodes.actuation._uniform_field.IntegratingRotatingNode.advance"


class IntegratingRotatingState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``steps``, ``phase`` and ``residual`` (the phase in turns as a fraction, ``phase`` in ``[0, 1)`` and ``residual``
    the part of it below ``phase``'s last place), and ``flux_density_<slot>`` and ``gradient_<slot>`` for each reader
    slot."""

    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape
    phase: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    residual: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape


class IntegratingRotatingNode(MimeNode):
    """The uniform rotating field whose phase is the integral of its commanded frequency: the angle turned over step
    ``n`` is ``2 pi f_n h``, the frequency commanded for the step times its length, whatever writes the frequency
    (a drive, a controller), so a frequency that changes changes the rate the field turns at and never jumps its angle.

    The phase is held in turns as a fraction ``Theta_n`` (a pair of floating numbers, :mod:`~mime.nodes.actuation._turns`)
    to which each step adds ``f_n h`` as an exact product, by an error-free sum, so that what rounds each step is far
    below the phase's own last place and the phase's error does not grow with the number of turns the run has made: in
    float32 the field's angle is within about ``1e-6`` rad of the integral of the frequencies written at 100 s and at
    1000 s alike, where the product of a frequency and a time carries a rounding of the phase's own size (0.025 rad at
    1000 s and 40 Hz). The field is read through the cosine and sine of ``2 pi`` times the fraction taken about zero
    (``Theta_n`` less its nearest whole turn), the same direction as at a whole turn more. The step ``h`` is the node's
    own timestep, the exact Python float, which MADDENING keeps equal to the step the graph calls the node at."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-119",
        description="A uniform magnetic field turning right-handed about a fixed axis by the integral of a commanded "
                    "frequency, at a commanded amplitude: the same at every point, with a zero gradient.",
        governing_equations=INTEGRATING_EQUATIONS,
        assumptions=("The field is uniform: it is the same at every point of every reader slot, and its gradient is "
                     "zero",
                     "The rotation axis in the equation is a unit vector: the node normalises `rotation_axis` before "
                     "it reads it",
                     "The frequency commanded at a step holds over that step: the field turns by the frequency times "
                     "the step's length, so its angle is the integral of the frequency held over each step",
                     "The phase is a sum of the turns of each step, each held exactly and added by an error-free "
                     "sum, so its rounding per step is far below the phase's own last place and does not carry the "
                     "phase's size (the bound and its measurement are in tests/oracles/test_uniform_rotating_phase.py)",
                     "The amplitude commanded at a step is the one that step's field is made with"),
    )
    symbols = (
        Symbol(r"\mathbf{B}", SymbolKind.STATE, "flux_density_{slot}", "T", _INTEGRATE),
        Symbol("G", SymbolKind.STATE, "gradient_{slot}", "T/m", _INTEGRATE),
        Symbol("B_{0}", SymbolKind.INPUT, "amplitude", "T", _INTEGRATE),
        Symbol("f_{n}", SymbolKind.INPUT, "frequency", "Hz", _INTEGRATE),
        Symbol(r"\Theta_{n}", SymbolKind.STATE, "phase", "1", _INTEGRATE),
        Symbol(r"\theta_{n}", SymbolKind.STATE, "residual", "1", _INTEGRATE),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "rotation_axis", "1", _INTEGRATE),
        Symbol(r"\mathbf{d}", SymbolKind.PARAMETER, "initial_direction", "1", _INTEGRATE),
        Symbol("n", SymbolKind.STATE, "steps", "1", _INTEGRATE),
    )

    def __init__(self, name: str, timestep: float, *, slots: Sequence[str], rotation_axis: Sequence[float],
                 initial_direction: Sequence[float]) -> None:
        axis = tuple(float(c) for c in rotation_axis)
        start = tuple(float(c) for c in initial_direction)
        along = sum(a * s for a, s in zip(axis, start, strict=True)) / math.sqrt(sum(a * a for a in axis))
        if math.sqrt(sum(s * s for s in start)) * (1 - 1e-6) <= abs(along):
            raise ValueError(f"initial_direction {start} lies along the rotation axis {axis}: the field's direction at "
                             f"t = 0 must have a part in the plane of rotation")
        super().__init__(name, timestep, slots=tuple(slots), rotation_axis=axis, initial_direction=start)

    def _slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants()["slots"])

    def _fields(self, steps: Array, phase: Array, residual: Array, field: Array) -> NodeFields:
        out: NodeFields = {"steps": steps, "phase": phase, "residual": residual}
        for slot in self._slots():
            out[f"flux_density_{slot}"] = field
            out[f"gradient_{slot}"] = jnp.zeros((3, 3), dtype=field.dtype)
        return out

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "rotation_axis": ParamSpec(units="1"),
                "initial_direction": ParamSpec(units="1")}

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(0.0)
        return self._fields(jnp.asarray(0, dtype=jnp.int32), zero, zero, as_run_array(jnp.zeros(3)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"frequency": node_input("Hz", ()), "amplitude": node_input("T", ())}

    def field_at(self, p: Mapping[str, Any], phase: Array, amplitude: Array) -> Array:
        """The field at ``phase`` radians."""
        axis = _unit(jnp.asarray(p["rotation_axis"], dtype=phase.dtype))
        start = jnp.asarray(p["initial_direction"], dtype=phase.dtype)
        first = _unit(start - jnp.dot(start, axis) * axis)
        return amplitude * (jnp.cos(phase) * first + jnp.sin(phase) * jnp.cross(axis, first))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held = typed_view(IntegratingRotatingState, state)
        given = typed_view(UniformRotatingInputs, boundary_inputs)
        zero = as_run_array(0.0)
        frequency = given.get("frequency", zero).astype(zero.dtype)
        # The step's length is the node's own timestep, a Python float: the argument ``dt`` arrives traced, in the run's
        # precision, which would put float32's representation of the timestep into the phase of every turn.
        phase, residual = advance_phase(held["phase"], held["residual"], frequency, self.delta_t)
        about_zero = phase - jnp.round(phase)
        angle = 2 * jnp.pi * (about_zero + residual)
        return self._fields(held["steps"] + 1, phase, residual, self.field_at(p, angle, given.get("amplitude", zero)))
