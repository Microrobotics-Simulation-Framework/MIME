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
