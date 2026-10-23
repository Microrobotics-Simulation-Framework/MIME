"""A uniform magnetic field rotating about a fixed axis at a commanded frequency and amplitude."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_input, typed_view


class UniformRotatingFieldInputs(TypedDict, closed=True):
    frequency: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    amplitude: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape


class UniformRotatingFieldState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``steps``, and ``flux_density_<slot>`` and ``gradient_<slot>`` for each reader slot."""

    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


def _unit(v: Array) -> Array:
    return v / jnp.sqrt(jnp.sum(v * v))


class UniformRotatingFieldNode(MimeNode):
    """``B(t) = amplitude (cos(2 pi f t) e1 + sin(2 pi f t) axis x e1)``: the same at every point, turning right-handed
    about ``rotation_axis`` at the commanded frequency ``f`` (``e1``, ``initial_direction`` projected onto the plane of
    rotation and normalised, is where it points at ``t = 0``). Its time is its step count times its timestep, never a
    sum of timesteps, which drifted in float32 in 0.2. Each reader slot reads the field and its gradient, zero."""

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
        held, given = typed_view(UniformRotatingFieldState, state), typed_view(UniformRotatingFieldInputs,
                                                                                boundary_inputs)
        steps = held["steps"] + 1
        zero = as_run_array(0.0)
        time = steps.astype(zero.dtype) * dt
        return self._fields(steps, self.field_at(p, given.get("frequency", zero), given.get("amplitude", zero), time))
