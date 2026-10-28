"""A body's swim speed along a direction, kept in the observing node's own state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_fields, node_input, switch, typed_view


class SwimSpeedState(TypedDict, closed=True):
    speed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    mean_speed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    distance: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    observed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


class SwimSpeedInputs(TypedDict, closed=True):
    twist: NotRequired[ReadOnly[Float[Array, "6"]]]


class SwimSpeedNode(MimeNode):
    """``speed = v . along`` each step; ``mean_speed``, the distance travelled along ``along`` over the time observed
    (every step whose end lies after ``after``), so it is the mean of ``speed`` over those steps. Its time is its
    step count times its timestep. The time observed is zero or at least one timestep, so the mean is taken from
    half a timestep on, which keeps its derivative finite however the state is drawn."""

    def __init__(self, name: str, timestep: float, *, along: Sequence[float], after: float = 0.0) -> None:
        super().__init__(name, timestep, along=tuple(along), after=after)

    @override
    def initial_state(self) -> NodeFields:
        zero = as_run_array(0.0)
        return node_fields(SwimSpeedState(speed=zero, mean_speed=zero, distance=zero, observed=zero,
                                          steps=jnp.asarray(0, dtype=jnp.int32)))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"twist": node_input("m/s; rad/s", (6,))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(SwimSpeedState, state), typed_view(SwimSpeedInputs, boundary_inputs)
        twist = given.get("twist", jnp.zeros(6, dtype=held["speed"].dtype))
        steps = held["steps"] + 1
        elapsed = steps.astype(held["speed"].dtype) * dt
        counted = elapsed > p["after"]
        speed = jnp.dot(twist[:3], jnp.asarray(p["along"], dtype=twist.dtype))

        def after_by() -> Array:
            return elapsed - p["after"]

        def rounded() -> Array:
            return switch.rounding(elapsed)

        distance = held["distance"] + switch.where(counted, speed * dt, 0.0, owner="swim.window", resolution=rounded,
                                                   indicator=after_by)
        observed = held["observed"] + switch.where(counted, dt, 0.0, owner="swim.window", resolution=rounded,
                                                   indicator=after_by)
        some = observed > dt / 2
        mean = switch.where(some, distance / switch.domain_guard(some, observed, 1.0), 0.0, owner="swim.mean",
                            resolution=rounded, indicator=lambda: observed - dt / 2)
        return node_fields(SwimSpeedState(speed=speed, mean_speed=mean, distance=distance, observed=observed,
                                          steps=steps))
