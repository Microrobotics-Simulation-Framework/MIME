"""A body's swim speed along a direction, kept in the observing node's own state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import (
    MimeNode,
    NodeFields,
    Symbol,
    SymbolKind,
    Units,
    as_run_array,
    jnp,
    node_fields,
    node_input,
    switch,
    typed_view,
)

EQUATIONS = (r"s = \mathbf{v} \cdot \mathbf{e}, \quad D = \sum_{k : t_{k} > t_{a}} s_{k} \, h, \quad "
             r"T = \sum_{k : t_{k} > t_{a}} h, \quad \bar{s} = D / T")

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.body._swim_speed.SwimSpeedNode.advance"


class SwimSpeedState(TypedDict, closed=True):
    speed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    mean_speed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    distance: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    observed: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Annotated[Int[Array, ""], Units("1")]]  # noqa: F722 - a jaxtyping shape


class SwimSpeedInputs(TypedDict, closed=True):
    twist: NotRequired[ReadOnly[Float[Array, "6"]]]


class SwimSpeedNode(MimeNode):
    """``speed = v . along`` each step; ``mean_speed``, the distance travelled along ``along`` over the time observed
    (every step whose end lies after ``after``), so it is the mean of ``speed`` over those steps. Its time is its
    step count times its timestep. The time observed is zero or at least one timestep, so the mean is taken from
    half a timestep on, which keeps its derivative finite however the state is drawn."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-115",
        description="A body's velocity along a direction each step, and its mean over the steps that end after a "
                    "start time, kept in the node's own state.",
        governing_equations=EQUATIONS,
        assumptions=("The body's velocity read at a step holds over that step, so the distance is the sum of the "
                     "speed times the timestep",
                     "The time of a step is its count times the timestep, and a step counts whole once its end lies "
                     "after the start time"),
        limitations=("The window's edge is a switch: the distance and the time observed move in jumps of a step as "
                     "the start time moves, and the mean is zero until some time has been observed",),
    )
    symbols = (
        Symbol("s", SymbolKind.STATE, "speed", "m/s", _ADVANCE),
        Symbol("D", SymbolKind.STATE, "distance", "m", _ADVANCE),
        Symbol("T", SymbolKind.STATE, "observed", "s", _ADVANCE),
        Symbol(r"\bar{s}", SymbolKind.STATE, "mean_speed", "m/s", _ADVANCE),
        Symbol(r"\mathbf{v}", SymbolKind.INPUT, "twist", "m/s; rad/s", _ADVANCE),
        Symbol(r"\mathbf{e}", SymbolKind.PARAMETER, "along", "1", _ADVANCE),
        Symbol("t_{a}", SymbolKind.PARAMETER, "after", "s", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, along: Sequence[float], after: float = 0.0) -> None:
        super().__init__(name, timestep, along=tuple(along), after=after)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "along": ParamSpec(units="1"), "after": ParamSpec(units="s")}

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
