"""Command generators: nodes whose numbers are parameters, written into an apparatus's commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.params import ParamSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, switch, typed_view

EQUATIONS = r"\mathbf{u}_{n} = \mathbf{c}"

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.actuation._drive.ConstantNode.advance"


class Command(TypedDict, closed=True):
    value: ReadOnly[Float[Array, "..."]]


class ConstantNode(MimeNode):
    """``value``, every step: a ``params`` leaf, so a study can sweep it without a rebuild."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-117",
        description="A command generator: the value it was built with, written into the command every step.",
        governing_equations=EQUATIONS,
        assumptions=("The value is the same at every step: it is a parameter of the graph, never a function of time "
                     "or of anything the run computes",),
    )
    symbols = (
        Symbol(r"\mathbf{u}", SymbolKind.STATE, "value", "the command's own", _ADVANCE),
        Symbol(r"\mathbf{c}", SymbolKind.PARAMETER, "value", "the command's own", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, value: float | Sequence[float]) -> None:
        super().__init__(name, timestep, value=float(value) if isinstance(value, (int, float)) else tuple(value))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "value": ParamSpec(description="in the units of the command it is written "
                                                                        "into, which the node cannot know")}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(Command(value=as_run_array(self.constants()["value"])))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return node_fields(Command(value=as_run_array(p["value"])))


RAMP_EQUATIONS = (r"u_{n} = u_{0} + r \min\left( \max\left( \left( n - \tfrac{1}{2} \right) h - t_{h}, 0 \right), T "
                  r"\right)")

#: Where the ramp's symbol rows' function is.
_RAMP = "mime.nodes.actuation._drive.RampNode.advance"


class RampState(TypedDict, closed=True):
    value: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


class RampNode(MimeNode):
    """A command that holds ``start`` for ``hold`` seconds, then moves at ``rate`` (signed, towards where it ends) for
    ``duration`` seconds, then holds where it ends. Its value at step ``n`` is the ramp's at the step's middle,
    ``(n - 1/2) h``, which is its mean over the step wherever the ramp's start and end fall on steps' ends: so a field
    that integrates the value it is commanded each step (``IntegratingRotatingNode``) turns by the ramp's exact
    integral. Every number is a ``params`` leaf, so a study can sweep it without a rebuild."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-120",
        description="A command generator: a value held, then moved at a rate to where it ends and held there, written "
                    "into the command every step as its value at the step's middle.",
        governing_equations=RAMP_EQUATIONS,
        assumptions=("The value written for a step is the ramp's value at the step's middle, its mean over the step "
                     "where the ramp neither starts nor ends inside the step",
                     "The time of a step is its count times the timestep, never a sum of timesteps"),
        limitations=("In the step inside which the ramp starts or ends, the value at the middle is not the mean over "
                     "the step: they differ by at most the rate times an eighth of a step",),
    )
    symbols = (
        Symbol("u_{n}", SymbolKind.STATE, "value", "the command's own", _RAMP),
        Symbol("u_{0}", SymbolKind.PARAMETER, "start", "the command's own", _RAMP),
        Symbol("r", SymbolKind.PARAMETER, "rate", "the command's own per second", _RAMP),
        Symbol("T", SymbolKind.PARAMETER, "duration", "s", _RAMP),
        Symbol("t_{h}", SymbolKind.PARAMETER, "hold", "s", _RAMP),
        Symbol("n", SymbolKind.STATE, "steps", "1", _RAMP),
    )

    def __init__(self, name: str, timestep: float, *, start: float, rate: float, duration: float,
                 hold: float = 0.0) -> None:
        super().__init__(name, timestep, start=float(start), rate=float(rate), duration=float(duration),
                         hold=float(hold))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        own = "in the units of the command it is written into, which the node cannot know"
        return {**super().param_specs(), "start": ParamSpec(description=own),
                "rate": ParamSpec(description=f"{own}, per second"), "duration": ParamSpec(units="s"),
                "hold": ParamSpec(units="s")}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(RampState(value=as_run_array(self.constants()["start"]),
                                     steps=jnp.asarray(0, dtype=jnp.int32)))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held = typed_view(RampState, state)
        steps = held["steps"] + 1
        zero = as_run_array(0.0)
        middle = (steps.astype(zero.dtype) - 0.5) * dt
        hold, duration = as_run_array(p["hold"]), as_run_array(p["duration"])

        def rounded() -> Array:
            return switch.rounding(middle)

        since = switch.maximum(middle - hold, zero, owner="ramp.start", indicator=lambda: middle - hold,
                               resolution=rounded)
        along = switch.minimum(since, duration, owner="ramp.end", indicator=lambda: duration - (middle - hold),
                               resolution=rounded)
        return node_fields(RampState(value=as_run_array(p["start"]) + as_run_array(p["rate"]) * along, steps=steps))
