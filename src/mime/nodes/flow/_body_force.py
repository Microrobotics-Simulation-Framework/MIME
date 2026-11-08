"""The body force a law bound to a medium drives its fluid with: a uniform force per unit mass along a direction,
steady or harmonic, written into the flow field's body-force input each step at the step's end, the instant IMEX
Euler's stage evaluates its drive at (the node keeps its own integer step count, so the time is exact)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float, Int
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.params import ParamSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, typed_view

EQUATIONS = r"\mathbf{g}_{n} = \hat{\mathbf{e}} \left( g_{0} + g_{1} \cos\left( 2 \pi f n h + \varphi \right) \right)"

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.flow._body_force.BodyForceNode.advance"


class BodyForceState(TypedDict, closed=True):
    force: ReadOnly[Float[Array, "3"]]  # noqa: F722 - a jaxtyping shape
    steps: ReadOnly[Int[Array, ""]]  # noqa: F722 - a jaxtyping shape


class BodyForceNode(MimeNode):
    """``along (steady + amplitude cos(2 pi frequency t + phase))`` at the end of each step: every number a ``params``
    leaf, so a study moves it without a rebuild."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-122",
        description="A medium's body force per unit mass, steady or harmonic along one direction, written into its flow "
                    "field each step at the step's end.",
        governing_equations=EQUATIONS,
        assumptions=("The force is uniform over the medium's fluid: the flow field applies it to the fluid alone",
                     "The time of a step is its count times the timestep, never a sum of timesteps"),
    )
    symbols = (
        Symbol(r"\mathbf{g}", SymbolKind.STATE, "force", "m/s^2", _ADVANCE),
        Symbol(r"\hat{\mathbf{e}}", SymbolKind.PARAMETER, "along", "1", _ADVANCE),
        Symbol("g_{0}", SymbolKind.PARAMETER, "steady", "m/s^2", _ADVANCE),
        Symbol("g_{1}", SymbolKind.PARAMETER, "amplitude", "m/s^2", _ADVANCE),
        Symbol("f", SymbolKind.PARAMETER, "frequency", "Hz", _ADVANCE),
        Symbol(r"\varphi", SymbolKind.PARAMETER, "phase", "rad", _ADVANCE),
        Symbol("n", SymbolKind.STATE, "steps", "1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, along: Sequence[float], steady: float, amplitude: float = 0.0,
                 frequency: float = 0.0, phase: float = 0.0) -> None:
        super().__init__(name, timestep, along=tuple(float(c) for c in along), steady=float(steady),
                         amplitude=float(amplitude), frequency=float(frequency), phase=float(phase))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "along": ParamSpec(units="1", description="a unit direction"),
                "steady": ParamSpec(units="m/s^2"), "amplitude": ParamSpec(units="m/s^2"),
                "frequency": ParamSpec(units="Hz"), "phase": ParamSpec(units="rad")}

    def force_at(self, p: Mapping[str, Any], time: Any) -> Array:
        wave = as_run_array(p["amplitude"]) * jnp.cos(2.0 * jnp.pi * as_run_array(p["frequency"]) * time
                                                      + as_run_array(p["phase"]))
        return as_run_array(p["along"]) * (as_run_array(p["steady"]) + wave)

    @override
    def initial_state(self) -> NodeFields:
        p = self.constants()
        return node_fields(BodyForceState(force=self.force_at(p, as_run_array(0.0)),
                                          steps=jnp.asarray(0, dtype=jnp.int32)))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held = typed_view(BodyForceState, state)
        steps = held["steps"] + 1
        time = steps.astype(as_run_array(0.0).dtype) * dt
        return node_fields(BodyForceState(force=self.force_at(p, time), steps=steps))


__all__ = ["BodyForceNode", "BodyForceState"]
