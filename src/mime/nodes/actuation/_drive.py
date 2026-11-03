"""Command generators: nodes whose numbers are parameters, written into an apparatus's commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.params import ParamSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, node_fields

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
