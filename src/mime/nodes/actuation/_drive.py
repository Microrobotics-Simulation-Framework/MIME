"""Command generators: nodes whose numbers are parameters, written into an apparatus's commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, node_fields


class Command(TypedDict, closed=True):
    value: ReadOnly[Float[Array, "..."]]


class ConstantNode(MimeNode):
    """``value``, every step: a ``params`` leaf, so a study can sweep it without a rebuild."""

    def __init__(self, name: str, timestep: float, *, value: float | Sequence[float]) -> None:
        super().__init__(name, timestep, value=float(value) if isinstance(value, (int, float)) else tuple(value))

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(Command(value=as_run_array(self.constants()["value"])))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return node_fields(Command(value=as_run_array(p["value"])))
