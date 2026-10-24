"""A lab's package in miniature: it owns the ``labx`` namespace and registers one interaction there, beside MIME's
toy entries, without editing MIME. Its ``mime.entries`` entry point is how a saved experiment that uses it finds it.
Its setting declares its dimension, and its node types its state, as MIME's own entries and nodes do."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from maddening.core.node import SimulationNode
from typing_extensions import ReadOnly, TypedDict

from mime.build import InteractionBuilder
from mime.core import NodeFields, node_fields, typed_view
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, RoleKind, register_interaction
from mime.testing.toys import LOAD
from mime.units import Force, N, Quantity
from mime.vocabulary import register_namespace

register_namespace("labx", "0.1.0")


class NudgeState(TypedDict, closed=True):
    load: ReadOnly[Float[Array, "3"]]


class NudgeNode(SimulationNode):
    """A constant load."""

    def __init__(self, name: str, timestep: float, force: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, force=tuple(force))

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(NudgeState(load=jnp.asarray(cast(dict[str, Any], self.params)["force"],
                                                       dtype=jnp.zeros(()).dtype)))

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        force = cast(dict[str, Any], self.params)["force"] if params is None else params["force"]
        return node_fields(NudgeState(load=jnp.asarray(force, dtype=typed_view(NudgeState, state)["load"].dtype)))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Nudge(Interaction):
    """A constant load on a toy body."""

    name = "labx.nudge"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (NudgeNode,)}
    roles = {"body": RoleKind.BODY}
    delivers = {"load": Delivery("body", LOAD.name)}
    force: Annotated[Quantity, Force] = N((0.0, 0.0, -0.05))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(NudgeNode(ctx.node_name(), ctx.timestep, force=ctx.vector("force")))
        b.connect(node["load"], into=b.target("load"))
