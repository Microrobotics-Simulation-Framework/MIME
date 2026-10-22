"""A lab's package in miniature: it owns the ``labx`` namespace and registers one interaction there, beside MIME's
toy entries, without editing MIME. Its ``mime.entries`` entry point is how a saved experiment that uses it finds it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax.numpy as jnp
from jax import Array
from maddening.core.node import SimulationNode

from mime.build import InteractionBuilder
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, RoleKind, register_interaction
from mime.testing.toys import LOAD
from mime.units import Force, N, Quantity
from mime.vocabulary import register_namespace

register_namespace("labx", "0.1.0")


class NudgeNode(SimulationNode):
    """A constant load."""

    def __init__(self, name: str, timestep: float, force: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, force=tuple(force))

    @override
    def initial_state(self) -> dict[str, Array]:
        return {"load": jnp.asarray(cast(dict[str, Any], self.params)["force"], dtype=jnp.zeros(()).dtype)}

    @override
    def update(self, state: dict[str, Array], boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> dict[str, Array]:
        force = cast(dict[str, Any], self.params)["force"] if params is None else params["force"]
        return {"load": jnp.asarray(force, dtype=state["load"].dtype)}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Nudge(Interaction):
    """A constant load on a toy body."""

    name = "labx.nudge"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    delivers = {"load": Delivery("body", LOAD.name)}
    force: Annotated[Quantity, Force] = N((0.0, 0.0, -0.05))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(NudgeNode(ctx.node_name(), ctx.timestep, force=ctx.vector("force")))
        b.connect(node["load"], into=b.target("load"))
