"""A lab's package in miniature: it owns the ``labx`` namespace and registers one interaction there, beside MIME's
toy entries, without editing MIME."""

from dataclasses import dataclass

import jax.numpy as jnp
from maddening.core.node import SimulationNode

from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, RoleKind, register_interaction
from mime.testing.toys import LOAD
from mime.vocabulary import register_namespace

register_namespace("labx", "0.1.0")


class NudgeNode(SimulationNode):
    """A constant load."""

    def __init__(self, name, timestep, force=(0.0, 0.0, 0.0)):
        super().__init__(name, timestep, force=force)

    def initial_state(self):
        return {"load": jnp.asarray(self.params["force"], dtype=jnp.zeros(()).dtype)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        force = self.params["force"] if params is None else params.get("force", self.params["force"])
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
    force: tuple[float, float, float] = (0.0, 0.0, -0.05)

    def build(self, ctx, b):
        node = b.add_response_node(NudgeNode(ctx.node_name(), ctx.timestep, force=self.force))
        b.connect(node["load"], into=b.target("load"))
