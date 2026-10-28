"""A lab's package in miniature: it owns the ``labx`` namespace and registers an interaction and a chain segment
there, beside MIME's toy entries, without editing MIME. Its ``mime.entries`` entry point is how a saved experiment
that uses it finds it. Its settings declare their dimensions, and its nodes type their state, as MIME's own entries and
nodes do.

The chain segment, ``labx.gantry``, is a linear stage described by its own settings, not by a URDF file: it publishes
the same frame poses as MIME's segments, so a part mounted on its carriage, and the field of a magnet there, work as
they do on MIME's arm."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from maddening.core.node import SimulationNode
from typing_extensions import ReadOnly, TypedDict

from maddening.core.node import BoundaryInputSpec

from mime.build import DynamicsBuilder, InteractionBuilder, ResolvedRegistration
from mime.core import NodeFields, node_fields, typed_view
from mime.core.multibody import compose
from mime.gradients import GradientStatus
from mime.registry import (
    BuildContext,
    ChainSegment,
    Command,
    Delivery,
    Interaction,
    Problem,
    RoleKind,
    Segment,
    register_dynamics,
    register_interaction,
)
from mime.testing.toys import LOAD
from mime.units import LENGTH, Force, Magnitude, N, Quantity
from mime.vocabulary import Capability, register_namespace

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


class GantryState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "2 7"]]


@jax.jit
def gantry_poses(placement: Array, travel: Array, position: Array) -> Array:
    """The base's pose and the carriage's, ``position`` along ``travel`` from it: compiled once, for every gantry,
    since a step run eagerly is a hundred small calls."""
    slid = jnp.concatenate([position * travel, jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=position.dtype)])
    return jnp.stack([placement, compose(placement, slid)])


class GantryNode(SimulationNode):
    """A carriage sliding along ``travel`` from its base, placed in the world by ``placement``: its frames' poses,
    the base's and the carriage's, as a chain segment publishes them."""

    def __init__(self, name: str, timestep: float, *, travel: Sequence[float], placement: Sequence[float],
                 initial_position: float) -> None:
        super().__init__(name, timestep, travel=tuple(float(c) for c in travel),
                         placement=tuple(float(c) for c in placement), initial_position=float(initial_position))

    def poses(self, params: Mapping[str, Any], position: Array) -> Array:
        dtype = position.dtype
        return gantry_poses(jnp.asarray(params["placement"], dtype=dtype), jnp.asarray(params["travel"], dtype=dtype),
                            position)

    @override
    def initial_state(self) -> NodeFields:
        held = cast(dict[str, Any], self.params)
        return node_fields(GantryState(frame_poses=self.poses(held, jnp.asarray(held["initial_position"],
                                                                                dtype=jnp.zeros(()).dtype))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        return {"position": BoundaryInputSpec(shape=(), dtype=dtype, default=jnp.zeros((), dtype=dtype),
                                              expected_units="m")}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        held = {**cast(dict[str, Any], self.params), **(params or {})}
        dtype = typed_view(GantryState, state)["frame_poses"].dtype
        position = jnp.asarray(boundary_inputs.get("position", jnp.zeros(())), dtype=dtype)
        return node_fields(GantryState(frame_poses=self.poses(held, position)))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class Gantry(ChainSegment):
    """A linear stage, described by its own settings rather than a URDF: a carriage at the commanded ``position``
    (m) along ``travel``, a direction in its base frame. Its frames are ``base`` and ``carriage``. It is placed in the
    world, and mounts on nothing."""

    name = "labx.gantry"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (GantryNode,)}
    flags = frozenset({Capability.RIGID})
    travel: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @override
    def segment(self, settings: Mapping[str, Magnitude]) -> Segment:
        identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        problems = (Problem("labx.gantry-mounted", "the gantry mounts on nothing; it is placed in the world",
                            "drop mount=, and make it the chain's first segment"),) if self.mount else ()
        return Segment(("base", "carriage"), {"position": Command("position", LENGTH)}, problems=problems,
                       initial=(identity, identity))

    @override
    def load_refusal(self) -> str | None:
        return "the gantry's carriage position is prescribed, so a load would not move it"

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(GantryNode(ctx.node_name(), ctx.timestep, travel=self.travel, placement=self.placement(ctx.settings),
                              initial_position=0.0))
