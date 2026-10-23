"""Weight and buoyancy on a body in a medium at rest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_fields, node_input, quaternion, typed_view


class Wrench(TypedDict, closed=True):
    wrench: ReadOnly[Float[Array, "6"]]


class GravityInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]


class GravityNode(MimeNode):
    """``F = m g - rho_f V g``: the weight at the centre of mass and the buoyancy at the centre of buoyancy, so the
    wrench about the centre of mass carries the couple ``(R_q c) x (-rho_f V g)`` when the centre of buoyancy ``c``
    (body frame) is given; only then does it read the body's pose. ``gravity`` is the world's gravity vector."""

    def __init__(self, name: str, timestep: float, *, mass: float, buoyant_volume: float, fluid_density: float,
                 gravity: Sequence[float], centre_of_buoyancy: Sequence[float] | None = None) -> None:
        super().__init__(name, timestep, mass=mass, buoyant_volume=buoyant_volume, fluid_density=fluid_density,
                         gravity=tuple(gravity),
                         centre_of_buoyancy=None if centre_of_buoyancy is None else tuple(centre_of_buoyancy))

    def _couples(self) -> bool:
        return self.constants()["centre_of_buoyancy"] is not None

    def wrench(self, p: Mapping[str, Any], pose: Array | None) -> Array:
        g = jnp.asarray(p["gravity"])
        buoyancy = -p["fluid_density"] * p["buoyant_volume"] * g
        force = p["mass"] * g + buoyancy
        if pose is None:
            return jnp.concatenate([force, jnp.zeros_like(force)])
        lever = quaternion.rotate(pose[3:], jnp.asarray(p["centre_of_buoyancy"], dtype=pose.dtype))
        return jnp.concatenate([force, jnp.cross(lever, buoyancy)])

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(Wrench(wrench=as_run_array(jnp.zeros(6))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,))} if self._couples() else {}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(GravityInputs, boundary_inputs)
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))) \
            if self._couples() else None
        return node_fields(Wrench(wrench=as_run_array(self.wrench(p, pose))))
