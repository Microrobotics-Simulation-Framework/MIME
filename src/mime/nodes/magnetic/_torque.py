"""The torque of a uniform field on a permanent magnetic moment fixed in a body."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_fields, node_input, quaternion, typed_view
from mime.nodes.body import Wrench


class MagneticTorqueInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    flux_density: NotRequired[ReadOnly[Float[Array, "3"]]]


class MagneticTorqueNode(MimeNode):
    """``tau = (R_q m) x B``, with ``m`` the body-frame moment and ``q`` the body's orientation: the wrench ``[0;
    tau]`` about the centre of mass. No force: that is the field gradient's, another law's."""

    def __init__(self, name: str, timestep: float, *, moment: Sequence[float]) -> None:
        super().__init__(name, timestep, moment=tuple(moment))

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(Wrench(wrench=as_run_array(jnp.zeros(6))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,)), "flux_density": node_input("T", (3,))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(MagneticTorqueInputs, boundary_inputs)
        field = given.get("flux_density", as_run_array(jnp.zeros(3)))
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        moment = quaternion.rotate(pose[3:], jnp.asarray(p["moment"], dtype=field.dtype))
        return node_fields(Wrench(wrench=jnp.concatenate([jnp.zeros_like(field), jnp.cross(moment, field)])))
