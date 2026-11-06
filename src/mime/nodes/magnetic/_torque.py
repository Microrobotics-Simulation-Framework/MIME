"""The torque of a uniform field on a permanent magnetic moment fixed in a body."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import (
    MimeNode,
    NodeFields,
    Symbol,
    SymbolKind,
    as_run_array,
    jnp,
    node_fields,
    node_input,
    quaternion,
    typed_view,
)
from mime.nodes.body import Wrench

EQUATIONS = r"\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}, \quad \mathbf{m} = R(\mathbf{q}) \, \mathbf{m}_{b}"

#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.magnetic._torque.MagneticTorqueNode.advance"


class MagneticTorqueInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    flux_density: NotRequired[ReadOnly[Float[Array, "3"]]]


class MagneticTorqueNode(MimeNode):
    """``tau = (R_q m) x B``, with ``m`` the body-frame moment and ``q`` the body's orientation: the wrench ``[0;
    tau]`` about the centre of mass. No force: that is the field gradient's, another law's."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-114",
        description="The torque of a magnetic field on a body's permanent moment, tau = m x B, as a wrench about the "
                    "centre of mass with no force.",
        governing_equations=EQUATIONS,
        assumptions=("A point dipole with a rigid, permanent moment fixed in the body",
                     "The field is read at one point, where the law reads it: it is taken uniform across the "
                     "magnet",
                     "The body's orientation is the one read at the start of the step, so the torque of a step turns "
                     "the body as it stood after the step before"),
        references=(Reference("Ling2016UniversityPhysics2", "the torque on a magnetic dipole in a field"),),
    )
    symbols = (
        Symbol(r"\boldsymbol{\tau}", SymbolKind.STATE, "wrench", "N; N m", _ADVANCE),
        Symbol(r"\mathbf{B}", SymbolKind.INPUT, "flux_density", "T", _ADVANCE),
        Symbol(r"\mathbf{m}_{b}", SymbolKind.PARAMETER, "moment", "A m^2", _ADVANCE),
        Symbol(r"\mathbf{q}", SymbolKind.INPUT, "pose", "m; 1", _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, moment: Sequence[float]) -> None:
        super().__init__(name, timestep, moment=tuple(moment))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "moment": ParamSpec(units="A m^2")}

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
