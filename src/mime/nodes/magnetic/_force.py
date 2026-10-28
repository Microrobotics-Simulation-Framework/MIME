"""The force of a field's gradient on a permanent magnetic moment fixed in a body, and its torque about the centre of
mass where the magnet sits off it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    quaternion, typed_view
from mime.nodes.body import Wrench

EQUATIONS = (r"\mathbf{F} = \mathbf{G}^{\top} \mathbf{m}, \quad \boldsymbol{\tau} = \mathbf{l} \times \mathbf{F}, "
             r"\quad \mathbf{m} = R(\mathbf{q}) \, \mathbf{m}_{b}, \quad \mathbf{l} = R(\mathbf{q}) \, \mathbf{c}_{b}")


#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.magnetic._force.MagneticForceNode.advance"


class MagneticForceInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    gradient: NotRequired[ReadOnly[Float[Array, "3 3"]]]


class MagneticForceNode(MimeNode):
    """``F = G^T m``, with ``G_ij = dB_i/dx_j`` the field's gradient where the body is and ``m`` its body-frame
    ``moment`` turned by the body's orientation ``q``: the force on a point dipole in a current-free field. It acts at
    the magnet's centroid, ``magnet_centre`` in the body frame turned by ``q``, so the wrench about the centre of mass
    is ``[F; l x F]``. No torque of the field itself: that is ``m x B``, another law's."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-107",
        description="The force of a field's gradient on a body's permanent moment, F = G^T m, and its torque about the "
                    "centre of mass where the magnet sits off it.",
        governing_equations=EQUATIONS,
        assumptions=("A point dipole in a current-free field, the gradient read at the body's centre of mass",
                     "A rigid, permanent moment"),
        references=(Reference("Jackson1998", "the force on a magnetic dipole in a field's gradient"),),
    )
    symbols = (
        Symbol(r"\mathbf{F}", SymbolKind.STATE, "wrench", "N; N m", _ADVANCE),
        Symbol(r"\boldsymbol{\tau}", SymbolKind.STATE, "wrench", "N; N m",
               _ADVANCE),
        Symbol(r"\mathbf{G}", SymbolKind.INPUT, "gradient", "T/m", _ADVANCE),
        Symbol(r"\mathbf{m}_{b}", SymbolKind.PARAMETER, "moment", "A m^2",
               _ADVANCE),
        Symbol(r"\mathbf{q}", SymbolKind.INPUT, "pose", "m; 1", _ADVANCE),
        Symbol(r"\mathbf{c}_{b}", SymbolKind.PARAMETER, "magnet_centre", "m",
               _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, moment: Sequence[float], magnet_centre: Sequence[float]) -> None:
        super().__init__(name, timestep, moment=tuple(float(c) for c in moment),
                         magnet_centre=tuple(float(c) for c in magnet_centre))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "moment": ParamSpec(units="A m^2"), "magnet_centre": ParamSpec(units="m")}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(Wrench(wrench=as_run_array(jnp.zeros(6))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input("m; 1", (7,)), "gradient": node_input("T/m", (3, 3))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(MagneticForceInputs, boundary_inputs)
        gradient = given.get("gradient", as_run_array(jnp.zeros((3, 3))))
        pose = given.get("pose", as_run_array(jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))))
        moment = quaternion.rotate(pose[3:], jnp.asarray(p["moment"], dtype=gradient.dtype))
        lever = quaternion.rotate(pose[3:], jnp.asarray(p["magnet_centre"], dtype=gradient.dtype))
        force = gradient.T @ moment
        return node_fields(Wrench(wrench=jnp.concatenate([force, jnp.cross(lever, force)])))


__all__ = ["EQUATIONS", "MagneticForceInputs", "MagneticForceNode"]
