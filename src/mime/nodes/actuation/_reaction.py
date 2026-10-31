"""The reaction of a law's wrench on a body, on the source the law reads: equal and opposite, moved from the body's
centre of mass to the origin of the frame the source is mounted on, in that frame's row of the loads its segment
takes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast, override

import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    typed_view
from mime.nodes.actuation._chain import POSE_UNITS

EQUATIONS = (r"\mathbf{W}_{k} = \begin{bmatrix} -\mathbf{F} \\ -\boldsymbol{\tau} - (\mathbf{c} - \mathbf{o}_{k}) "
             r"\times \mathbf{F} \end{bmatrix}")
#: Where the symbol rows' function is.
_ADVANCE = "mime.nodes.actuation._reaction.SourceReactionNode.advance"


class SourceReactionState(TypedDict, closed=True):
    link_wrench: ReadOnly[Float[Array, "frames 6"]]


class SourceReactionInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    wrench: NotRequired[ReadOnly[Float[Array, "6"]]]
    mount_poses: NotRequired[ReadOnly[Float[Array, "frames 7"]]]


class SourceReactionNode(MimeNode):
    """``W_k = [-F; -tau - (c - o_k) x F]``: the reaction of the wrench ``[F; tau]`` a law delivers on a body about its
    centre of mass ``c``, on the source the law reads, which is mounted on frame ``k`` (row ``mount_index`` of the
    ``mount_count`` frames of its segment) with its origin at ``o_k``. The pair sums to zero about any point: the
    law's action and its reaction, equal and opposite, and of equal and opposite moment. The other rows are zero."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-111",
        description="The reaction of a law's wrench on a body, on the source it reads: equal and opposite, its torque "
                    "moved to the origin of the frame the source is mounted on.",
        governing_equations=EQUATIONS,
        assumptions=("The law's force on the body and its reaction on the source are equal and opposite and act "
                     "along one line, as between two point dipoles",),
        references=(Reference("Jackson1998", "the force and torque between two magnetic dipoles"),),
    )
    symbols = (
        Symbol(r"\mathbf{W}", SymbolKind.STATE, "link_wrench", "N; N m", _ADVANCE),
        Symbol(r"\mathbf{F}", SymbolKind.INPUT, "wrench", "N; N m", _ADVANCE),
        Symbol(r"\boldsymbol{\tau}", SymbolKind.INPUT, "wrench", "N; N m", _ADVANCE),
        Symbol(r"\mathbf{c}", SymbolKind.INPUT, "pose", POSE_UNITS, _ADVANCE),
        Symbol(r"\mathbf{o}", SymbolKind.INPUT, "mount_poses", POSE_UNITS, _ADVANCE),
    )

    def __init__(self, name: str, timestep: float, *, mount_index: int, mount_count: int) -> None:
        if not 0 <= mount_index < mount_count:
            raise ValueError(f"mount_index {mount_index} is no row of {mount_count} frames")
        super().__init__(name, timestep, mount_index=int(mount_index), mount_count=int(mount_count))

    def _rows(self) -> tuple[int, int]:
        held = self.constants()
        return cast(int, held["mount_index"]), cast(int, held["mount_count"])

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(SourceReactionState(link_wrench=as_run_array(np.zeros((self._rows()[1], 6)))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"pose": node_input(POSE_UNITS, (7,)), "wrench": node_input("N; N m", (6,)),
                "mount_poses": node_input(POSE_UNITS, (self._rows()[1], 7))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(SourceReactionInputs, boundary_inputs)
        index, count = self._rows()
        wrench = given.get("wrench", as_run_array(jnp.zeros(6)))
        centre = given.get("pose", as_run_array(jnp.zeros(7)))[:3]
        origin = given.get("mount_poses", as_run_array(jnp.zeros((count, 7))))[index, :3]
        force, torque = wrench[:3], wrench[3:]
        row = jnp.concatenate([-force, -torque - jnp.cross(centre - origin, force)])
        return node_fields(SourceReactionState(link_wrench=jnp.zeros((count, 6), dtype=row.dtype).at[index].set(row)))


__all__ = ["EQUATIONS", "SourceReactionInputs", "SourceReactionNode", "SourceReactionState"]
