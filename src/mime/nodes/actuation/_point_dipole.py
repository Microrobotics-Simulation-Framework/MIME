"""The field of a point magnetic dipole, and its gradient, at each reader's point: a magnet part's moment, placed and
turned by the frame it is mounted on."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast, override

from jax import Array
from maddening.core.compliance.metadata import NodeMeta
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec

from mime.core import NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_input, quaternion, switch, typed_view
from mime.nodes.actuation._chain import POSE_UNITS, ChainInputs, MountedNode, flattened
from mime.units import VACUUM_PERMEABILITY

#: ``mu_0 / (4 pi)``, in T m / A.
DIPOLE_CONSTANT = VACUUM_PERMEABILITY / (4.0 * math.pi)
#: Where the symbol rows' functions are.
_MODULE = "mime.nodes.actuation._point_dipole"
_NODE = f"{_MODULE}.PointDipoleFieldNode"


def point_dipole(moment: Array, centre: Array, point: Array) -> tuple[Array, Array]:
    """The flux density ``B`` (T) of a point dipole of ``moment`` (A m^2) at ``centre``, at ``point`` (both m, in one
    frame), and its gradient ``G_ij = dB_i/dx_j`` (T/m). At the centre the field is singular; there both are zero,
    by the switch ``point_dipole_field.centre``, which no reader outside the magnet meets."""
    offset = point - centre
    squared = jnp.sum(offset * offset)
    away = squared > 0
    distance = jnp.sqrt(switch.domain_guard(away, squared, 1.0))
    along = jnp.dot(moment, offset)
    field = DIPOLE_CONSTANT * (3.0 * along * offset / distance**5 - moment / distance**3)
    gradient = (3.0 * DIPOLE_CONSTANT / distance**5) * (
        along * jnp.eye(3, dtype=offset.dtype) + jnp.outer(moment, offset) + jnp.outer(offset, moment)
        - 5.0 * along * jnp.outer(offset, offset) / distance**2)

    def resolution() -> Array:
        return switch.rounding(jnp.sum(point * point) + jnp.sum(centre * centre))

    field = switch.where(away, field, jnp.zeros_like(field), owner="point_dipole_field.centre", resolution=resolution,
                         indicator=squared)
    gradient = switch.where(away, gradient, jnp.zeros_like(gradient), owner="point_dipole_field.centre",
                            resolution=resolution, indicator=squared)
    return field, gradient


EQUATIONS = (r"\mathbf{B} = \frac{\mu_{0}}{4 \pi} \, \frac{3 (\mathbf{m} \cdot \mathbf{r}) \, \mathbf{r} / r^{2} "
             r"- \mathbf{m}}{r^{3}}, \quad "
             r"G_{ij} = \frac{\partial B_{i}}{\partial x_{j}} = \frac{3 \mu_{0}}{4 \pi r^{5}} \left( (\mathbf{m} \cdot "
             r"\mathbf{r}) \, \delta_{ij} + m_{i} r_{j} + r_{i} m_{j} - \frac{5 (\mathbf{m} \cdot \mathbf{r}) \, r_{i} "
             r"r_{j}}{r^{2}} \right), \quad \mathbf{r} = \mathbf{x} - \mathbf{c}; \quad "
             r"(\mathbf{c}, \mathbf{q}) = \mathbf{M}_{j} \circ \mathbf{O}, \quad \mathbf{m} = R(\mathbf{q}) \, "
             r"\mathbf{p}")


class PointDipoleFieldNode(MountedNode):
    """The flux density and its gradient, at each reader slot's point (the position of the pose it takes as
    ``points_<slot>``), of a point dipole: the moment ``moment`` of a magnet part, in the part's frame, at the part's
    centroid, the part placed by ``placement`` on the frame it is mounted on. Each slot publishes
    ``flux_density_<slot>`` and ``gradient_<slot>``. Faithful only far from the magnet (MIME-ANO-100): ``magnet_radius``, the radius of the
    smallest sphere about the part's centroid that holds it, is the length that far field is judged against."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-106",
        description="A point magnetic dipole: the flux density and its gradient at each reader's point, from a magnet "
                    "part's moment placed and turned by the frame it is mounted on.",
        governing_equations=EQUATIONS,
        assumptions=("A point dipole: faithful only far from the magnet, against its radius (MIME-ANO-100)",
                     "A rigid, uniform magnetisation along the part's declared direction"),
    )
    symbols = (
        Symbol(r"\mathbf{B}", SymbolKind.STATE, "flux_density_{slot}", "T",
               f"{_MODULE}.point_dipole"),
        Symbol("G", SymbolKind.STATE, "gradient_{slot}", "T/m", f"{_MODULE}.point_dipole"),
        Symbol(r"\mu_{0}", SymbolKind.CONSTANT, "mime.units.VACUUM_PERMEABILITY", "N A^-2",
               f"{_MODULE}.point_dipole"),
        Symbol(r"\mathbf{x}", SymbolKind.INPUT, "points_{slot}", POSE_UNITS,
               f"{_NODE}.advance"),
        Symbol(r"\mathbf{c}", SymbolKind.STATE, "centre", "m",
               f"{_NODE}.advance"),
        Symbol(r"\mathbf{q}", SymbolKind.STATE, "orientation", "1",
               f"{_NODE}.advance"),
        Symbol(r"\mathbf{M}", SymbolKind.INPUT, "mount_poses", POSE_UNITS,
               "mime.nodes.actuation._chain.MountedNode.base"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS,
               "mime.nodes.actuation._chain.MountedNode.base"),
        Symbol(r"\mathbf{p}", SymbolKind.PARAMETER, "moment", "A m^2",
               f"{_NODE}.advance"),
    )

    def __init__(self, name: str, timestep: float, *, slots: Sequence[str], moment: Sequence[float],
                 placement: Sequence[float], magnet_radius: float, mount_index: int = -1, mount_count: int = 0,
                 initial_mount_pose: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)) -> None:
        super().__init__(name, timestep, slots=tuple(slots), moment=flattened(moment), placement=flattened(placement),
                         magnet_radius=float(magnet_radius), mount_index=int(mount_index),
                         mount_count=int(mount_count), initial_mount_pose=flattened(initial_mount_pose))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "moment": ParamSpec(units="A m^2"),
                "placement": ParamSpec(units=POSE_UNITS),
                "magnet_radius": ParamSpec(trainable=False, units="m", description="the far field is judged against "
                                                                                   "it; the field does not read it")}

    def _slots(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.constants()["slots"])

    def _fields(self, part: Array, readings: Mapping[str, tuple[Array, Array]]) -> NodeFields:
        out: NodeFields = {"centre": part[:3], "orientation": part[3:]}
        for slot, (field, gradient) in readings.items():
            out[f"flux_density_{slot}"] = field
            out[f"gradient_{slot}"] = gradient
        return out

    @override
    def initial_state(self) -> NodeFields:
        part = self.initial_base()
        zero = (as_run_array(jnp.zeros(3)), as_run_array(jnp.zeros((3, 3))))
        return self._fields(part, {slot: zero for slot in self._slots()})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {**{f"points_{slot}": node_input(POSE_UNITS, (7,)) for slot in self._slots()}, **self.mounted_inputs()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        part = self.base(p, typed_view(ChainInputs, {k: v for k, v in boundary_inputs.items() if k == "mount_poses"}))
        moment = quaternion.rotate(part[3:], jnp.asarray(p["moment"], dtype=part.dtype))
        readings = {slot: point_dipole(moment, part[:3], boundary_inputs[f"points_{slot}"][:3])
                    for slot in self._slots()}
        return self._fields(part, readings)


__all__ = ["DIPOLE_CONSTANT", "EQUATIONS", "PointDipoleFieldNode", "point_dipole"]
