"""Weight and buoyancy on a body in a medium of constant density, at rest or flowing."""

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

EQUATIONS = (r"\mathbf{F} = m \, \mathbf{g} - \rho_{f} V \, \mathbf{g}, \quad \boldsymbol{\tau} = R(\mathbf{q}) \, "
             r"\mathbf{c}_{b} \times \left( -\rho_{f} V \, \mathbf{g} \right)")

#: Where the symbol rows' functions are.
_WRENCH = "mime.nodes.body._gravity.GravityNode.wrench"


class Wrench(TypedDict, closed=True):
    wrench: ReadOnly[Float[Array, "6"]]


class GravityInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]


class GravityNode(MimeNode):
    """``F = m g - rho_f V g``: the weight at the centre of mass and the buoyancy at the centre of buoyancy, so the
    wrench about the centre of mass carries the couple ``(R_q c) x (-rho_f V g)`` when the centre of buoyancy ``c``
    (body frame) is given; only then does it read the body's pose. ``gravity`` is the world's gravity vector."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-113",
        description="Weight and buoyancy on a body in a medium at rest: the wrench about the centre of mass, with the "
                    "buoyancy's couple where the centre of buoyancy is off it.",
        governing_equations=EQUATIONS,
        assumptions=("The medium is at rest and its density uniform: a flow that carries its own hydrostatic load "
                     "would count the buoyancy twice",
                     "The weight acts at the centre of mass and the buoyancy at the centre of buoyancy, both fixed "
                     "in the body"),
        references=(Reference("TatumClassicalMechanics", "weight, and the buoyant force of a displaced fluid"),),
    )
    symbols = (
        Symbol(r"\mathbf{F}", SymbolKind.STATE, "wrench", "N; N m", _WRENCH),
        Symbol(r"\boldsymbol{\tau}", SymbolKind.STATE, "wrench", "N; N m", _WRENCH),
        Symbol("m", SymbolKind.PARAMETER, "mass", "kg", _WRENCH),
        Symbol(r"\mathbf{g}", SymbolKind.PARAMETER, "gravity", "m/s^2", _WRENCH),
        Symbol(r"\rho_{f}", SymbolKind.PARAMETER, "fluid_density", "kg/m^3", _WRENCH),
        Symbol("V", SymbolKind.PARAMETER, "buoyant_volume", "m^3", _WRENCH),
        Symbol(r"\mathbf{c}_{b}", SymbolKind.PARAMETER, "centre_of_buoyancy", "m", _WRENCH),
    )

    def __init__(self, name: str, timestep: float, *, mass: float, buoyant_volume: float, fluid_density: float,
                 gravity: Sequence[float], centre_of_buoyancy: Sequence[float] | None = None) -> None:
        super().__init__(name, timestep, mass=mass, buoyant_volume=buoyant_volume, fluid_density=fluid_density,
                         gravity=tuple(gravity),
                         centre_of_buoyancy=None if centre_of_buoyancy is None else tuple(centre_of_buoyancy))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "mass": ParamSpec(units="kg"), "gravity": ParamSpec(units="m/s^2"),
                "fluid_density": ParamSpec(units="kg/m^3"), "buoyant_volume": ParamSpec(units="m^3"),
                "centre_of_buoyancy": ParamSpec(units="m")}

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
    def energy(self, p: Mapping[str, Any], configuration: Array) -> Array:
        """The gravity family's energy at the body's pose ``configuration``, of which :meth:`wrench` is the negative
        gradient: ``U = -(m g + F_b) . x - F_b . (R_q c_b)``, ``F_b = -rho_f V g`` the buoyancy, acting at the centre of
        buoyancy ``c_b`` (body frame); its last term only where the law is given ``c_b``, as the wrench's couple is."""
        g = jnp.asarray(p["gravity"], dtype=configuration.dtype)
        buoyancy = -p["fluid_density"] * p["buoyant_volume"] * g
        held = -jnp.dot(p["mass"] * g + buoyancy, configuration[:3])
        if not self._couples():
            return held
        lever = quaternion.rotate(configuration[3:], jnp.asarray(p["centre_of_buoyancy"], dtype=configuration.dtype))
        return held - jnp.dot(buoyancy, lever)

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
