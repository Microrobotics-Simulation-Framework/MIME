"""A stand-in for a field kind's unit field (F0's item 9.2: computed in the program from what it depends on, shared by
``vmap`` where that agrees): a spring on the toy probe whose stiffness is ``k`` plus ``amplitude`` times the first entry
of a unit field, the solution of ``(A + nu I) f = b`` for a fixed 16 x 16 ``A``, so that the field depends on the law's
``nu`` (the stand-in for a viscosity) and not on its ``amplitude`` (the stand-in for a drive's). The field is the node's
run-constant (``MimeNode.run_constants``): held as a leaf made at build with the value of ``nu`` it was made at beside
it (``held_from.nu``), and made from ``nu`` once per chunk inside the program where ``nu`` has moved, as the MRS drag's
held matrix is (``held_or_made``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
import numpy as np
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec

from mime.build import InteractionBuilder
from mime.core import MimeNode, NodeFields, as_run_array, node_fields, node_input, run_constants_given
from mime.gradients import GradientStatus
from mime.nodes.hydro.mrs._quadrature import held_or_made
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing.toys._entries import LOAD, POSITION
from mime.testing.toys._experiments import ADD_BATH, ADD_PROBE, toy_experiment
from mime.units import Force, Length, N, Quantity, m
from mime.vocabulary import register_namespace

register_namespace("twobuilds_fields", "0.1.0")

SIZE = 16
#: A fixed symmetric positive definite operator and right-hand side.
_RNG = np.random.default_rng(20261108)
_M = _RNG.standard_normal((SIZE, SIZE))
A = _M @ _M.T / SIZE + np.eye(SIZE)
B = _RNG.standard_normal(SIZE)
FIELD = "unit_field"
FROM = "held_from.nu"


def made(theta: Mapping[str, Any]) -> Any:
    """The unit field at ``nu``: ``(A + nu I)^-1 b``."""
    nu = jnp.asarray(theta["nu"])
    return jnp.linalg.solve(jnp.asarray(A, dtype=nu.dtype) + nu * jnp.eye(SIZE, dtype=nu.dtype),
                            jnp.asarray(B, dtype=nu.dtype))


class UnitFieldSpringNode(MimeNode):
    def __init__(self, name: str, timestep: float, k: float = 1.7, x0: Sequence[float] = (0.05, -0.02, 0.01),
                 nu: float = 0.5, amplitude: float = 0.3) -> None:
        super().__init__(name, timestep, k=k, x0=tuple(x0), nu=nu, amplitude=amplitude)

    @override
    def params_pytree(self) -> dict[str, Any]:
        inherited: Any = super()
        leaves = dict(inherited.params_pytree())
        nu = float(self.constants()["nu"])
        return {**leaves, FIELD: as_run_array(np.linalg.solve(A + nu * np.eye(SIZE), B)), FROM: leaves["nu"]}

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held: Any = super()
        return {**held.param_specs(), FIELD: ParamSpec(trainable=False, description="the unit field, made from nu"),
                FROM: ParamSpec(trainable=False, description="the nu the unit field was made at")}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"x": node_input("m", (3,))}

    @override
    def initial_state(self) -> NodeFields:
        return node_fields({"load": as_run_array(np.zeros(3))})

    def _field(self, p: Mapping[str, Any]) -> Any:
        return held_or_made(made, {"nu": p[FROM]})({"nu": p["nu"]}, p[FIELD])

    @override
    def run_constants(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {FIELD: self._field(params)}

    @override
    def advance(self, p: Mapping[str, Any], state: dict[str, Any], boundary_inputs: Mapping[str, Any],
                dt: float) -> dict[str, Any]:
        field = p[FIELD] if run_constants_given(p) else self._field(p)
        x = jnp.asarray(boundary_inputs.get("x", jnp.zeros(3)))
        stiffness = p["k"] + p["amplitude"] * field[0]
        return node_fields({"load": -stiffness * (x - jnp.asarray(p["x0"], dtype=x.dtype))})


@register_interaction
@dataclass(frozen=True, kw_only=True)
class UnitFieldSpring(Interaction):
    """The stand-in spring (the module's docstring)."""

    name = "twobuilds_fields.spring"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (UnitFieldSpringNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    references = ()
    k: Annotated[Quantity, Force / Length] = N(1.7) / m(1.0)
    #: The stand-in's viscosity and drive amplitude, numbers of the law's own.
    nu: float = 0.5
    amplitude: float = 0.3

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(UnitFieldSpringNode(ctx.node_name(), ctx.timestep, k=ctx.scalar("k"),
                                                       nu=float(self.nu), amplitude=float(self.amplitude)))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


def spring_experiment() -> tuple[Any, Any]:
    """The toy probe in its bath, pulled by the stand-in spring; and its handle."""
    exp = toy_experiment("unit_fields")
    ADD_BATH.apply(exp)
    ADD_PROBE.apply(exp)
    return exp, exp.attach(UnitFieldSpring(), body="probe")


__all__ = ["UnitFieldSpring", "UnitFieldSpringNode", "made", "spring_experiment"]
