"""Test-only entries and closed forms for the rigid body's oracles. The entries are registered under the ``oracle``
namespace, in this process only, so the registry walk (a fresh process) holds them to no round-trip case: a constant
applied wrench and a constant resistance, each delivered into a body.

The closed forms: a body under a constant force ``F`` against a linear resistance ``c`` approaches ``F / c`` as
``v(t) = F / c (1 - exp(-c t / m))``; turning under a constant torque against a rotational resistance, the same with
the inertia. With isotropic inertia the gyroscopic term is zero and the two are independent."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from maddening.core.node import BoundaryInputSpec
from typing_extensions import ReadOnly, TypedDict

from mime.build import InteractionBuilder
from mime.core import MimeNode, NodeFields, as_run_array, node_fields
from mime.gradients import GradientStatus
from mime.nodes.body import RigidInertialNode
from mime.registry import BuildContext, Delivery, Interaction, RoleKind, register_interaction
from mime.vocabulary import APPLIED_WRENCH, RESISTANCE, register_namespace

register_namespace("oracle", "0.0.1")

MASS, INERTIA = 2.0e-4, 3.0e-10
FORCE, TORQUE = np.array([1.0e-6, -2.0e-6, 0.5e-6]), np.array([2.0e-9, 1.0e-9, -1.0e-9])
LINEAR, ROTATIONAL = 1.0e-3, 1.0e-8
#: The time scales the resistances relax over: m / c = 0.2 s, I / c_r = 0.03 s.
TAU, TAU_ROTATION = MASS / LINEAR, INERTIA / ROTATIONAL
HORIZON = 0.4


class _Constant(TypedDict, closed=True):
    value: ReadOnly[Array]


class ConstantNode(MimeNode):
    """``value`` every step: a wrench (6,) or a resistance (6, 6), flattened in its constructor."""

    def __init__(self, name: str, timestep: float, *, value: Sequence[float], shape: Sequence[int]) -> None:
        super().__init__(name, timestep, value=tuple(value), shape=tuple(shape))

    def _value(self, params: Mapping[str, Any] | None = None) -> Array:
        p = self.constants(params)
        return as_run_array(jnp.reshape(jnp.asarray(p["value"]), tuple(self.constants()["shape"])))

    @override
    def initial_state(self) -> NodeFields:
        return node_fields(_Constant(value=self._value()))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {}

    @override
    def update(self, state: NodeFields, boundary_inputs: Mapping[str, Array], dt: float, *,
               params: Mapping[str, Any] | None = None) -> NodeFields:
        return node_fields(_Constant(value=self._value(params)))


@dataclass(frozen=True, kw_only=True)
class _ConstantDelivery(Interaction):
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    value: tuple[float, ...] = ()

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        (target,) = self.delivers
        shape = (6,) if target == "wrench" else (6, 6)
        node = b.add_response_node(ConstantNode(ctx.node_name(), ctx.timestep, value=self.value, shape=shape))
        b.connect(node["value"], into=b.target(target))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ConstantWrench(_ConstantDelivery):
    name = "oracle.constant_wrench"
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ConstantResistance(_ConstantDelivery):
    name = "oracle.constant_resistance"
    delivers = {"resistance": Delivery("body", RESISTANCE.name)}


def resistance(linear: float, rotational: float) -> tuple[float, ...]:
    return tuple(np.diag([linear] * 3 + [rotational] * 3).ravel())


def terminal_approach(t: np.ndarray) -> np.ndarray:
    """``[v; omega](t)`` from rest, by the closed form, at each time in ``t``."""
    v = np.outer(1 - np.exp(-t / TAU), FORCE / LINEAR)
    w = np.outer(1 - np.exp(-t / TAU_ROTATION), TORQUE / ROTATIONAL)
    return np.concatenate([v, w], axis=1)


class ExplicitNode(RigidInertialNode):
    """The planted control: the resistance integrated explicitly, ``V' = V + dt M^-1 (F - R V)``, as 0.2 did."""

    @override
    def velocities(self, system: Array, resistance: Array, twist: Array, load: Array) -> Array:
        # The probe's inertia is isotropic, so the gyroscopic Jacobian is zero and the system is ``M/dt + R``.
        return twist + jnp.linalg.solve(system - resistance, load - resistance @ twist)


def body(dt: float, cls: type[RigidInertialNode] = RigidInertialNode) -> RigidInertialNode:
    return cls("probe", dt, mass=MASS, inertia=(INERTIA,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
               initial_twist=(0.0,) * 6)


def twists(dt: float, steps: int, cls: type[RigidInertialNode] = RigidInertialNode) -> tuple[np.ndarray, np.ndarray]:
    """The node driven by the constant wrench and resistance: its twist after each of ``steps`` steps, and its
    pose's quaternion norm after each."""
    node = body(dt, cls)
    given = {"applied_wrench": jnp.asarray(np.concatenate([FORCE, TORQUE])),
             "resistance": jnp.asarray(np.diag([LINEAR] * 3 + [ROTATIONAL] * 3))}

    def advance(state: dict[str, Array], _: Any) -> tuple[dict[str, Array], tuple[Array, Array]]:
        state = node.update(state, given, dt)
        return state, (state["twist"], jnp.linalg.norm(state["pose"][3:]))

    _, (twist, norm) = jax.lax.scan(advance, node.initial_state(), None, length=steps)
    return np.asarray(twist), np.asarray(norm)


def velocity_error(steps: int, *, rotation: bool = False) -> float:
    """The largest error of the implicit scheme against the closed form over ``HORIZON`` in ``steps`` steps,
    relative to the terminal value: translation, or rotation."""
    dt = HORIZON / steps
    twist, _ = twists(dt, steps)
    exact = terminal_approach(dt * np.arange(1, steps + 1))
    part = slice(3, 6) if rotation else slice(0, 3)
    scale = float(np.linalg.norm((TORQUE / ROTATIONAL) if rotation else (FORCE / LINEAR)))
    return float(np.max(np.linalg.norm(twist[:, part] - exact[:, part], axis=1))) / scale


def first_order(errors: Sequence[float]) -> list[float]:
    """The order each halving of ``dt`` shows: ``log2(e(dt) / e(dt / 2))``."""
    return [math.log2(a / b) for a, b in zip(errors, errors[1:], strict=False)]
