"""A rigid body with inertia: Newton's and Euler's equations for its centre of mass and its rotation, with every
delivered resistance integrated implicitly, and an optional confinement projected after the step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, override

from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import DiscretizationOrder
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import (
    VELOCITY_RULES,
    MimeNode,
    NodeFields,
    as_run_array,
    jnp,
    node_fields,
    node_input,
    quaternion,
    shape_named,
    typed_view,
)


class RigidInertialState(TypedDict, closed=True):
    pose: ReadOnly[Float[Array, "7"]]
    twist: ReadOnly[Float[Array, "6"]]


class RigidInertialInputs(TypedDict, closed=True):
    applied_wrench: NotRequired[ReadOnly[Float[Array, "6"]]]
    resistance: NotRequired[ReadOnly[Float[Array, "6 6"]]]  # noqa: F722 - a jaxtyping shape
    background_load: NotRequired[ReadOnly[Float[Array, "6"]]]


#: The prefix of the constructor arguments that are the confining shape's parameters (``surface_radius``).
SURFACE = "surface_"


class RigidInertialNode(MimeNode):
    """A rigid body: state ``pose = [x; q]`` and ``twist = [v; omega]``, in the world frame about the centre of mass.

    Each step, with ``M = diag(m I, R_q diag(inertia) R_q^T)``, the delivered resistance ``R`` and load
    ``F = applied_wrench + background_load + [0; -omega x (I omega)]`` (the gyroscopic term, explicit):

    ``(M/dt + R) dV = F - R V``, ``V' = V + dV``, then ``x' = x + dt v'`` and ``q' = exp(dt omega') q``, normalised.

    Implicit in the resistance, so a stiff resistance (``dt`` far above ``M/R``) relaxes rather than diverges;
    explicit in the loads; first order in time. Solved for the change in velocity, ``(M/dt + R) V' = (M/dt) V + F``
    rearranged: a load far below the inertial term ``(M/dt) V`` is never added to it before the solve, where float32
    would round it at that term's last place (MIME-ANO-113: for a body locked to a rotating field the rounding was the
    same at every step and stayed in the phase lag). With ``surface_shape`` set, the position is then projected onto the
    shape's region shrunk by ``extent + margin``, and the velocity rule changes the velocity wherever it was.
    """

    def __init__(self, name: str, timestep: float, *, mass: float, inertia: Sequence[float],
                 initial_pose: Sequence[float], initial_twist: Sequence[float], surface_shape: str = "",
                 extent: float = 0.0, margin: float = 0.0, velocity_rule: str = "", **surface: object) -> None:
        if surface_shape:
            names = {f"{SURFACE}{p}" for p in shape_named(surface_shape).parameter_names}
            if set(surface) != names:
                raise TypeError(f"the shape {surface_shape!r} takes {sorted(names)}, not {sorted(surface)}")
            if velocity_rule not in VELOCITY_RULES:
                raise ValueError(f"velocity rule {velocity_rule!r} is not one of {sorted(VELOCITY_RULES)}")
        elif surface:
            raise TypeError(f"surface parameters {sorted(surface)} with no surface_shape")
        super().__init__(name, timestep, mass=mass, inertia=tuple(inertia), initial_pose=tuple(initial_pose),
                         initial_twist=tuple(initial_twist), surface_shape=surface_shape, extent=extent,
                         margin=margin, velocity_rule=velocity_rule, **surface)

    def discretization_order(self) -> DiscretizationOrder:
        return DiscretizationOrder(temporal=1.0)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "mass": ParamSpec(bounds=(0.0, None)),
                "inertia": ParamSpec(bounds=(0.0, None))}

    @override
    def initial_state(self) -> NodeFields:
        p = self.constants()
        return node_fields(RigidInertialState(pose=as_run_array(p["initial_pose"]),
                                              twist=as_run_array(p["initial_twist"])))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"applied_wrench": node_input("N; N m", (6,), additive=True),
                "resistance": node_input("N s/m; N s; N m s", (6, 6), additive=True),
                "background_load": node_input("N; N m", (6,), additive=True)}

    def velocities(self, mass_matrix: Array, resistance: Array, twist: Array, load: Array, dt: float) -> Array:
        """The twist after the step, ``V + dV`` with ``(M/dt + R) dV = F - R V``."""
        return twist + jnp.linalg.solve(mass_matrix / dt + resistance, load - resistance @ twist)

    def _confined(self, p: Mapping[str, Any], position: Array, velocity: Array) -> tuple[Array, Array]:
        """The position projected back onto the free side of the confining surface, less ``extent + margin``, and the
        velocity changed by the rule wherever it was projected; unchanged without a confinement."""
        own = self.constants()
        if not own["surface_shape"]:
            return position, velocity
        shape = shape_named(cast(str, own["surface_shape"]))
        parameters = {name: p[f"{SURFACE}{name}"] for name in shape.parameter_names}
        beyond = shape.signed_distance(parameters, position) + p["extent"] + p["margin"]
        normal = shape.outward_normal(parameters, position)
        projected = position - jnp.maximum(beyond, 0.0) * normal
        ruled = VELOCITY_RULES[cast(str, own["velocity_rule"])](velocity, normal)
        return projected, jnp.where(beyond > 0, ruled, velocity)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(RigidInertialState, state), typed_view(RigidInertialInputs, boundary_inputs)
        pose, twist = held["pose"], held["twist"]
        position, orientation, omega = pose[:3], pose[3:], twist[3:]
        rotation = quaternion.rotation_matrix(orientation)
        inertia = rotation @ jnp.diag(jnp.asarray(p["inertia"], dtype=twist.dtype)) @ rotation.T
        mass_matrix = jnp.zeros((6, 6), dtype=twist.dtype)
        mass_matrix = mass_matrix.at[:3, :3].set(p["mass"] * jnp.eye(3, dtype=twist.dtype)).at[3:, 3:].set(inertia)
        none = jnp.zeros_like(twist)
        load = (given.get("applied_wrench", none) + given.get("background_load", none)
                + jnp.concatenate([jnp.zeros_like(omega), -jnp.cross(omega, inertia @ omega)]))
        resistance = given.get("resistance", jnp.zeros((6, 6), dtype=twist.dtype))
        moved = self.velocities(mass_matrix, resistance, twist, load, dt)
        position, velocity = self._confined(p, position + dt * moved[:3], moved[:3])
        orientation = quaternion.normalized(quaternion.multiply(quaternion.exponential(dt * moved[3:]), orientation))
        return node_fields(RigidInertialState(pose=jnp.concatenate([position, orientation]),
                                              twist=jnp.concatenate([velocity, moved[3:]])))
