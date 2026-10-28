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
    switch,
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
#: A body counts as on its confining surface (less its extent and margin) within this many units in the last place of
#: its distance from the origin plus that inset: the projection leaves it there to its own rounding, a few units.
CONTACT_ULPS = 64
#: The velocity rules held inside the implicit solve while the body is on its surface.
HELD_IN_THE_SOLVE = frozenset({"remove_outward"})


class RigidInertialNode(MimeNode):
    """A rigid body: state ``pose = [x; q]`` and ``twist = [v; omega]``, in the world frame about the centre of mass.

    Each step, with ``M = diag(m I, R_q diag(inertia) R_q^T)``, the delivered resistance ``R`` and load
    ``F = applied_wrench + background_load + [0; -omega x (I omega)]`` (the gyroscopic term, explicit):

    ``(M/dt + R) dV = F - R V``, ``V' = V + dV``, then ``x' = x + dt v'`` and ``q' = exp(dt omega') q``, normalised.

    Implicit in the resistance, so a stiff resistance (``dt`` far above ``M/R``) relaxes rather than diverges;
    explicit in the loads; first order in time. Solved for the change in velocity, ``(M/dt + R) V' = (M/dt) V + F``
    rearranged: a load far below the inertial term ``(M/dt) V`` is never added to it before the solve, where float32
    would round it at that term's last place (MIME-ANO-113: for a body locked to a rotating field the rounding was the
    same at every step and stayed in the phase lag).

    **Confinement** (``surface_shape`` set): the body stays inside the shape's region shrunk by ``extent + margin``.

    - *In the solve.* Where the body starts the step on that shrunk surface (its signed distance within
      ``CONTACT_ULPS`` units in the last place of ``|x| + extent + margin`` of it, either side) and the solve's velocity ``V'`` moves it
      outward along the surface's normal ``n`` there faster than it closes its gap ``g`` (zero on the surface, the
      distance to it inside) in the step, the rule ``remove_outward`` is held inside the solve: with ``N = [n; 0]``
      and ``A = M/dt + R``, a multiplier along ``N`` (the surface's reaction, at the body's nearest point along
      ``n``, so it exerts no torque about the centre) solves ``A dV = F - R V - lambda N`` with ``N . V' = g / dt``:
      ``V'_held = V' - ((N . V' - g / dt) / N . A^-1 N) A^-1 N``, ``lambda >= 0``. A body a little inside its surface
      reaches it rather than being held where it is (in float32 the band is about 0.1 um wide). The resistance then resists
      only motion the body makes. Projected after the step instead, as 0.1.0 did, the solve resisted the outward
      velocity of a fall the projection then discarded, and the resistance's coupling turned it into a load the body
      never felt, an error first order in ``dt`` (2.35% in script A's speed at 0.5 ms). Without coupling in ``A``
      (``A^-1 N`` along ``N``) the held twist is ``remove_outward`` applied to ``V'``.
    - *The active set* is chosen per step from the start of the step: on the surface and moving outward. The
      held and free twists are equal where ``N . V' = 0``, so the twist is continuous there and its derivative is not
      (the normal direction drops out of it); across the contact band's edge, ``CONTACT_ULPS`` units from the surface,
      it is discontinuous, by what holding the contact changes. The derivative is that of the branch taken.
    - *After the step*, the position is projected onto the shrunk region and the velocity rule changes the velocity
      wherever it was: a body arriving at the surface, one that started the step beyond it (an initial state, or a
      step too long for the load), and the drift of a body moving along a curved one.
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

    def held(self, p: Mapping[str, Any], position: Array, system: Array, moved: Array, dt: float) -> Array:
        """``moved`` with the confinement held inside the solve of ``system = M/dt + R``: where the body starts on its
        shrunk surface and ``moved`` takes it outward faster than closing its gap ``g`` in the step, the velocity along
        the normal is held at ``g / dt`` through ``system`` (see the class); otherwise ``moved``."""
        own = self.constants()
        if not own["surface_shape"] or own["velocity_rule"] not in HELD_IN_THE_SOLVE:
            return moved
        shape = shape_named(cast(str, own["surface_shape"]))
        parameters = {name: p[f"{SURFACE}{name}"] for name in shape.parameter_names}
        inset = p["extent"] + p["margin"]
        beyond = shape.signed_distance(parameters, position) + inset
        scale = switch.norm(position, owner="contact.scale", resolution=0.0,
                            indicator=lambda: jnp.sqrt(jnp.sum(position * position)))
        resolution = CONTACT_ULPS * float(jnp.finfo(moved.dtype).eps) * (scale + inset)
        direction = jnp.concatenate([shape.outward_normal(parameters, position), jnp.zeros(3, dtype=moved.dtype)])
        response = jnp.linalg.solve(system, direction)
        gap = -beyond
        closing = switch.maximum(gap, 0.0, owner="contact.closing", resolution=resolution, indicator=gap) / dt
        outward = jnp.dot(direction, moved) - closing

        def speed() -> Array:
            return switch.rounding(jnp.sqrt(jnp.sum(moved * moved)) + closing)

        distance = switch.absolute(beyond, owner="contact.band", resolution=resolution, indicator=beyond)
        in_band = switch.predicate(distance <= resolution, owner="contact.band", resolution=resolution,
                                   indicator=beyond)
        moving_out = switch.predicate(outward > 0, owner="contact.outward", resolution=speed, indicator=outward)
        active = in_band & moving_out
        reach = jnp.dot(direction, response)
        multiplier = switch.where(active, outward / switch.domain_guard(active, reach, 1.0), 0.0,
                                  owner="contact.hold", resolution=speed, indicator=outward)
        return moved - multiplier * response

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

        def band() -> Array:
            scale = jnp.sqrt(jnp.sum(position * position)) + p["extent"] + p["margin"]
            return CONTACT_ULPS * float(jnp.finfo(position.dtype).eps) * scale

        projected = position - switch.maximum(beyond, 0.0, owner="contact.project", resolution=band,
                                              indicator=beyond) * normal
        ruled = VELOCITY_RULES[cast(str, own["velocity_rule"])](velocity, normal, owner="contact.rule_outward")
        return projected, switch.where(beyond > 0, ruled, velocity, owner="contact.rule", resolution=band,
                                       indicator=beyond)

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
        moved = self.held(p, position, mass_matrix / dt + resistance,
                          self.velocities(mass_matrix, resistance, twist, load, dt), dt)
        position, velocity = self._confined(p, position + dt * moved[:3], moved[:3])
        orientation = quaternion.normalized(quaternion.multiply(quaternion.exponential(dt * moved[3:]), orientation),
                                            owner="quaternion.norm")
        return node_fields(RigidInertialState(pose=jnp.concatenate([position, orientation]),
                                              twist=jnp.concatenate([velocity, moved[3:]])))
