"""A rigid body with inertia: Newton's and Euler's equations for its centre of mass and its rotation, with every
delivered resistance integrated implicitly, and an optional confinement: its centre held by its extent, or its hull's
points held against the surface by impulses through the step's system and projected after the step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import jax
import numpy as np
from jax import Array
from numpy.typing import NDArray
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
from mime.nodes._kinds import kind_from_settings, registered_kind, setting_units, traced_settings
from mime.registry import Geometry


class RigidInertialState(TypedDict, closed=True):
    pose: ReadOnly[Float[Array, "7"]]
    twist: ReadOnly[Float[Array, "6"]]


class RigidContactState(TypedDict, closed=True):
    """A body held by its hull's points: its pose and twist, and each group's point the contact held in the step, in
    the world at the step's start, with the force it took there (N, zero where the point was free)."""

    pose: ReadOnly[Float[Array, "7"]]
    twist: ReadOnly[Float[Array, "6"]]
    contact_points: ReadOnly[Float[Array, "groups 3"]]
    contact_forces: ReadOnly[Float[Array, "groups"]]


class RigidInertialInputs(TypedDict, closed=True):
    applied_wrench: NotRequired[ReadOnly[Float[Array, "6"]]]
    resistance: NotRequired[ReadOnly[Float[Array, "6 6"]]]  # noqa: F722 - a jaxtyping shape
    background_load: NotRequired[ReadOnly[Float[Array, "6"]]]


#: The prefix of the constructor arguments that are the confining shape's parameters (``surface_radius``), and of
#: those that are the settings of the kind of the hull a contact holds (``hull_shape_radius``).
SURFACE, HULL = "surface_", "hull_shape_"
#: The rows of the identity rotation, a hull part's placement when it sits unturned in its body.
_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#: A body counts as on its confining surface (less its extent and margin) within this many units in the last place of
#: its distance from the origin plus that inset: the projection leaves it there to its own rounding, a few units.
CONTACT_ULPS = 64
#: The velocity rules held inside the implicit solve while the body is on its surface.
HELD_IN_THE_SOLVE = frozenset({"remove_outward"})
#: The groups a body's hull points are split into (``point_groups``): each step the contact holds each group's point
#: nearest the surface, so that a body touching in several places is held at each, whichever is deeper. A declaration:
#: eight keep a screw's two ends and a cylinder's two ends in groups of their own (``tests/oracles``); each costs the
#: sweeps one row.
HULL_GROUPS = 8
#: The projected Gauss-Seidel sweeps over the groups' points in the solve, as a dynamic chain's stops take (each
#: leaves the square of two points' rows' coupling of the change still to come: a rod 1 mm by 6 mm pressed into both
#: walls at its tilt limit, its levers short and its rows coupled at 0.89, takes 0.8% less than its couple; one 0.5 mm
#: by 8 mm, coupled at 0.1, its couple to rounding; phase 7o's pull request); and in each pass of the projection after
#: the step, which moves the pose by a step's drift and takes what the solve leaves. Declarations.
HULL_SWEEPS, PROJECTION_SWEEPS = 20, 10
#: The passes of the projection after the step, each the linearised projection at the pose the last one left: the
#: second takes what the first's linearisation leaves, the square of a step's turning.
HULL_PASSES = 2
#: Squared distances within this share of the hull's squared size are one tie when its points are grouped, broken by
#: the lower index, as the sampling's choice of samples breaks them (``mime.scene``'s ``TIE``).
GROUP_TIE = 1e-12


def point_groups(points: np.ndarray, count: int) -> np.ndarray:
    """``points`` (N x 3) split into ``min(count, N)`` groups, each a row of indices padded with ``-1``: each point to
    the nearest of the seeds that farthest-point selection chooses from the first point, squared distances read in
    whole units of ``GROUP_TIE`` times the points' squared size so that ties go to the lower index on any machine."""
    held = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def squared(point: np.ndarray) -> NDArray[np.float64]:
        return np.asarray(np.sum((held - point) ** 2, axis=1), dtype=np.float64)

    quantum = GROUP_TIE * max(float(np.max(squared(held.mean(axis=0)))), 1e-300)
    seeds = [0]
    nearest = squared(held[0])
    for _ in range(1, min(count, len(held))):
        seeds.append(int(np.argmax(np.floor(nearest / quantum))))
        nearest = np.minimum(nearest, squared(held[seeds[-1]]))
    away = np.floor(np.sum((held[:, None, :] - held[seeds][None, :, :]) ** 2, axis=2) / quantum)
    owner = np.argmin(away, axis=1)
    members = [held_by for held_by in (np.flatnonzero(owner == g) for g in range(len(seeds))) if len(held_by)]
    rows = np.full((len(members), max(len(m) for m in members)), -1, dtype=np.int32)
    for g, held_by in enumerate(members):
        rows[g, :len(held_by)] = held_by
    return rows


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

    **Hull contact** (``hull_points`` set, with the surface): the body's hull points (``N x 3`` in its frame) stay
    inside the shape's region shrunk by ``margin``, its orientation read: each point at or past the shrunk surface
    is held, not the centre, so a turned body touching at two points takes the couple that holds it. The points are
    split into groups (``point_groups``), and each step each group's point nearest the surface is a contact, with its gap ``g``
    (positive inside), the normal ``n`` into the region and its arm ``r`` from the centre of mass: the row ``J =
    [n; r x n]``.

    - *In the solve.* The twist is the one nearest the solve's ``V'`` in ``A``'s norm, ``A = M/dt + R``, among those
      that move no point towards the surface by more than it can close in the step: ``J V >= -max(g, 0) / dt``
      (a point past it may come no closer). Its forces ``lambda >= 0`` (N, over the step) solve ``V = V' + A^-1 J^T
      lambda`` with ``lambda (J V + max(g, 0) / dt) = 0``, by projected Gauss-Seidel over the groups' points,
      ``HULL_SWEEPS`` sweeps, each point's update exact for its own force with the others held: one point exact in
      one sweep, two coupled at ``c`` converging by ``c^2`` a sweep. Inelastic and
      frictionless: a point arriving at the surface ends the step on it, the force normal, and none pulls. Every
      update lowers the problem's dual, and the bound is never positive, so the kinetic energy in ``A``'s norm never
      rises at any sweep: the contact does no work on the body but to stop it.
    - *After the step*, the pose is projected back inside, ``HULL_PASSES`` passes: each moves the body by the least
      change of pose in ``M``'s norm that takes every group's point nearest the surface at that pose back to it
      (``J delta >= -g``, by ``PROJECTION_SWEEPS`` of the same sweeps), its turn applied as a rotation; the second
      takes what the first's linearisation, and its sweeps, left. The velocity is the solve's: the projection moves the pose by the drift of a step, the
      square of its turning, and the next step's hold takes the rest.
    - The state adds ``contact_points``, each group's point at the step's start, and ``contact_forces``, the force
      each took. Where no point is held or projected, the step is the free body's to the last bit.
    """

    def __init__(self, name: str, timestep: float, *, mass: float, inertia: Sequence[float],
                 initial_pose: Sequence[float], initial_twist: Sequence[float], surface_shape: str = "",
                 extent: float = 0.0, margin: float = 0.0, velocity_rule: str = "", hull_kind: str = "",
                 hull_kind_version: str = "", hull_patches: Sequence[str] = (),
                 hull_counts: Sequence[Sequence[int]] = (), hull_centre: Sequence[float] = (0.0, 0.0, 0.0),
                 hull_rotation: Sequence[float] = _IDENTITY, hull_groups: np.ndarray | None = None,
                 **constants: object) -> None:
        surface = {key: value for key, value in constants.items() if key.startswith(SURFACE)}
        settings = {key.removeprefix(HULL): value for key, value in constants.items() if key.startswith(HULL)}
        unknown = set(constants) - set(surface) - {f"{HULL}{name}" for name in settings}
        if unknown:
            raise TypeError(f"unknown arguments {sorted(unknown)}: a surface's parameters are {SURFACE}<name> and a "
                            f"hull kind's settings {HULL}<setting>")
        if surface_shape:
            shape = shape_named(surface_shape)
            names = {f"{SURFACE}{p}" for p in shape.parameter_names}
            if set(surface) != names:
                raise TypeError(f"the shape {surface_shape!r} takes {sorted(names)}, not {sorted(surface)}")
            if velocity_rule not in VELOCITY_RULES:
                raise ValueError(f"velocity rule {velocity_rule!r} is not one of {sorted(VELOCITY_RULES)}")
            if hull_kind and not getattr(shape, "takes_many", False):
                raise ValueError(f"the shape {surface_shape!r} takes one point at a time, and a hull contact holds "
                                 f"many (Shape.takes_many)")
            if hull_kind and velocity_rule not in HELD_IN_THE_SOLVE:
                raise ValueError(f"a hull contact holds its points by {sorted(HELD_IN_THE_SOLVE)}, not "
                                 f"{velocity_rule!r}")
        elif surface:
            raise TypeError(f"surface parameters {sorted(surface)} with no surface_shape")
        hull: dict[str, object] = {}
        if hull_kind:
            record = registered_kind(hull_kind, hull_kind_version)
            self._hull: Geometry = kind_from_settings(hull_kind, hull_kind_version, settings)
            self._hull_traced = traced_settings(record, settings)
            hull = {"hull_kind": hull_kind, "hull_kind_version": hull_kind_version, "hull_patches": tuple(hull_patches),
                    "hull_counts": tuple(tuple(int(c) for c in count) for count in hull_counts),
                    "hull_centre": tuple(float(c) for c in hull_centre),
                    "hull_rotation": tuple(float(c) for c in hull_rotation)}
            if len(hull_patches) != len(hull_counts) or not hull_patches:
                raise ValueError(f"hull patches {hull_patches!r} and counts {hull_counts!r} name one count per patch, "
                                 f"and one patch or more")
            held = point_groups(self._body_points(hull, settings), HULL_GROUPS) if hull_groups is None else hull_groups
            hull["hull_groups"] = np.asarray(held, dtype=np.int32)
        super().__init__(name, timestep, mass=mass, inertia=tuple(inertia), initial_pose=tuple(initial_pose),
                         initial_twist=tuple(initial_twist), surface_shape=surface_shape, extent=extent,
                         margin=margin, velocity_rule=velocity_rule, **hull, **surface,
                         **{f"{HULL}{name}": value for name, value in settings.items()})

    def _body_points(self, hull: Mapping[str, Any], settings: Mapping[str, object]) -> np.ndarray:
        """The hull's points in the body's frame at its settings as built, in float64: what its groups are made of."""
        with cast(Any, jax).enable_x64(True):
            parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                          for name, value in settings.items()}
            found = [self._hull.samples(parameters, patch, tuple(count))
                     for patch, count in zip(hull["hull_patches"], hull["hull_counts"], strict=True)]
            points = np.concatenate([np.asarray(f.points, dtype=np.float64).reshape(-1, 3) for f in found])
        turn = np.asarray(hull["hull_rotation"], dtype=np.float64).reshape(3, 3)
        return points @ turn.T + np.asarray(hull["hull_centre"], dtype=np.float64)

    @property
    def holds_hull(self) -> bool:
        """Whether the body is held by its hull's points: a confinement with a hull kind."""
        own = self.constants()
        return bool(own["surface_shape"]) and bool(own.get("hull_kind"))

    @property
    def groups(self) -> np.ndarray:
        """The hull's points by group (``point_groups``), each row padded with ``-1``."""
        return np.asarray(self.constants()["hull_groups"], dtype=np.int32)

    def hull_points(self, p: Mapping[str, Any], dtype: Any) -> Array:
        """The hull's points in the body's frame, from the kind's settings: each a ``params`` leaf where it is a number
        (so a study's derivative in it moves them), its directions and integers held as built."""
        own = self.constants()
        parameters = {name: p[f"{HULL}{name}"] if name in self._hull_traced else own[f"{HULL}{name}"]
                      for name in (key.removeprefix(HULL) for key in own if key.startswith(HULL))}
        found = [self._hull.samples(parameters, patch, count)
                 for patch, count in zip(own["hull_patches"], own["hull_counts"], strict=True)]
        points = jnp.concatenate([jnp.reshape(jnp.asarray(f.points, dtype=dtype), (-1, 3)) for f in found])
        turn = jnp.asarray(p["hull_rotation"], dtype=dtype).reshape(3, 3)
        return jnp.matmul(points, turn.T) + jnp.asarray(p["hull_centre"], dtype=dtype)

    def discretization_order(self) -> DiscretizationOrder:
        return DiscretizationOrder(temporal=1.0)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        held = {**super().param_specs(), "mass": ParamSpec(bounds=(0.0, None)),
                "inertia": ParamSpec(bounds=(0.0, None))}
        if not self.constants()["surface_shape"]:
            unread = ParamSpec(trainable=False, description="unconfined: no surface to keep the body from, so the "
                                                            "step reads no extent or margin")
            held.update(extent=unread, margin=unread)
        own = self.constants()
        if own.get("hull_kind"):
            record = registered_kind(own["hull_kind"], own["hull_kind_version"])
            placed = "held as built: where the hull's part sits in the body, its frame"
            held.update({f"{HULL}{name}": ParamSpec(units=setting_units(record, name)) if name in self._hull_traced
                         else ParamSpec(trainable=False, description="held as built: a direction of the kind is its "
                                                                     "part's frame")
                         for name in (key.removeprefix(HULL) for key in own if key.startswith(HULL))})
            held.update(hull_centre=ParamSpec(units="m", trainable=False, description=placed),
                        hull_rotation=ParamSpec(units="1", trainable=False, description=placed))
        if self.holds_hull:
            held["extent"] = ParamSpec(trainable=False, description="held by its hull's points, so the step reads no "
                                                                    "extent")
        return held

    @override
    def initial_state(self) -> NodeFields:
        p = self.constants()
        pose, twist = as_run_array(p["initial_pose"]), as_run_array(p["initial_twist"])
        if self.holds_hull:
            groups = len(self.groups)
            return node_fields(RigidContactState(pose=pose, twist=twist,
                                                 contact_points=as_run_array(np.zeros((groups, 3))),
                                                 contact_forces=as_run_array(np.zeros(groups))))
        return node_fields(RigidInertialState(pose=pose, twist=twist))

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

    def surface(self, p: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """The confining surface's shape and its parameters: the region the body is held inside, less the margin. The
        wall's distance and normal at a point are the region's own, negated, as a wall kind gives them
        (``mime.registry.WallKind.distance``)."""
        shape = shape_named(cast(str, self.constants()["surface_shape"]))
        return shape, {name: p[f"{SURFACE}{name}"] for name in shape.parameter_names}

    def gaps(self, p: Mapping[str, Any], points: Array) -> Array:
        """Each point's gap to the confining surface shrunk by the margin (m, positive inside)."""
        shape, parameters = self.surface(p)
        return -shape.signed_distance(parameters, points) - p["margin"]

    def normals(self, p: Mapping[str, Any], points: Array) -> Array:
        """The surface's normal into the region at each point."""
        shape, parameters = self.surface(p)
        return -shape.outward_normal(parameters, points)

    def nearest_points(self, p: Mapping[str, Any], body: Array, pose: Array) -> Array:
        """Each group's point of the hull (``body``, its points in the body's frame) nearest the shrunk surface at
        ``pose``, by its index."""
        gaps = self.gaps(p, pose[:3] + jnp.matmul(body, quaternion.rotation_matrix(pose[3:]).T))
        rows = np.maximum(self.groups, 0)
        grouped = switch.static_select(self.groups >= 0, gaps[rows], jnp.inf)

        def runner_up() -> Array:
            ordered = jnp.sort(grouped, axis=-1)
            return ordered[..., 1] - ordered[..., 0] if ordered.shape[-1] > 1 else jnp.full(ordered.shape[:-1], jnp.inf)

        index = switch.argmin(grouped, owner="contact.point", resolution=lambda: switch.rounding(p["margin"]),
                              indicator=runner_up)
        return jnp.asarray(rows)[jnp.arange(len(rows)), index]

    def contacts(self, p: Mapping[str, Any], body: Array, chosen: Array, pose: Array) -> tuple[Array, Array, Array]:
        """The ``chosen`` points of the hull at ``pose``: each one's gap, its row ``J = [n; r x n]`` (the normal into
        the region and its moment about the centre of mass), and where it is in the world."""
        arms = jnp.matmul(body[chosen], quaternion.rotation_matrix(pose[3:]).T)
        points = pose[:3] + arms
        normals = self.normals(p, points)
        return self.gaps(p, points), jnp.concatenate([normals, jnp.cross(arms, normals)], axis=-1), points

    @staticmethod
    def sweeps(coupling: Array, start: Array, count: int, *, owner: str) -> Array:
        """The multipliers ``x >= 0`` with ``start + coupling x >= 0`` and ``x (start + coupling x) = 0``, by ``count``
        projected Gauss-Seidel sweeps from zero, each update exact for its own multiplier with the others held; a
        row of no length (a point on the surface's axis, where it has no normal) is never moved."""
        own = jnp.diagonal(coupling)
        held = switch.domain_guard(own > 0, own, 1.0)
        found = jnp.zeros_like(start)
        for _ in range(count):
            for k in range(start.shape[0]):
                residual = start[k] + jnp.dot(coupling[k], found)
                moved = found[k] - residual / held[k]
                found = found.at[k].set(switch.maximum(moved, 0.0, owner=owner, indicator=moved, resolution=lambda:
                                                       switch.rounding(jnp.abs(found[k]) + jnp.abs(residual / held[k]))))
        return found

    def hull_held(self, p: Mapping[str, Any], body: Array, pose: Array, system: Array, moved: Array,
                  dt: float) -> tuple[Array, Array, Array]:
        """``moved`` with the hull's points held in the solve of ``system = M/dt + R`` (see the class), each group's
        point nearest the surface at the step's start in the world, and the force each took."""
        gaps, rows, points = self.contacts(p, body, self.nearest_points(p, body, pose), pose)
        closing = switch.maximum(gaps, 0.0, owner="contact.gap", indicator=gaps,
                                 resolution=lambda: switch.rounding(jnp.abs(gaps) + p["margin"]))
        response = jnp.linalg.solve(system, rows.T)
        forces = self.sweeps(rows @ response, rows @ moved + closing / dt, HULL_SWEEPS, owner="contact.impulse")
        return moved + response @ forces, points, forces

    def hull_projected(self, p: Mapping[str, Any], body: Array, pose: Array, mass_matrix: Array) -> Array:
        """``pose`` moved back inside the shrunk surface by ``HULL_PASSES`` passes of the least change of pose in
        ``M``'s norm that takes each group's point nearest it, chosen at the pose the pass starts from, back to it
        (see the class)."""
        for _ in range(HULL_PASSES):
            gaps, rows, _ = self.contacts(p, body, self.nearest_points(p, body, pose), pose)
            response = jnp.linalg.solve(mass_matrix, rows.T)
            shift = response @ self.sweeps(rows @ response, gaps, PROJECTION_SWEEPS, owner="contact.push")
            pose = jnp.concatenate([pose[:3] + shift[:3], quaternion.multiply(quaternion.exponential(shift[3:]),
                                                                              pose[3:])])
        return pose

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
        if self.holds_hull:
            body = self.hull_points(p, twist.dtype)
            moved, points, forces = self.hull_held(p, body, pose, mass_matrix / dt + resistance,
                                                   self.velocities(mass_matrix, resistance, twist, load, dt), dt)
            orientation = quaternion.normalized(quaternion.multiply(quaternion.exponential(dt * moved[3:]),
                                                                    orientation))
            ended = self.hull_projected(p, body, jnp.concatenate([position + dt * moved[:3], orientation]),
                                        mass_matrix)
            return node_fields(RigidContactState(pose=ended, twist=moved, contact_points=points,
                                                 contact_forces=forces))
        moved = self.held(p, position, mass_matrix / dt + resistance,
                          self.velocities(mass_matrix, resistance, twist, load, dt), dt)
        position, velocity = self._confined(p, position + dt * moved[:3], moved[:3])
        orientation = quaternion.normalized(quaternion.multiply(quaternion.exponential(dt * moved[3:]), orientation))
        return node_fields(RigidInertialState(pose=jnp.concatenate([position, orientation]),
                                              twist=jnp.concatenate([velocity, moved[3:]])))
