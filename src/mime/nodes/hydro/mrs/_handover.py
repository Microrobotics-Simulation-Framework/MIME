"""Lubrication's hand-over on the MRS rung (drag 0.5.0's ``lubrication``): the near-field model's completion in the
gap, keeping the MRS's near-wall accuracy at a coarse spacing, which a fine enough MRS would reach alone at far greater
cost; not physics beyond Stokes flow.

**The lift form.** The resistance is the coarse MRS evaluated once with the body's near contact opened to the hand-over
gap ``h_m``, plus the contact's thin-film increment:

    R = R_MRS(the wall opened by delta) + TF(h) - TF(h + delta),     delta = lift(h) - h,

``h`` the contact's gap, ``lift`` a smooth maximum of ``h`` and ``h_m`` (``mime.nodes.hydro._film.smooth_floor``) that
is ``h`` itself from ``h_m`` plus its width up, so that a body farther off is the coarse MRS's alone, step for step, and
costs nothing more. The contact is opened by moving its opposing surface, never the body: the wall's own opening
(``WallCorrection.opened``: the plane translated, the tube dilated). The MRS is never evaluated at a gap below ``h_m``.

**The hand-over gap** is :data:`HANDOVER_SPACINGS` of the force spacing (the finest graded spacing where the sampling
is graded), measured: at it the coarse MRS's worst entry on a sphere above a plane, the normal, read 5.6 to 6.1 % off
at blob 0.1, 3.5 to 4.2 % at 0.05 and 2.6 to 3.4 % at 0.025, with the image at the force points, which unlike the
integrated image does not depend on how the sampling faces the wall. The film's form holds within about 7 % while
``h_m`` is at most :data:`WINDOW` of the contact's smaller radius of curvature; beyond it the step raises
``handover_window`` (measured on a sphere and a spheroid: 15 to 19 % at a force spacing of 0.45 to 0.6 radii, about 30 %
at 0.9).

**The roughness cutoff.** Below the two surfaces' composite roughness (:func:`roughness_cutoff`) the film's gap is
floored, smoothly over :data:`ROUGHNESS_WIDTH` of it: the film's resistance stops growing without vanishing, and the
contact (W2b's traction) carries the gap. The roughness of each surface is a declared length, never fitted
(:data:`DEFAULT_ROUGHNESS` where none is declared).

**Who owns the coupling.** The near-wall resistance (the coarse MRS plus the film's increment) owns every hydrodynamic
coupling of rotation and translation at every gap down to the cutoff, and below it the film's coupling stays at its
floored value; a contact owns only the normal reaction, traction and rolling resistance and adds no hydrodynamic term.
The increment's own coupling block is reported (``film_coupling``), so that a double count can be tested for.

**This form covers** one point contact a body (its own deepest point against the wall, which a convex body against a
plane or a lumen has one of) whose surface is smooth there; a contact the film's point form does not cover is
reported, never silent: a lumen the body nearly fills, whose contact is a ring (``film_annulus``)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp, quaternion, shape_named
from mime.nodes.body._contact import START_APART, Hull, Pair, Wall, pair_contacts, spacing_of, start_count
from mime.nodes.hydro._film import FILM_NODES, about, film, principal, smooth_floor, tangent_frame
from mime.registry import Geometry

_lax: Any = cast(Any, jax).lax

#: The hand-over gap in force spacings (the finest graded spacing where graded): a labelled default, measured on a sphere
#: above a plane through the drag's node (W2a's first step; the module's docstring).
HANDOVER_SPACINGS = 0.75
#: The lift's width, a share of the hand-over gap: the lift is the identity from ``h_m`` plus it up and ``h_m`` from
#: ``h_m`` less it down.
LIFT_WIDTH = 0.5
#: The roughness floor's width, a share of the cutoff: a declared numerical setting.
ROUGHNESS_WIDTH = 0.5
#: Each surface's roughness where none is declared (m): a declaration, never fitted (sources span about 0.03 to a few
#: micrometres).
DEFAULT_ROUGHNESS = 1e-6
#: The hand-over gap over the contact's smaller radius of curvature beyond which the film's measured reach (about 7 %)
#: is left (``handover_window``).
WINDOW = 0.15
#: The film's elimination residual, in units of the dtype's resolution, above which a step raises ``film_unsolved``:
#: measured at about a thousand (1e-13 in float64, 1e-4 in float32, at gaps 1e-4 to 0.1 of the radius and aspects 1
#: to 20), so ten times over it.
FILM_RESIDUAL_ULPS = 1e4
#: The radius of curvature the support's derivative is regularised by, a share of the body's size: a sharp edge's
#: radius is then this small, and its film near zero.
RADIUS_FLOOR = 1e-6
#: A film reaching round the lumen: its patch's half-width along the gap's slower direction past this share of a
#: quarter turn of the wall there (``film_annulus``).
ANNULUS_SHARE = 1.0


def roughness_cutoff(*roughness: Any) -> Any:
    """Where lubrication hands the gap over to contact: the composite roughness of the surfaces, the root of the sum of
    their squares (m)."""
    return sum(r * r for r in roughness) ** 0.5


def handover_gap(spacing: float) -> float:
    """The hand-over gap at the local force ``spacing`` (m)."""
    return HANDOVER_SPACINGS * spacing


@dataclass(frozen=True)
class Search:
    """What a near contact's search holds from the build, in the hull kind's own frame: the groups of the force points
    each search starts from (one group), the starts a group takes and where (the convex hull's vertices against a
    convex region), the square of the least separation of two starts, how far the body reaches past its samples and
    its size; the part's placement in the body (``turn`` rows and ``centre``, and the turn as a quaternion); and the
    region's registered shape."""

    groups: np.ndarray
    places: np.ndarray
    starts: int
    apart_squared: float
    shortfall: float
    size: float
    turn: np.ndarray
    centre: np.ndarray
    turned: np.ndarray
    shape: str


def search_for(kind: Geometry, settings: Mapping[str, Any], points: np.ndarray, patches: Sequence[str],
               counts: Sequence[Sequence[int]], turn: Sequence[float], centre: Sequence[float], shape: str) -> Search:
    """The search's build-time data from the kind's force points in its own frame (``points``, float64)."""
    from scipy.spatial.transform import Rotation

    from mime.nodes.body._rigid_inertial import convex_candidates, point_groups
    from mime.scene._sampling import hull_shortfall

    groups = point_groups(points, 1)
    places = convex_candidates(points, groups) if getattr(shape_named(shape), "convex", False) else groups
    apart = START_APART * spacing_of(points)
    shortfall = hull_shortfall(kind, settings, tuple(patches), [tuple(int(c) for c in count) for count in counts])
    rows = np.asarray(turn, dtype=np.float64).reshape(3, 3)
    x, y, z, w = Rotation.from_matrix(rows).as_quat()
    return Search(np.asarray(groups, dtype=np.int32), np.asarray(places, dtype=np.int32),
                  start_count(points, places, apart, shortfall), apart ** 2, shortfall,
                  float(np.max(np.linalg.norm(points, axis=1))), rows, np.asarray(centre, dtype=np.float64),
                  np.asarray((w, x, y, z), dtype=np.float64), shape)


class NearContact(NamedTuple):
    """The body's near contact with the wall, in the world frame: its nearest point (the body's), the wall's normal
    into the medium, the gap (m), the contact's frame (rows: the gap's faster and slower directions, the normal), the
    gap's curvatures along them (1/m), and the wall's own curvature along the slower one (1/m)."""

    point: Array
    normal: Array
    gap: Array
    frame: Array
    k1: Array
    k2: Array
    wall_along: Array


def near_contact(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                 region: Mapping[str, Any], leaves: Mapping[str, Any]) -> NearContact:
    """The body's deepest point against the region's wall (``mime.nodes.body._contact``'s search, the margin zero):
    ``points`` the kind's force points in its own frame (traced, so that they move with its settings), ``parameters``
    the kind's settings, ``pose`` the body's, ``region`` the shape's parameters; ``leaves`` the search's held numbers
    (``film_search_*``). The gap's curvature is the body's (the inverse of its support's derivative in the direction,
    its radii of curvature, regularised by :data:`RADIUS_FLOOR`) less the wall's (its shape's)."""
    dtype = pose.dtype
    shape = cast(Any, shape_named(search.shape))  # a shape that gives its curvature (Shape.curvature)
    rotation = quaternion.rotation_matrix(pose[3:])
    kind_pose = jnp.concatenate([pose[:3] + rotation @ jnp.asarray(search.centre, dtype=dtype),
                                 quaternion.multiply(pose[3:], jnp.asarray(search.turned, dtype=dtype))])
    # one start reads no separation, and the step keeps no shortfall (the contact's report): neither is then a leaf
    apart = leaves["film_search_apart_squared"] if search.starts > 1 else jnp.zeros((), dtype=dtype)
    hull = Hull(points, search.groups, kind_pose, kind, parameters, search.places, apart, jnp.zeros((), dtype=dtype),
                leaves["film_search_tolerance"], search.starts)
    zero = jnp.zeros((), dtype=dtype)
    found = pair_contacts(Pair(hull, Wall(shape, region, zero), True), zero)
    point, normal, gap = found.points[0], found.rows[0, :3], found.gaps[0]
    turn = quaternion.rotation_matrix(kind_pose[3:])
    support = kind.support(parameters, turn.T @ -normal, found.local[0])
    radii = turn @ support.jacobian @ turn.T
    tangent = tangent_frame(normal)
    floor = RADIUS_FLOOR * leaves["film_reach"]
    body = jnp.linalg.inv(tangent @ radii @ tangent.T + floor * jnp.eye(2, dtype=dtype))
    wall = tangent @ shape.curvature(region, point) @ tangent.T
    frame, k1, k2 = principal(body - wall, tangent, normal)
    return NearContact(point, normal, gap, frame, k1, k2, frame[1] @ shape.curvature(region, point) @ frame[1])


class Increment(NamedTuple):
    """The film's increment about the body's centre of mass in the world frame, and the larger of its two solves'
    residuals."""

    resistance: Array
    residual: Array


def increment(contact: NearContact, delta: Array, cutoff: Array, handover: Array, reach: Array, viscosity: Array,
              centre: Array, nodes: int = FILM_NODES) -> Increment:
    """``TF(h) - TF(h + delta)`` at the contact, each gap floored at the roughness ``cutoff``, carried to the world
    frame about ``centre``. Its patch's cap is held above the widest gap it evaluates (the hand-over gap and its lift's
    width) so that both solves share it, and the film's O(1) part, which is not the body's, cancels between them."""
    dtype = contact.gap.dtype
    width = ROUGHNESS_WIDTH * cutoff
    k1 = jnp.maximum(contact.k1, 1.0 / reach)
    k2 = jnp.clip(contact.k2, 0.0, k1)
    cap_floor = 1.5 * (1.0 + LIFT_WIDTH) * handover
    near = film(smooth_floor(contact.gap, cutoff, width), k1, k2, reach, viscosity, cap_floor, nodes)
    far = film(smooth_floor(contact.gap + delta, cutoff, width), k1, k2, reach, viscosity, cap_floor, nodes)
    local = near.resistance - far.resistance
    return Increment(about(local, contact.frame, contact.point - centre).astype(dtype),
                     jnp.maximum(near.residual, far.residual))


class Handover(NamedTuple):
    """A step's hand-over: the wall's constants opened, the film's increment (zero where the contact is beyond the
    lift), the contact, the hand-over gap, the opening and the cutoff, the film's residual, and the step's flags as
    0 or 1: ``window`` (the hand-over gap past :data:`WINDOW` of the contact's smaller radius), ``annulus`` (the film's
    patch reaching round the lumen) and ``unsolved`` (the film not finite, or its residual past
    :data:`FILM_RESIDUAL_ULPS`)."""

    opened: dict[str, Any]
    increment: Array
    contact: NearContact
    handover: Array
    delta: Array
    cutoff: Array
    residual: Array
    window: Array
    annulus: Array
    unsolved: Array


def handed_over(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                wall: Any, constants: Mapping[str, Any], leaves: Mapping[str, Any], viscosity: Array,
                nodes: int = FILM_NODES) -> Handover:
    """The step's hand-over: the near contact against the wall at its constants, the wall opened by the lift, and the
    film's increment where the lift opened it (a ``cond``: where it did not, the increment is zero and its branch is
    not evaluated; the two agree there in value and derivative, the lift being the identity)."""
    dtype = pose.dtype
    region = wall.region(constants)[1]
    contact = near_contact(search, kind, parameters, points, pose, region, leaves)
    handover = HANDOVER_SPACINGS * jnp.asarray(leaves["handover_spacing"], dtype=dtype)
    delta = smooth_floor(contact.gap, handover, LIFT_WIDTH * handover) - contact.gap
    cutoff = roughness_cutoff(jnp.asarray(leaves["roughness_body"], dtype=dtype),
                              jnp.asarray(leaves["roughness_wall"], dtype=dtype))
    reach = jnp.asarray(leaves["film_reach"], dtype=dtype)

    def filmed() -> Increment:
        return increment(contact, delta, cutoff, handover, reach, viscosity, pose[:3], nodes)

    def none() -> Increment:
        return Increment(jnp.zeros((6, 6), dtype=dtype), jnp.zeros((), dtype=dtype))

    found = _lax.cond(delta > 0.0, filmed, none)
    finite = jnp.all(jnp.isfinite(found.resistance))
    unsolved = jnp.logical_not(finite) | (found.residual > FILM_RESIDUAL_ULPS * jnp.finfo(dtype).eps)
    k1 = jnp.maximum(contact.k1, 1.0 / reach)
    window = handover * k1 > WINDOW
    # the film's patch along the gap's slower direction, sqrt(2 cap / k2), past a quarter turn of the wall there
    annulus = 2.0 * (0.5 / k1) * contact.wall_along ** 2 > (ANNULUS_SHARE * 0.5 * math.pi) ** 2 * \
        jnp.maximum(contact.k2, 0.0)
    flags = _lax.stop_gradient(jnp.stack([window & (delta > 0.0), annulus & (delta > 0.0), unsolved]).astype(dtype))
    return Handover(wall.opened(constants, delta), found.resistance, contact, handover, delta, cutoff,
                    _lax.stop_gradient(found.residual), flags[0], flags[1], flags[2])


__all__ = ["ANNULUS_SHARE", "DEFAULT_ROUGHNESS", "FILM_RESIDUAL_ULPS", "HANDOVER_SPACINGS", "LIFT_WIDTH",
           "RADIUS_FLOOR", "ROUGHNESS_WIDTH", "WINDOW", "Handover", "Increment", "NearContact", "Search", "handed_over",
           "handover_gap", "increment", "near_contact", "roughness_cutoff", "search_for"]
