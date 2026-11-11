"""Lubrication's hand-over on the MRS rung (drag 0.5.0's ``near_contact``): the MRS's side of the near-contact seam
(``mime.nodes.hydro.near_contact``), the near-field model's completion in the gap, keeping the MRS's near-wall accuracy
at a coarse spacing, which a fine enough MRS would reach alone at far greater cost; not physics beyond Stokes flow.

**The lift form.** The resistance is the coarse MRS evaluated once with the body's near contact opened to the hand-over
gap ``h_m`` (the gap the rung declares it resolves down to), plus the near-contact law's increment:

    R = R_MRS(the wall opened by delta) + A(h) - A(h + delta),     delta = opening(h, h_m),

``h`` the contact's gap, the opening a smooth floor of ``h`` at ``h_m`` that is zero from ``h_m`` plus its width up,
so that a body farther off is the coarse MRS's alone, step for step, and costs nothing more. The contact is opened by
moving its opposing surface, never the body: the wall's own opening (``WallCorrection.opened``: the plane translated,
the tube dilated). The MRS is never evaluated at a gap below ``h_m``. The law's form is the plan's, by the body's kind:
a sphere's exact increment, or the thin film's.

**The hand-over gap** is :data:`HANDOVER_SPACINGS` of the force spacing (the finest graded spacing where the sampling
is graded), measured: at it the coarse MRS's worst entry on a sphere above a plane, the normal, read 5.6 to 6.1 % off
at blob 0.1, 3.5 to 4.2 % at 0.05 and 2.6 to 3.4 % at 0.025, with the image at the force points, which unlike the
integrated image does not depend on how the sampling faces the wall. The film's form holds within about 7 % while
``h_m`` is at most :data:`WINDOW` of the contact's smaller radius of curvature; beyond it the step raises
``handover_window`` (measured on a sphere and a spheroid: 15 to 19 % at a force spacing of 0.45 to 0.6 radii, about 30 %
at 0.9). The sphere's form is exact at every gap, and has no window.

**The roughness cutoff** is the seam's (``roughness_cutoff``): the roughness of each surface is a declared length,
never fitted (:data:`DEFAULT_ROUGHNESS` where none is declared).

**Who owns the coupling.** The near-wall resistance (the coarse MRS plus the law's increment) owns every hydrodynamic
coupling of rotation and translation at every gap down to the cutoff, and below it the law's coupling stays at its
floored value; a contact owns only the normal reaction, traction and rolling resistance and adds no hydrodynamic term.
The increment's own coupling block is reported (``contact_coupling``), so that a double count can be tested for.

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
from mime.nodes.hydro._film import FILM_NODES, principal, tangent_frame
from mime.nodes.hydro.near_contact import (
    CURVATURE_FLOOR,
    LIFT_WIDTH,
    ROUGHNESS_WIDTH,
    Increment,
    NearContact,
    increment,
    opening,
    roughness_cutoff,
)
from mime.registry import Geometry

_lax: Any = cast(Any, jax).lax

#: The hand-over gap in force spacings (the finest graded spacing where graded): a labelled default, measured on a sphere
#: above a plane through the drag's node (W2a's first step; the module's docstring).
HANDOVER_SPACINGS = 0.75
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
#: The share of a lumen's diameter the body's width across it (along the contact's normal) may fill before the opening
#: by dilation moves the far side past the gates' line (``handover_far_side``): dilating the tube to open the near
#: contact opens the far side as well, and the coarse MRS's worst entry (the axial) at the opened geometry, against the
#: near contact opened with the far side kept, read -6.3 % at a/R 0.5 and -8.9 % at 0.55 for a sphere at the deepest
#: opening the window allows (s/a 0.2, h_m 0.15 a), -18 % at 0.67 and -37 % at 0.8 [measured: W2a's fix round].
FAR_SIDE_FILL = 0.5
#: The largest sphere-to-tube radius ratio the sphere's form in a tube was gated at (0.22 to 0.34, against an independent
#: tube MFS): above it a step within the opening's band raises ``handover_tube_ratio``. Along the axis the rolling
#: misses by -12 % at 0.34 and was measured -18 % at 0.40 and -25 % at 0.46 (the PR's review, its tube MFS resolved).
TESTED_TUBE_RATIO = 0.34
#: The share past TESTED_TUBE_RATIO a ratio may read and still be the gated one: the sphere's radius is read from its
#: curvature at the contact, regularised by RADIUS_FLOOR of its size, so a sphere at exactly 0.34 reads 1e-6 more.
RATIO_ROUNDING = 1e-4
#: What the sphere's form is measured to do, by where it runs (a 1 mm sphere, blob 0.05, float64, through the node;
#: tests/oracles/test_near_contact.py and the fix round's probes): at a uniform force spacing above a plane, in a tube
#: by the sphere-to-tube ratio, and on a sampling graded towards the contact (refused, drag.near-contact-graded).
UNIFORM_SPHERE_TEXT = ("on a sphere above a plane at uniform force spacings of 0.15 to 0.6 radii, the normal +2.1 to "
                       "+3.6 % off Brenner's series at and below the hand-over gap, every other entry within 1.8 % and "
                       "the rolling rate within 1.4 % of exact")
TUBE_RATIOS_TEXT = ("against an independent tube MFS at sphere-to-tube ratios of 0.22, 0.28 and 0.34, at gaps of 0.1 to "
                    "0.0075 radii: the rolling round the circumference within 3.3 % of the plane's rate, every diagonal "
                    "entry within 4.7 % and the couplings within 10 % of the plane's coupling (along the axis +16.5 % "
                    "at 0.34); the rolling along the axis misses by -3.6, -7.0 and -11.9 % of the plane's rate at the "
                    "family's spacing and by -2.7, -5.2 and -8.9 % at a spacing of 0.2 radii, a stated limit, which a "
                    "finer spacing lowers; not gated above 0.34, where the same method read -18 % at 0.40 and -25 % at "
                    "0.46")
GRADED_SPHERE_TEXT = ("graded towards its contact at a force spacing of 0.2 radii, a sphere's normal read +9.5 % and "
                      "+19.7 % off Brenner's series at hand-over gaps of 0.075 and 0.0375 radii, against +2.1 to +3.6 % "
                      "at uniform spacings of 0.15 to 0.6 radii")


def handover_gap(spacing: float) -> float:
    """The hand-over gap at the local force ``spacing`` (m)."""
    return HANDOVER_SPACINGS * spacing


def least_radius(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str]) -> float:
    """The body's least radius of curvature over its surface (m), as the step reads a contact's: the smaller of the two
    radii the support's derivative gives along each sample's outward normal (``kind.support``), regularised by
    :data:`RADIUS_FLOOR` of its size, the least over the kind's samples at ``spacing``; near zero at a sharp edge. At
    the build, in float64: the radius the hand-over's window is judged against (:data:`WINDOW`), for any contact the
    body can make whatever its pose."""
    key = (kind.name, kind.version, repr(sorted(settings.items())), float(spacing), tuple(patches))
    found = _least_radii.get(key)
    if found is None:
        found = _least_radius(kind, settings, spacing, patches)
        _least_radii[key] = found
    return found


_least_radii: dict[tuple[Any, ...], float] = {}


def _least_radius(kind: Geometry, settings: Mapping[str, Any], spacing: float, patches: Sequence[str]) -> float:
    with cast(Any, jax).enable_x64(True):
        sampling = kind.sampling(settings, spacing, tuple(patches))
        parameters = {name: (jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple)) else value)
                      for name, value in settings.items()}
        samples = sampling.samples(kind, parameters)
        points = jnp.asarray(samples.points, dtype=jnp.float64)
        normals = jnp.asarray(samples.normals, dtype=jnp.float64)
        jacobian = jnp.asarray(kind.support(parameters, normals, points).jacobian, dtype=jnp.float64)
        across = jax.vmap(tangent_frame)(normals)
        radii = jnp.linalg.eigvalsh(across @ (0.5 * (jacobian + jnp.swapaxes(jacobian, -1, -2)))
                                    @ jnp.swapaxes(across, -1, -2))
        size = float(jnp.max(jnp.sqrt(jnp.sum(points * points, axis=-1))))
        return max(float(jnp.min(radii)), 0.0) + RADIUS_FLOOR * size


def window_text(handover: float, radius: float) -> str:
    """Where the hand-over gap ``handover`` lies against the film's window over the least ``radius`` of curvature."""
    return (f"the hand-over gap {handover * 1e3:.4g} mm is {handover / radius:.4g} of the body's least radius of "
            f"curvature, {radius * 1e3:.4g} mm")


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


class Contacted(NamedTuple):
    """The body's near contact with the wall (the seam's :class:`NearContact`, in the world frame), with what the MRS's
    own flags read of it: the wall's curvature along the contact's slower direction (1/m) and the body's width along
    the contact's normal (m)."""

    contact: NearContact
    wall_along: Array
    width: Array


def near_contact(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                 region: Mapping[str, Any], leaves: Mapping[str, Any], reach: Array) -> Contacted:
    """The body's deepest point against the region's wall (``mime.nodes.body._contact``'s search, the margin zero):
    ``points`` the kind's force points in its own frame (traced, so that they move with its settings), ``parameters``
    the kind's settings, ``pose`` the body's, ``region`` the shape's parameters; ``leaves`` the search's held numbers
    (``film_search_*``); ``reach`` the body's size (traced). The body's curvature is the inverse of its support's
    derivative in the direction (its radii of curvature, regularised by :data:`RADIUS_FLOOR` of its size); the gap's is
    the body's less the wall's (its shape's: :func:`relative_curvature`), and the contact's frame the gap's principal
    directions, along which the body's own curvatures are read too."""
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
    # the search's Newton steps unrolled (the contact 0.4.0's program), so that every guard it reaches is evaluated
    # on values where the switch harness reads it
    found = pair_contacts(Pair(hull, Wall(shape, region, zero), False), zero)
    point, normal, gap = found.points[0], found.rows[0, :3], found.gaps[0]
    turn = quaternion.rotation_matrix(kind_pose[3:])
    support = kind.support(parameters, turn.T @ -normal, found.local[0])
    radii = turn @ support.jacobian @ turn.T
    tangent = tangent_frame(normal)
    floor = RADIUS_FLOOR * reach
    body = jnp.linalg.inv(tangent @ radii @ tangent.T + floor * jnp.eye(2, dtype=dtype))
    wall = tangent @ shape.curvature(region, point) @ tangent.T
    frame, k1, k2 = principal(relative_curvature(body, wall), tangent, normal)
    own = tangent.T @ body @ tangent
    across = turn.T @ normal
    width = kind.reach(parameters, across) + kind.reach(parameters, -across)
    return Contacted(NearContact(point, frame, gap, k1, k2, frame[0] @ own @ frame[0], frame[1] @ own @ frame[1]),
                     frame[1] @ shape.curvature(region, point) @ frame[1], width)


def relative_curvature(body: Array, wall: Array) -> Array:
    """The gap's curvature: the body's less the wall's, each ``2 x 2`` in one tangent frame, the wall's positive where
    it curves away from the medium (a tube's lumen curves towards the body, so its film is wider than a plane's). A
    module attribute so that a control can plant the wrong sign."""
    return body - wall


class Handover(NamedTuple):
    """A step's hand-over: the wall's constants opened, the law's increment (zero where the contact is beyond the
    lift), the contact, the hand-over gap, the opening and the cutoff, the law's residual (the film's; zero for a
    closed form), and the step's flags as 0 or 1: ``window`` (the film's hand-over gap past :data:`WINDOW` of the
    contact's smaller radius), ``annulus`` (the film's patch reaching round the lumen), ``unsolved`` (the film not
    finite, or its residual past :data:`FILM_RESIDUAL_ULPS`), ``far_side`` (a lumen opened by dilation about a body
    filling more than :data:`FAR_SIDE_FILL` of its diameter) and ``tube_ratio`` (the sphere's form in a tube past
    :data:`TESTED_TUBE_RATIO`)."""

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
    far_side: Array
    tube_ratio: Array


def handed_over(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                wall: Any, constants: Mapping[str, Any], leaves: Mapping[str, Any], viscosity: Array,
                form: str = "film", nodes: int = FILM_NODES) -> Handover:
    """The step's hand-over: the near contact against the wall at its constants, the wall opened to the hand-over gap,
    and the near-contact law's increment by ``form`` where the contact was opened (a ``cond``: where it was not, the
    increment is zero and its branch is not evaluated; the two agree there in value and derivative, the opening being
    zero). The film's flags (``window``, ``annulus``, ``unsolved``) are the film form's; the sphere's form has no
    window and no solve."""
    dtype = pose.dtype
    region = wall.region(constants)[1]
    reach = jnp.max(jnp.sqrt(jnp.sum(points * points, axis=-1)))
    found_contact = near_contact(search, kind, parameters, points, pose, region, leaves, reach)
    contact = found_contact.contact
    handover = HANDOVER_SPACINGS * jnp.asarray(leaves["handover_spacing"], dtype=dtype)
    delta = opening(contact.gap, handover)
    cutoff = roughness_cutoff(jnp.asarray(leaves["roughness_body"], dtype=dtype),
                              jnp.asarray(leaves["roughness_wall"], dtype=dtype))

    def opened() -> Increment:
        return increment(form, contact, delta, cutoff, handover, viscosity, pose[:3], reach, nodes)

    def none() -> Increment:
        return Increment(jnp.zeros((6, 6), dtype=dtype), jnp.zeros((), dtype=dtype))

    found = _lax.cond(delta > 0.0, opened, none)
    lumen = region.get("radius")
    far_side = jnp.asarray(False) if lumen is None else found_contact.width > FAR_SIDE_FILL * 2.0 * lumen
    # the sphere's radius is its own curvature at the contact, as its form reads it
    tube_ratio = jnp.asarray(False) if lumen is None or form != "sphere" else \
        2.0 / (contact.b1 + contact.b2) > TESTED_TUBE_RATIO * (1.0 + RATIO_ROUNDING) * lumen
    if form == "film":
        finite = jnp.all(jnp.isfinite(found.resistance))
        unsolved = jnp.logical_not(finite) | (found.residual > FILM_RESIDUAL_ULPS * jnp.finfo(dtype).eps)
        k1 = jnp.maximum(contact.k1, CURVATURE_FLOOR / reach)
        window = handover * k1 > WINDOW
        # the film's patch along the gap's slower direction, sqrt(2 cap / k2), past a quarter turn of the wall there
        annulus = 2.0 * (0.5 / k1) * found_contact.wall_along ** 2 > (ANNULUS_SHARE * 0.5 * math.pi) ** 2 * \
            jnp.maximum(contact.k2, 0.0)
    else:
        unsolved = window = annulus = jnp.asarray(False)
    flags = _lax.stop_gradient(jnp.stack([window & (delta > 0.0), annulus & (delta > 0.0), unsolved,
                                          far_side & (delta > 0.0), tube_ratio & (delta > 0.0)]).astype(dtype))
    return Handover(wall.opened(constants, delta), found.resistance, contact, handover, delta, cutoff,
                    _lax.stop_gradient(found.residual), flags[0], flags[1], flags[2], flags[3], flags[4])


__all__ = ["ANNULUS_SHARE", "CURVATURE_FLOOR", "FAR_SIDE_FILL", "DEFAULT_ROUGHNESS", "FILM_RESIDUAL_ULPS", "HANDOVER_SPACINGS",
           "GRADED_SPHERE_TEXT", "LIFT_WIDTH", "RADIUS_FLOOR", "RATIO_ROUNDING", "ROUGHNESS_WIDTH", "TESTED_TUBE_RATIO",
           "TUBE_RATIOS_TEXT", "UNIFORM_SPHERE_TEXT", "WINDOW", "Contacted", "Handover", "Increment", "NearContact",
           "Search", "handed_over", "handover_gap", "increment", "least_radius", "near_contact", "opening",
           "relative_curvature", "roughness_cutoff", "search_for", "window_text"]
