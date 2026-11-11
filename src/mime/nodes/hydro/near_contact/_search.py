"""A body's near contacts with a wall, found for the near-contact law: the seam's own search, the same for every rung
(MIME-ANO-154's review's m5: it was the MRS's). The body's hull is split into groups of its points (``slots``: one for
a body whose single deepest point is its only contact, a sphere's; :data:`GROUPS` for any other, as the rigid body's
contact splits it), and each group's deepest point against the wall is searched for by the contact's own search
(``mime.nodes.body._contact``), the margin zero. At each point the body's curvature is the inverse of its support's
derivative there (its radii of curvature, regularised by :data:`RADIUS_FLOOR` of its size); the gap's is the body's
less the wall's (:func:`relative_curvature`), and the contact's frame the gap's principal directions, along which the
body's own curvature is read too (its cross term ``b12`` included)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from jax import Array

from mime.core import jnp, quaternion, shape_named
from mime.nodes.body._contact import START_APART, Hull, Pair, Wall, pair_contacts, spacing_of, start_count
from mime.nodes.hydro.near_contact._seam import (
    LIFT_WIDTH,
    NearContact,
    NearContacts,
    classified,
    merged,
    principal,
    tangent_frame,
)
from mime.registry import Geometry

_lax: Any = cast(Any, jax).lax
_tree_util: Any = cast(Any, jax).tree_util
_vmap: Any = cast(Any, jax).vmap

#: The groups a body's hull is split into for its near contacts where it may touch a wall at more than one point (any
#: kind but the sphere): the rigid body's contact's own count (``HULL_GROUPS``), which keeps a screw's crests and its
#: two ends in groups of their own.
GROUPS = 8
#: The radius of curvature the support's derivative is regularised by, a share of the body's size: a sharp edge's
#: radius is then this small, and the law's window past it.
RADIUS_FLOOR = 1e-6


@dataclass(frozen=True)
class Search:
    """What a body's near-contact search holds from the build, in the hull kind's own frame: the groups of the force
    points each search starts from (one row a slot), the starts a group takes and where (the convex hull's vertices
    against a convex region), the square of the least separation of two starts, how far the body reaches past its
    samples and its size; the part's placement in the body (``turn`` rows and ``centre``, and the turn as a quaternion);
    and the region's registered shape."""

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

    @property
    def slots(self) -> int:
        """The contacts the search can find, one a group."""
        return int(self.groups.shape[0])


def search_for(kind: Geometry, settings: Mapping[str, Any], points: np.ndarray, patches: Sequence[str],
               counts: Sequence[Sequence[int]], turn: Sequence[float], centre: Sequence[float], shape: str,
               slots: int = 1) -> Search:
    """The search's build-time data from the kind's force points in its own frame (``points``, float64), its hull split
    into ``slots`` groups."""
    from scipy.spatial.transform import Rotation

    from mime.nodes.body._rigid_inertial import convex_candidates, point_groups
    from mime.scene._sampling import hull_shortfall

    groups = point_groups(points, slots)
    places = convex_candidates(points, groups) if getattr(shape_named(shape), "convex", False) else groups
    apart = START_APART * spacing_of(points)
    shortfall = hull_shortfall(kind, settings, tuple(patches), [tuple(int(c) for c in count) for count in counts])
    rows = np.asarray(turn, dtype=np.float64).reshape(3, 3)
    x, y, z, w = Rotation.from_matrix(rows).as_quat()
    return Search(np.asarray(groups, dtype=np.int32), np.asarray(places, dtype=np.int32),
                  start_count(points, places, apart, shortfall), apart ** 2, shortfall,
                  float(np.max(np.linalg.norm(points, axis=1))), rows, np.asarray(centre, dtype=np.float64),
                  np.asarray((w, x, y, z), dtype=np.float64), shape)


class Found(NamedTuple):
    """A body's near contacts (the seam's :class:`NearContacts`, in the world frame, each slot classed at the resolved
    gap) with the body's width along each contact's normal (m), which a rung's flags read."""

    contacts: NearContacts
    width: Array


def near_contacts(search: Search, kind: Geometry, parameters: Mapping[str, Any], points: Array, pose: Array,
                  region: Mapping[str, Any], leaves: Mapping[str, Any], reach: Array, resolved: Array,
                  spacing: Array) -> Found:
    """The body's deepest point against the region's wall in each of the search's groups: ``points`` the kind's force
    points in its own frame (traced, so that they move with its settings), ``parameters`` the kind's settings, ``pose``
    the body's, ``region`` the shape's parameters; ``leaves`` the search's held numbers (``contact_search_*``); ``reach``
    the body's size (traced); ``resolved`` the rung's resolved gap and ``spacing`` the points' spacing, at which each
    contact is classed. The slots found at one point share it (``merged``)."""
    dtype = pose.dtype
    shape = cast(Any, shape_named(search.shape))  # a shape that gives its curvature (Shape.curvature)
    rotation = quaternion.rotation_matrix(pose[3:])
    kind_pose = jnp.concatenate([pose[:3] + rotation @ jnp.asarray(search.centre, dtype=dtype),
                                 quaternion.multiply(pose[3:], jnp.asarray(search.turned, dtype=dtype))])
    # one start reads no separation, and the step keeps no shortfall (the contact's report): neither is then a leaf
    apart = leaves["contact_search_apart_squared"] if search.starts > 1 else jnp.zeros((), dtype=dtype)
    hull = Hull(points, search.groups, kind_pose, kind, parameters, search.places, apart, jnp.zeros((), dtype=dtype),
                leaves["contact_search_tolerance"], search.starts)
    zero = jnp.zeros((), dtype=dtype)
    # the search's Newton steps unrolled (the contact 0.4.0's program), so that every guard it reaches is evaluated
    # on values where the switch harness reads it
    found = pair_contacts(Pair(hull, Wall(shape, region, zero), False), zero)
    turn = quaternion.rotation_matrix(kind_pose[3:])
    floor = RADIUS_FLOOR * reach
    band = (1.0 + LIFT_WIDTH) * resolved
    world = _lax.stop_gradient(kind_pose[:3] + points @ turn.T)

    def one(point: Array, normal: Array, gap: Array, local: Array) -> tuple[NearContact, Array, Array, Array]:
        support = kind.support(parameters, turn.T @ -normal, local)
        radii = turn @ support.jacobian @ turn.T
        tangent = tangent_frame(normal)
        body = jnp.linalg.inv(tangent @ radii @ tangent.T + floor * jnp.eye(2, dtype=dtype))
        wall = tangent @ shape.curvature(region, point) @ tangent.T
        frame, k1, k2 = principal(relative_curvature(body, wall), tangent, normal)
        own = tangent.T @ body @ tangent
        across = turn.T @ normal
        width = kind.reach(parameters, across) + kind.reach(parameters, -across)
        # how far the body's points within the band of its tangent plane at the contact reach from it: a report
        held = _lax.stop_gradient(point)
        within = (world - held) @ _lax.stop_gradient(normal) <= band
        footprint = jnp.max(jnp.where(within, jnp.sqrt(jnp.sum((world - held) ** 2, axis=-1)), 0.0))
        contact = NearContact(point, frame, gap, k1, k2, frame[0] @ own @ frame[0], frame[1] @ own @ frame[1],
                              frame[0] @ own @ frame[1])
        return contact, frame[1] @ shape.curvature(region, point) @ frame[1], width, footprint

    rows = (found.points, found.rows[:, :3], found.gaps, found.local)
    if search.slots == 1:
        contact, wall_along, width, footprint = one(*(row[0] for row in rows))
        def lead(value: Any) -> Array:
            return jnp.asarray(value)[None]

        stacked = cast(NearContact, _tree_util.tree_map(lead, contact))
        wall_along, width, footprint = wall_along[None], width[None], footprint[None]
    else:
        stacked, wall_along, width, footprint = _vmap(one)(*rows)
    kind_of = _lax.stop_gradient(classified(stacked.k2, stacked.b2, footprint, wall_along, resolved, spacing))
    return Found(NearContacts(stacked, merged(stacked.point, reach), kind_of, wall_along, footprint), width)


def relative_curvature(body: Array, wall: Array) -> Array:
    """The gap's curvature: the body's less the wall's, each ``2 x 2`` in one tangent frame, the wall's positive where
    it curves away from the medium (a tube's lumen curves towards the body, so its gap is wider than a plane's). A
    module attribute so that a control can plant the wrong sign."""
    return body - wall


__all__ = ["GROUPS", "RADIUS_FLOOR", "Found", "Search", "near_contacts", "relative_curvature", "search_for"]
