"""The plane wall's correction for the method of regularised Stokeslets: the image system of a regularised Stokeslet
above an unbounded no-slip plane, in closed form, regularised by the drag's own blob (Blake 1971's image system;
Ainley, Durkin, Embid, Boindala & Cortez 2008 for its regularised form).

**The image.** The plane is ``x . n = offset``, the medium on the side its unit normal ``n`` points into. For a force
``f`` at a source ``y`` a height ``h = y . n - offset`` above it, with ``y* = y - 2 h n`` its mirror image, ``R = x - y*``
from the mirror to the target, ``s = |R|^2``, ``rho^2 = s + eps^2``, ``M = I - 2 n n^T`` the reflection and ``g = M f``,
the image's velocity at ``x`` is

    8 pi mu u = - S(R) f                                       (a Stokeslet, the source's blob)
                - 2 h g_k d/dR_k [S(R) n]                      (a Stokes doublet, the source's blob)
                + h^2 [D1 g + D2 R (g . R)]                    (a potential dipole, D1 = (2 s - 4 eps^2) / rho^5,
                                                                D2 = -6 / rho^5)
                - 6 h eps^2 / rho^5 [n (R . f) - f (R . n)]    (a rotlet about f x n, of order eps^2)

with ``S_ij = delta_ij (s + 2 eps^2) / rho^3 + R_i R_j / rho^3``, MIME's regularised Stokeslet
(``mime.nodes.hydro.mrs.stokeslet_matrix``). Added to that free Stokeslet of the same blob, the velocity is exactly
zero on the plane and exactly divergence-free, and the whole kernel is reciprocal, ``G(x, y) = G(y, x)^T``; as ``eps``
goes to zero it is Blake's singular image. Written in the plane's normal, never in a frame across it, so it is a smooth
function of the normal and the offset.

**The regularisation is the drag's.** The node passes ``eps_squared``, the blob's ``eps^2`` its free part is made with
(``WallCorrection.regularised``), so the image of each source is the no-slip partner of that source's own regularised
Stokeslet: integrated over a body's cells (``cell_image``, sources the cells' quadrature points) it is the partner of
the free part's integral; taken at the force points (the default), an approximation of it, as the force points are of
the cells.

**Its domain** is the medium's side: a sample through the plane (``h < 0``) has its mirror image in the medium and the
formula there is no image of anything. Nothing refuses it in the program (the plane has no table whose range a read can
leave): the node reports the body's least distance from the plane each step (``wall_gap``, below zero through it) and
raises ``near_wall`` within a force spacing, and the build refuses a body that starts through it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

from jax import Array

from mime.core import jnp, switch
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection, register_wall_correction

#: The plane wall kind's name and version (``mime.walls.PlaneWall``).
PLANE, PLANE_VERSION = "plane", "0.1.0"
#: The plane's constants: its normal (held as built: the wall's frame, a direction) and its offset along it (the
#: node's leaf ``wall_offset``, trainable, which lubrication's opening of a contact moves; not yet a study axis).
PLANE_PARAMETERS = ("normal", "offset")
PLANE_UNITS = {"normal": "1", "offset": "m"}
#: The floating values one pair of the image holds while a chunk of a cell's quadrature is read: its offset and its
#: mirror's (6), five scalars of its distance (5), the reflection's two vectors (6) and five 3 x 3 blocks, the terms
#: and their sum (45), with room.
PLANE_PAIR_VALUES = 96


def unit_normal(parameters: Mapping[str, Any], dtype: Any) -> Array:
    """The plane's normal scaled to unit length, in the program: a direction, which the plan refuses at zero."""
    normal = jnp.asarray(parameters["normal"], dtype=dtype)
    return normal / switch.nonzero_norm(normal)


def heights(points: Array, normal: Array, offset: Any) -> Array:
    """Each point's height above the plane along its unit ``normal`` (m): above zero in the medium."""
    return jnp.sum(points * normal, axis=-1) - offset


class ImageTerms(NamedTuple):
    """The image's four terms at unit viscosity with ``8 pi`` left out, each ``(T, S, 3, 3)``: its Stokeslet, its Stokes
    doublet, its potential dipole and its rotlet (the module's formula). Their sum over ``8 pi mu`` is the image."""

    stokeslet: Array
    doublet: Array
    dipole: Array
    rotlet: Array


def image_terms(targets: Array, sources: Array, parameters: Mapping[str, Any], eps_squared: Any) -> ImageTerms:
    """The plane's image's terms at each of ``targets`` per unit force at each of ``sources`` (world frame, m), with the
    plane's ``normal`` and ``offset`` read from ``parameters`` and the drag's blob ``eps_squared`` (m^2, one value or one
    per source)."""
    dtype = targets.dtype
    n = unit_normal(parameters, dtype)
    h = heights(sources, n, jnp.asarray(parameters["offset"], dtype=dtype))[None, :]
    e2 = jnp.broadcast_to(jnp.asarray(eps_squared, dtype=dtype), (sources.shape[0],))[None, :]
    mirror = sources - 2.0 * h[0][:, None] * n
    R = targets[:, None, :] - mirror[None, :, :]
    along = jnp.sum(R * n, axis=-1)
    reflected = R - 2.0 * along[..., None] * n
    s = jnp.sum(R * R, axis=-1)
    rho2 = s + e2
    rho3 = rho2 ** 1.5
    rho5 = rho2 ** 2.5
    h1 = (s + 2.0 * e2) / rho3
    h2 = 1.0 / rho3
    dh1 = -(0.5 * s + 2.0 * e2) / rho5
    dh2 = -1.5 / rho5
    d1 = (2.0 * s - 4.0 * e2) / rho5
    d2 = -6.0 / rho5
    eye = jnp.eye(3, dtype=dtype)
    reflection = eye - 2.0 * n[:, None] * n[None, :]

    def scalar(x: Array) -> Array:
        return x[..., None, None]

    def outer(a: Array, b: Array) -> Array:
        return a[..., :, None] * b[..., None, :]

    normal = jnp.broadcast_to(n, R.shape)
    stokeslet = -(scalar(h1) * eye + scalar(h2) * outer(R, R))
    # -2 h d/dR_k [S(R) n]_i M_kj: the doublet's derivative times the reflection
    doublet = -2.0 * scalar(h) * (2.0 * scalar(dh1) * outer(normal, reflected) + scalar(along * h2) * reflection
                                  - scalar(h2) * outer(R, normal) + 2.0 * scalar(dh2 * along) * outer(R, reflected))
    dipole = scalar(h * h) * (scalar(d1) * reflection + scalar(d2) * outer(R, reflected))
    rotlet = -6.0 * scalar(h * e2 / rho5) * (outer(normal, R) - scalar(along) * eye)
    return ImageTerms(stokeslet, doublet, dipole, rotlet)


def plane_image(targets: Array, sources: Array, parameters: Mapping[str, Any], viscosity: Any,
                eps_squared: Any) -> Corrected:
    """The plane's regularised image at each of ``targets`` per unit force at each of ``sources`` (world frame, m),
    ``(T, S, 3, 3)`` blocks in m/(N s), of the drag's blob ``eps_squared`` (m^2, one value or one per source): the sum
    of :func:`image_terms` over ``8 pi mu``, the plane's ``normal`` and ``offset`` read from ``parameters`` at each call,
    so that a caller may pass a moved offset. No read to refuse."""
    terms = image_terms(targets, sources, parameters, eps_squared)
    blocks = (terms.stokeslet + terms.doublet + terms.dipole + terms.rotlet) / (8.0 * math.pi * viscosity)
    return Corrected(blocks, jnp.asarray(True))


def plane_correction(points: Array, parameters: Mapping[str, Any], viscosity: Any, eps_squared: Any) -> Corrected:
    """The plane's image at a body's samples ``points`` (world frame, m), the ``3N x 3N`` block matrix: ``plane_image``
    with the targets the sources."""
    found = plane_image(points, points, parameters, viscosity, eps_squared)
    count = 3 * points.shape[0]
    return Corrected(found.matrix.transpose(0, 2, 1, 3).reshape(count, count), found.accepted)


def plane_distance(points: Array, parameters: Mapping[str, Any]) -> Array:
    """Each of ``points``' (world frame, m) height above the plane: the plane wall kind's own distance
    (``WallKind.distance``, minus the half-space's signed distance), from the correction's constants; a report, which
    no derivative is taken through (the node stops it)."""
    n = unit_normal(parameters, points.dtype)
    return heights(points, n, jnp.asarray(parameters["offset"], dtype=points.dtype))


def plane_preparation(parameters: Mapping[str, Any], dtype: Any) -> dict[str, object]:
    """The plane's ``prepare``: its normal as written (normalised in the program) and its offset, from the half-space
    the medium fills."""
    return {"normal": tuple(float(c) for c in parameters["normal"]), "offset": float(parameters["offset"])}


#: What the build report says of a step near the plane: its own measurements, a sphere against Brenner's series (estimates;
#: the wall kind's validity lines carry the rest), and not the tube's.
PLANE_NEAR_WALL = ("on a sphere above the plane (blob 0.1) the normal entry was measured +3.6% at one force spacing, "
                   "+10.3% at 0.55 and +14% at 0.5 with the image at the force points, and the worst entry 2.6% to 14.8% "
                   "off at 0.5 with it integrated; one pose was indefinite, at 0.36 spacings")
PLANE_COST = ("measured on a 4-core laptop CPU in float64, a jitted step took about 9 ms at 231 force points with the "
              "image at the force points and 23 ms integrated (wall_refinement=2), 30 and 48 ms at 338, and a run's "
              "peak was 0.60 and 0.64 GiB at 231 and 0.68 and 0.73 GiB at 338 (estimates for one machine and two sizes; "
              "the dense solve's matrices grow as the square of the count)")
#: The MRS's correction for the plane wall kind: regularised by the drag's blob, its normal held as built (normalised
#: in the program, so that declaring it a leaf later changes no program) and its offset the node's leaf.
PLANE_CORRECTION = register_wall_correction(WallCorrection(
    PLANE, PLANE_VERSION, PLANE_PARAMETERS, plane_correction, prepare=plane_preparation, units=PLANE_UNITS,
    held=frozenset({"normal"}), distance=plane_distance, image=plane_image, pair_values=PLANE_PAIR_VALUES,
    regularised=True, cost_text=PLANE_COST, near_wall_text=PLANE_NEAR_WALL))


__all__ = ["PLANE", "PLANE_CORRECTION", "PLANE_COST", "PLANE_NEAR_WALL", "PLANE_PAIR_VALUES", "PLANE_PARAMETERS", "PLANE_UNITS", "PLANE_VERSION",
           "ImageTerms", "heights", "image_terms", "plane_correction", "plane_distance", "plane_image", "plane_preparation", "unit_normal"]
