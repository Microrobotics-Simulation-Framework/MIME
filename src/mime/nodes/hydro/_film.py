"""The thin film between a body and a surface it nearly touches: Reynolds' lubrication equation on the gap's second-order
form at the contact, the near-field model's completion in the gap (lubrication's hand-over, ``mime.nodes.hydro.mrs``),
not physics beyond Stokes flow.

**The film.** In the contact's frame (the origin on the wall below the body's nearest point, ``z`` along the wall's
normal into the medium, ``x`` and ``y`` along the gap's principal directions), the body's surface moves rigidly over the
still wall, ``V(X) = U + Omega x (X - Q)``, ``Q`` the body's nearest point, across the gap

    h(x, y) = h_c + k_1 x^2 / 2 + k_2 y^2 / 2,     k_1 >= k_2 >= 0,

the relative curvature of the two surfaces (the body's less the wall's, from the body's support and the wall's shape).
Reynolds' equation for that motion, ``div(h^3 grad p) = 12 mu (V_z - V_t . grad h) + 6 mu div(h V_t)`` with ``p = 0``
on the patch's edge, is discretised by vertex-centred finite volumes on a tensor grid of :data:`FILM_NODES` a side,
each axis stretched by ``sinh`` towards the contact over the film's own width ``sqrt(2 h_c / k_i)``; it is solved
exactly by block-tridiagonal elimination along ``x`` (one dense solve per grid line, the six rigid modes at once), so
the solve has no iteration, no cap and no tolerance. The tractions on the body are the pressure along the normal and
``-p grad h - mu V_t / h - (h / 2) grad p`` along the wall; their force and moment about ``Q`` per unit rigid motion is
the film's 6 x 6, the force the body exerts on the fluid (positive definite, the drag's convention).

**Its patch.** The gap is capped smoothly at :data:`CAP_SHARE` of the smaller radius of curvature (``1 / k_1``), so the
film past it is wide and inert, and each axis reaches where the paraboloid meets that cap, no farther than the body's
own ``reach``. The film's form is right for the gap's singular terms (the sphere's ``a / h`` and its three logarithms,
a line contact's ``h^-3/2``) and holds no O(1) part of its own: it is used only as an increment between two gaps,
``TF(h) - TF(h')``, never alone. Its 6 x 6 is not exactly symmetric: the grid's error, about 2.5 % of the coupling at a
gap of 0.002 of the radius and 4.4 % at 5e-4 on 65 nodes a side (1 % on 257), which the drag's node averages away when
it symmetrises the resistance.

**Its validity.** A smooth contact at gaps well below its smaller radius of curvature (C3); a sharp edge (a radius
near zero) leaves a film near zero, and the MRS with its sampling carries the edge. Below the surfaces' roughness the
film's gap is floored (:func:`smooth_floor`): the resistance stops growing there, and contact carries the rest."""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
from jax import Array

from mime.core import jnp
from mime.core.solves import dense_solve

_lax: Any = cast(Any, jax).lax

#: The film's grid, nodes per axis (odd: one node on the contact).
FILM_NODES = 65
#: The film's gap is capped smoothly at this share of the contact's smaller radius of curvature (the hand-over spike's
#: patch: a cap at half the radius held the spheroid's increment to 3.5 % at a hand-over gap of a tenth of it, a cap at
#: twice the radius let the film run past the body's ends, 11.6 %).
CAP_SHARE = 0.5
#: The cap's smoothing width, a share of the cap.
CAP_SOFT = 0.05


def smooth_floor(value: Array, floor: Array, width: Array) -> Array:
    """A smooth ``max(value, floor)``: ``value`` itself from ``floor + width`` up, ``floor`` from ``floor - width``
    down, and between them ``floor + width g(t)``, ``t = (value - floor) / width``, ``g`` the integral of the smoothstep
    (``g(t) = 2 v^3 - v^4``, ``v = (t + 1) / 2``): never below either, continuous with two derivatives. The lift of a
    contact's gap to the hand-over gap, and the film's floor at the surfaces' roughness."""
    t = (value - floor) / width
    v = jnp.clip(0.5 * (t + 1.0), 0.0, 1.0)
    return jnp.where(t >= 1.0, value, floor + width * (2.0 * v ** 3 - v ** 4))


def _smooth_min(a: Array, b: Array, soft: Array) -> Array:
    """``-soft log(exp(-a / soft) + exp(-b / soft))``: within ``soft log 2`` of ``min(a, b)``, smooth."""
    low = jnp.minimum(a, b)
    return low - soft * jnp.log(jnp.exp(-(a - low) / soft) + jnp.exp(-(b - low) / soft))


def _axis(count: int, half: Array, core: Array) -> Array:
    """``count`` nodes on ``[-half, half]``, stretched by ``sinh`` towards 0 over ``core``."""
    xi = jnp.linspace(-1.0, 1.0, count, dtype=half.dtype)
    return core * jnp.sinh(xi * jnp.arcsinh(half / core))


def _block_tridiagonal(T: Array, W: Array, E: Array, r: Array) -> Array:
    """``p`` with ``-W_i p_(i-1) + T_i p_i - E_i p_(i+1) = r_i`` (``p_(-1) = p_m = 0``): ``T`` ``(m, q, q)``, ``W`` and
    ``E`` the diagonals ``(m, q)`` of the neighbouring blocks, ``r`` ``(m, q, k)``; exact elimination, a dense solve per
    line."""
    q = T.shape[1]

    def forward(carry: Array, line: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        T_i, W_i, E_i, r_i = line
        G = T_i - W_i[:, None] * carry[:, :q]
        y = r_i + W_i[:, None] * carry[:, q:]
        X = dense_solve(G, jnp.concatenate([jnp.diag(E_i), y], axis=1))
        return X, X

    _, eliminated = _lax.scan(forward, jnp.zeros((q, q + r.shape[-1]), dtype=T.dtype), (T, W, E, r))

    def backward(after: Array, X: Array) -> tuple[Array, Array]:
        p = X[:, q:] + X[:, :q] @ after
        return p, p

    _, solved = _lax.scan(backward, jnp.zeros((q, r.shape[-1]), dtype=T.dtype), eliminated, reverse=True)
    return solved


def _above(diagonal: Array) -> Array:
    return jnp.diag(diagonal, 1)


def _below(diagonal: Array) -> Array:
    return jnp.diag(diagonal, -1)


def _rigid(X: Array, Q: Array) -> Array:
    """``(..., 3, 6)``: the velocity at ``X`` per unit rigid mode about ``Q`` (three translations, three rotations)."""
    arm = X - Q
    eye = jnp.eye(3, dtype=X.dtype)
    turning = jnp.stack([jnp.cross(eye[i], arm) for i in range(3)], axis=-1)
    return jnp.concatenate([jnp.broadcast_to(eye, arm.shape[:-1] + (3, 3)), turning], axis=-1)


def _gradient(f: Array, nodes: Array, axis: int) -> Array:
    """``df / d(nodes)`` along ``axis``: second order inside, first at the ends."""
    f = jnp.moveaxis(f, axis, 0)
    step = nodes[1:] - nodes[:-1]
    shape = (-1,) + (1,) * (f.ndim - 1)
    left, right = step[:-1].reshape(shape), step[1:].reshape(shape)
    inner = (-(right / (left * (left + right))) * f[:-2] + ((right - left) / (left * right)) * f[1:-1]
             + (left / (right * (left + right))) * f[2:])
    found = jnp.concatenate([(f[1:2] - f[:1]) / step[0], inner, (f[-1:] - f[-2:-1]) / step[-1]], axis=0)
    return jnp.moveaxis(found, 0, axis)


class Film(NamedTuple):
    """A film's solve: its 6 x 6 (:func:`film`) and the elimination's residual, the largest of the system's residuals
    over the largest of its right-hand sides (rounding, for an exact elimination: a report, which no derivative passes
    through)."""

    resistance: Array
    residual: Array


def film(gap: Array, k1: Array, k2: Array, reach: Array, viscosity: Array, cap_floor: Array,
         nodes: int = FILM_NODES) -> Film:
    """The film's 6 x 6 about the body's nearest point, in the contact's frame (``x`` along ``k1``'s direction, ``z``
    the wall's normal into the medium): the force and moment the body exerts on the fluid per unit rigid motion, at the
    gap ``gap`` (m) with the relative curvatures ``k1 >= k2 >= 0`` (1/m), ``k1`` above zero, the patch no wider than
    ``reach`` (m) along either axis, the cap at :data:`CAP_SHARE` of ``1 / k1`` and never below ``cap_floor`` (m),
    which a caller sets above every gap it evaluates, so that a sharp edge (``k1`` large) leaves a film near zero."""
    dtype = gap.dtype
    cap = smooth_floor(CAP_SHARE / k1, cap_floor, 0.5 * cap_floor)
    soft = CAP_SOFT * cap
    halves = [1.0 / jnp.sqrt(k / (2.0 * cap) + 1.0 / reach ** 2) for k in (k1, k2)]
    cores = [1.0 / jnp.sqrt(k / (2.0 * gap) + 1.0 / half ** 2) for k, half in zip((k1, k2), halves, strict=True)]
    xs, ys = (_axis(nodes, half, core) for half, core in zip(halves, cores, strict=True))

    def height(x: Array, y: Array) -> Array:
        return _smooth_min(gap + 0.5 * k1 * x * x + 0.5 * k2 * y * y, cap, soft)

    def slope(x: Array, y: Array) -> tuple[Array, Array]:
        raw = gap + 0.5 * k1 * x * x + 0.5 * k2 * y * y
        low = jnp.minimum(raw, cap)
        share = jnp.exp(-(raw - low) / soft) / (jnp.exp(-(raw - low) / soft) + jnp.exp(-(cap - low) / soft))
        return share * k1 * x, share * k2 * y

    Q = jnp.stack([jnp.zeros((), dtype), jnp.zeros((), dtype), gap])
    XX, YY = jnp.meshgrid(xs, ys, indexing="ij")
    H = height(XX, YY)
    Hx, Hy = slope(XX, YY)
    xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    He, Hn = height(xm[:, None], ys[None, :]), height(xs[:, None], ym[None, :])
    dx, dy = xs[1:] - xs[:-1], ys[1:] - ys[:-1]
    wx = jnp.concatenate([dx[:1] / 2, (dx[1:] + dx[:-1]) / 2, dx[-1:] / 2])
    wy = jnp.concatenate([dy[:1] / 2, (dy[1:] + dy[:-1]) / 2, dy[-1:] / 2])
    area = wx[:, None] * wy[None, :]
    at_nodes = _rigid(jnp.stack([XX, YY, H], -1), Q)
    east = _rigid(jnp.stack([jnp.broadcast_to(xm[:, None], He.shape), jnp.broadcast_to(ys[None, :], He.shape), He], -1),
                  Q)
    north = _rigid(jnp.stack([jnp.broadcast_to(xs[:, None], Hn.shape), jnp.broadcast_to(ym[None, :], Hn.shape), Hn],
                             -1), Q)
    squeeze = at_nodes[..., 2, :] - at_nodes[..., 0, :] * Hx[..., None] - at_nodes[..., 1, :] * Hy[..., None]
    through_east = 6.0 * He[..., None] * east[..., 0, :] * wy[None, :, None]
    through_north = 6.0 * Hn[..., None] * north[..., 1, :] * wx[:, None, None]
    source = 12.0 * squeeze * area[..., None]
    source = source.at[:-1].add(through_east).at[1:].add(-through_east)
    source = source.at[:, :-1].add(through_north).at[:, 1:].add(-through_north)
    conduct_east = He ** 3 * wy[None, :] / dx[:, None]
    conduct_north = Hn ** 3 * wx[:, None] / dy[None, :]
    inside = slice(1, -1)
    lines = nodes - 2
    west_i, east_i = conduct_east[:-1, inside], conduct_east[1:, inside]
    south_i, north_i = conduct_north[inside, :-1], conduct_north[inside, 1:]
    across = -conduct_north[inside, 1:-1]
    T = (jax.vmap(jnp.diag)(west_i + east_i + south_i[:, :lines] + north_i[:, :lines])
         + jax.vmap(_above)(across) + jax.vmap(_below)(across))
    rhs = -source[inside, inside, :]
    interior = solve_film(T, west_i, east_i, rhs)
    padded = jnp.concatenate([jnp.zeros_like(interior[:1]), interior, jnp.zeros_like(interior[:1])])
    unmet = (jnp.einsum("iqr,irk->iqk", T, interior) - west_i[..., None] * padded[:-2] - east_i[..., None] * padded[2:]
             - rhs)
    residual = _lax.stop_gradient(jnp.max(jnp.abs(unmet)) / jnp.max(jnp.abs(rhs)))
    P = jnp.zeros((nodes, nodes, 6), dtype=dtype).at[inside, inside, :].set(interior)
    Px, Py = _gradient(P, xs, 0), _gradient(P, ys, 1)
    shear_x = -P * Hx[..., None] - at_nodes[..., 0, :] / H[..., None] - 0.5 * H[..., None] * Px
    shear_y = -P * Hy[..., None] - at_nodes[..., 1, :] / H[..., None] - 0.5 * H[..., None] * Py
    traction = jnp.stack([shear_x, shear_y, P], axis=-2)
    force = jnp.einsum("ijkm,ij->km", traction, area)
    arm = jnp.stack([XX, YY, H], -1) - Q
    moment = jnp.einsum("ijkm,ij->km", jnp.cross(arm[..., None, :], jnp.moveaxis(traction, -1, -2), axis=-1)
                        .transpose(0, 1, 3, 2), area)
    return Film(-viscosity * jnp.concatenate([force, moment], axis=0), residual)


#: The film's elimination, a module attribute so that a control can plant a wrong solve in its place.
solve_film = _block_tridiagonal


def tangent_frame(normal: Array) -> Array:
    """Two unit directions across ``normal`` and orthogonal to each other (rows, ``2 x 3``), from the coordinate
    direction least along it: a frame across a direction cannot be continuous in every direction; the film's own frame
    (:func:`principal`) is the gap's, whatever this one is, but at a contact equally curved every way."""
    first = jnp.cross(normal, jnp.eye(3, dtype=normal.dtype)[jnp.argmin(jnp.abs(normal))])
    first = first / jnp.sqrt(jnp.sum(first * first))
    return jnp.stack([first, jnp.cross(normal, first)])


def principal(curvature: Array, tangent: Array, normal: Array) -> tuple[Array, Array, Array]:
    """The contact's frame (rows: the larger curvature's direction, the smaller's, ``normal``) and its curvatures
    ``k1 >= k2`` from the gap's ``2 x 2`` ``curvature`` in the ``tangent`` frame; at a contact curved equally every way
    (within rounding of the mean) the tangent frame itself, the film being the same in every frame there but for its
    grid."""
    mean = 0.5 * (curvature[0, 0] + curvature[1, 1])
    half = 0.5 * (curvature[0, 0] - curvature[1, 1])
    split_squared = half * half + curvature[0, 1] * curvature[0, 1]
    resolution = (64.0 * jnp.finfo(curvature.dtype).eps * jnp.abs(mean)) ** 2
    open_ = split_squared > resolution
    split = jnp.where(open_, jnp.sqrt(jnp.where(open_, split_squared, 1.0)), 0.0)
    angle = jnp.where(open_, 0.5 * jnp.arctan2(curvature[0, 1], jnp.where(open_, half, 1.0)), 0.0)
    c, s = jnp.cos(angle), jnp.sin(angle)
    frame = jnp.stack([c * tangent[0] + s * tangent[1], -s * tangent[0] + c * tangent[1], normal])
    return frame, mean + split, mean - split


def about(resistance: Array, frame: Array, arm: Array) -> Array:
    """``resistance`` (6 x 6 about a point, in ``frame``, whose rows are its axes in the world) in the world frame and
    about the point ``arm`` back from it (``arm`` the point less the new centre): ``T^T B R B^T T``, ``B`` the frame's
    rotation to the world on both blocks and ``T`` the twist about the centre carried to the point."""
    turn = frame.T
    zero = jnp.zeros((3, 3), dtype=resistance.dtype)
    rotation = jnp.block([[turn, zero], [zero, turn]])
    world = rotation @ resistance @ rotation.T
    cross = jnp.stack([jnp.stack([0.0 * arm[0], -arm[2], arm[1]]), jnp.stack([arm[2], 0.0 * arm[0], -arm[0]]),
                       jnp.stack([-arm[1], arm[0], 0.0 * arm[0]])])
    carry = jnp.eye(6, dtype=resistance.dtype).at[:3, 3:].set(-cross)
    return carry.T @ world @ carry


__all__ = ["CAP_SHARE", "CAP_SOFT", "FILM_NODES", "Film", "about", "film", "principal", "smooth_floor", "solve_film",
           "tangent_frame"]
