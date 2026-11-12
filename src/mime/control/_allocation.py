"""A bounded allocation law in JAX (camera spike R2): currents within declared limits that come nearest a wanted
quantity in least squares, the problem :func:`mime.fields.allocate` hands SciPy's BVLS, solved exactly by its active
sets so that a derivative passes through it.

**The law.** ``min ||A i - b||`` over ``|i_k| <= L_k``. For ``K`` coils every one of the ``3^K`` active sets is
enumerated (each coil free, at ``+L`` or at ``-L``); a set's free currents are the unconstrained least squares of
its free columns against what its bound coils leave. The minimiser is the solution of the set whose candidate is
feasible and has the least residual: the box minimiser's free currents have a zero gradient, so it is its own set's
candidate, and no feasible candidate does better. Where the rows are rank deficient, a minimiser exists whose free
columns are independent (move along a null direction until a coil reaches its limit), so the sets whose free columns
are dependent are dropped and the residual is still the least; the currents are then one of several minimisers.

**The traps** (the camera design's v4, its differentiable closed loop):

- *exact by active sets*: the enumeration above, ``3^K`` small solves, for the done case's three coils 27;
- *a float64 guard or scaled rows*: each free solve is a QR of the columns scaled to unit length (never the normal
  equations, which square the condition number), so float32 loses ``kappa`` and not ``kappa^2``;
- *the double ``where``*: a set whose free columns are dependent has a singular triangle, whose solve is not finite;
  its columns are replaced by a well-conditioned stand-in *before* the solve (the first ``where``) and its candidate
  discarded after (the second), so that no NaN reaches the derivative through ``0 * NaN``;
- *rank deficiency*: such a set is never selected (its rank is judged on the scaled triangle's diagonal against a
  relative ``rcond``).

``variant`` plants the controls: ``"single_where"`` drops the first ``where``; ``"stop_branch"`` stops the derivative
of what a bound coil leaves the free ones (a branch with a bound coil differentiated as if the bound coils were
constants of the problem); ``"free_tangent"`` differentiates every point as the unconstrained solve (the clipped
primal with the derivative of the wrong active set)."""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any, NamedTuple

import jax
import numpy as np

jnp: Any = jax.numpy
lax: Any = jax.lax

#: A free set is taken as rank deficient below this ratio of its scaled triangle's least and greatest diagonal entry
#: (columns scaled to unit length, so the ratio is a lower bound on the reciprocal condition up to the set's size).
RCOND = {np.dtype(np.float64): 1e-10, np.dtype(np.float32): 1e-5}
#: A candidate's free current is within its limit up to this many units in the last place of the limit.
FEASIBLE_ULPS = 64


class Allocated(NamedTuple):
    """The currents, the index of the selected active set, its signs (``-1``, ``0`` free, ``+1``) and the residual."""

    currents: Any
    selected: Any
    signs: Any
    residual: Any


@cache
def active_sets(coils: int) -> np.ndarray:
    """Every active set of ``coils`` coils, ``(3^K, K)``: ``0`` free, ``+1`` at its upper limit, ``-1`` at its lower."""
    return np.asarray(list(itertools.product((0, 1, -1), repeat=coils)), dtype=np.int8)


def _free_solve(columns: Any, rest: Any, ok: Any, variant: str) -> Any:
    """The least squares of the scaled ``columns`` ``(R, F)`` against ``rest``: by QR, the triangle's solve."""
    if variant != "single_where":
        stand_in = jnp.eye(columns.shape[0], columns.shape[1], dtype=columns.dtype)
        columns = jnp.where(ok, columns, stand_in)
    q, r = jnp.linalg.qr(columns)
    return jax.scipy.linalg.solve_triangular(r, q.T @ rest, lower=False)


def bounded_lsq(matrix: Any, target: Any, limits: Any, *, variant: str = "double_where") -> Allocated:
    """The currents ``(K,)`` within ``limits`` nearest ``target`` ``(R,)`` through ``matrix`` ``(R, K)``, exactly, by
    enumerating the active sets (see the module). Every argument may be traced."""
    matrix, target, limits = jnp.asarray(matrix), jnp.asarray(target), jnp.asarray(limits)
    dtype = matrix.dtype
    eps = float(np.finfo(dtype).eps)
    coils = matrix.shape[1]
    scale = jnp.sqrt(jnp.sum(matrix * matrix, axis=0))
    scaled = matrix / scale
    bounds = limits * scale
    candidates, residuals, feasible = [], [], []
    sets = active_sets(coils)
    for signs in sets:
        free = [k for k in range(coils) if signs[k] == 0]
        bound = [k for k in range(coils) if signs[k] != 0]
        y = jnp.zeros((coils,), dtype=dtype)
        held = jnp.asarray([float(signs[k]) for k in bound], dtype=dtype) * bounds[np.asarray(bound, dtype=int)] \
            if bound else jnp.zeros((0,), dtype=dtype)
        if bound:
            y = y.at[np.asarray(bound)].set(held)
        left = scaled[:, np.asarray(bound, dtype=int)] @ held if bound else jnp.zeros_like(target)
        if variant == "stop_branch" and bound:
            left = lax.stop_gradient(left)
        ok, inside = jnp.asarray(True), jnp.asarray(True)
        if len(free) > matrix.shape[0]:
            # More free coils than rows: the free columns are dependent whatever they are, and no candidate is formed.
            ok = jnp.asarray(False)
        elif free:
            columns = scaled[:, np.asarray(free)]
            diagonal = jnp.abs(jnp.diag(jnp.linalg.qr(lax.stop_gradient(columns))[1]))
            ok = jnp.min(diagonal) > RCOND[np.dtype(dtype)] * jnp.max(diagonal)
            solved = _free_solve(columns, target - left, ok, variant)
            y = y.at[np.asarray(free)].set(solved)
            inside = jnp.all(jnp.abs(solved) <= bounds[np.asarray(free)] * (1.0 + FEASIBLE_ULPS * eps))
        miss = scaled @ y - target
        candidates.append(y)
        residuals.append(jnp.sum(miss * miss))
        feasible.append(ok & inside)
    stacked, residual, admitted = jnp.stack(candidates), jnp.stack(residuals), jnp.stack(feasible)
    selected = jnp.argmin(jnp.where(admitted, lax.stop_gradient(residual), jnp.inf))
    chosen = jnp.arange(len(sets)) == selected
    y = jnp.sum(jnp.where(chosen[:, None], stacked, 0.0), axis=0)
    currents = y / scale
    if variant == "free_tangent":
        unconstrained = _free_solve(scaled, target, jnp.asarray(True), "double_where") / scale
        currents = lax.stop_gradient(currents) + unconstrained - lax.stop_gradient(unconstrained)
    return Allocated(currents, selected, jnp.asarray(sets)[selected], residual[selected])


def force_rows(gradients: Any, moment: Any) -> Any:
    """The force per unit current on a moment ``(3,)`` in a field whose gradient per unit current is ``gradients``
    ``(3, 3, K)`` (``dB_i/dx_j``): ``F_j = m_i dB_i/dx_j``, ``(3, K)``."""
    return jnp.einsum("i,ijk->jk", moment, gradients)


def gradient5(gradients: Any) -> Any:
    """The gradient's five independent rows ``[Gxx, Gxy, Gxz, Gyy, Gyz]`` per unit current, ``(5, K)``."""
    return jnp.stack([gradients[0, 0], gradients[0, 1], gradients[0, 2], gradients[1, 1], gradients[1, 2]])


__all__ = ["RCOND", "Allocated", "active_sets", "bounded_lsq", "force_rows", "gradient5"]
