"""The regularised-Stokeslet boundary-element method for a rigid body in Stokes flow: the matrix of a body's samples,
its solve and the 6 x 6 resistance (Cortez, Fauci & Medovikov 2005).

**The formulation: point forces.** The unknown at each sample ``x_n`` is the force ``g_n`` the body puts on the fluid
there, the traction times the sample's quadrature weight, and the velocity it makes at ``x_m`` is ``S(x_m, x_n) g_n``,
with the regularised Stokeslet

    S(x, y) = [(r^2 + 2 eps^2) I + r r^T] / (8 pi mu (r^2 + eps^2)^(3/2)),   r = x - y.

``S(x, y) = S(y, x)`` and each block is symmetric, so the matrix ``A`` of every pair is symmetric whatever the weights:
the adjoint solves with it again. (The 0.2 line's unknown was the traction, ``A_mn = w_n S(x_m, x_n)``, symmetric only
where the weights are equal; its resistance is the same, the weights cancelling between the solve and the force sum.)
A rigid motion ``(U, Omega)`` about a centre ``c`` puts ``U + Omega x (x_m - c)`` at each sample: the columns of ``M``,
three unit translations and three unit rotations, so that ``M^T g`` is the force and the torque about ``c`` of the
forces ``g``. Then

    R = M^T A^-1 M,

the force and torque the body puts on the fluid for each unit motion: symmetric positive definite, and what the body
receives is ``-R [U; Omega]``.

**The blob** ``eps`` is ``blob`` times the samples' mean spacing, the square root of their area over their count (the
0.2 line took half of it): so the weights reach the resistance through the blob alone. Only ``eps^2`` is used.

**A wall** reaches the matrix as a block matrix added to ``A`` (``mime.nodes.hydro.bem.wall_correction``), in the same
units, the velocity at each sample per unit force at each: free space adds nothing.

**Differentiation.** ``jax.numpy.linalg.solve`` factorises ``A`` once with its derivatives stopped and differentiates
by the linear-solve rule, ``d(A^-1 b) = A^-1 (db - dA A^-1 b)``, reusing the factors for each tangent, and its transpose
solves with the same factors: nothing is differentiated through the factorisation."""

from __future__ import annotations

import math
from typing import Any

from jax import Array

from mime.core import jnp

#: Matrix products here, and the node's placement of the samples, are taken at full precision: on a GPU the default is
#: a reduced one (TF32), which the 0.2 line found moved its results by 15%.
PRECISION = "highest"


def blob_squared(weights: Array, blob: Any) -> Array:
    """``eps^2``: ``blob`` squared times the samples' area over their count."""
    return blob * blob * jnp.sum(weights) / weights.shape[0]


def stokeslet_matrix(points: Array, eps_squared: Any, viscosity: Any) -> Array:
    """The ``3N x 3N`` matrix of the regularised Stokeslet between every pair of the ``N`` points: the velocity at
    each point per unit force at each, in m/(N s)."""
    count = points.shape[0]
    r = points[:, None, :] - points[None, :, :]
    r2 = jnp.sum(r * r, axis=-1)
    scale = 1.0 / (8.0 * math.pi * viscosity * (r2 + eps_squared) ** 1.5)
    eye = jnp.eye(3, dtype=points.dtype)
    blocks = (eye * (r2 + 2.0 * eps_squared)[..., None, None] + r[..., :, None] * r[..., None, :]) \
        * scale[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(3 * count, 3 * count)


def rigid_motions(points: Array, centre: Array) -> Array:
    """``M``, ``3N x 6``: each sample's velocity for a unit translation along x, y, z and a unit rotation about each,
    about ``centre``."""
    arm = points - centre
    eye = jnp.eye(3, dtype=points.dtype)
    columns = [jnp.broadcast_to(eye[i], arm.shape) for i in range(3)] + [jnp.cross(eye[i], arm) for i in range(3)]
    return jnp.stack(columns, axis=-1).reshape(3 * points.shape[0], 6)


def point_forces(matrix: Array, motions: Array) -> Array:
    """``A^-1 M``: the force at each sample for each unit motion, ``3N x 6``, by the linear-solve rule."""
    return jnp.linalg.solve(matrix, motions)


def resistance(points: Array, eps_squared: Any, viscosity: Any, centre: Array, correction: Array | None = None) -> Array:
    """``R = M^T (A + G)^-1 M`` about ``centre``, with ``G`` the wall's ``correction`` (none in free space); its
    symmetric part, since it is symmetric but for the solve's rounding."""
    matrix = stokeslet_matrix(points, eps_squared, viscosity)
    if correction is not None:
        matrix = matrix + correction
    motions = rigid_motions(points, centre)
    found = jnp.matmul(motions.T, point_forces(matrix, motions), precision=PRECISION)
    return (found + found.T) / 2.0


__all__ = ["PRECISION", "blob_squared", "point_forces", "resistance", "rigid_motions", "stokeslet_matrix"]
