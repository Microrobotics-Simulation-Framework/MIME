"""Analytical signed distance functions for IBM bodies.

All functions return ``φ(x)`` of shape matching the leading dims of the
input position array, with the convention

    φ < 0  inside body
    φ > 0  outside body
    φ = 0  on surface.

All routines are pure JAX so they're vmap-/grad-/jit-transparent. In
particular, ``jax.jacobian`` of an IBM-derived force with respect to
the body's pose parameters works because every constructor below
captures pose as JAX arrays.

Implemented primitives:

* :func:`sphere_sdf`
* :func:`infinite_cylinder_sdf` (pipe wall)
* :func:`pipe_interior_sdf` — convenience: negative inside the pipe
  bore (i.e. wall is the IBM "body")
* :func:`capsule_sdf` (axis-aligned segment + spherical caps)

References
----------
- Inigo Quilez, "Distance functions" (https://iquilezles.org/articles/distfunctions/)
  — canonical analytical SDF formulae.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sphere_sdf(x: jnp.ndarray, *, center: jnp.ndarray, radius: float) -> jnp.ndarray:
    """``||x − c|| − r``. Differentiable in ``center`` and ``radius``."""
    d = x - center
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30) - radius


def infinite_cylinder_sdf(
    x: jnp.ndarray, *, center: jnp.ndarray, axis: jnp.ndarray, radius: float,
) -> jnp.ndarray:
    """SDF of an infinite circular cylinder of radius ``r``.

    ``axis`` need not be unit length — it is normalised internally. A
    point ``x`` projects onto the axis at ``c + (axis · (x − c)) axis``;
    the SDF is the distance from ``x`` to that projection minus ``r``.
    """
    a = axis / (jnp.linalg.norm(axis) + 1e-30)
    d = x - center
    proj = jnp.einsum("...i,i->...", d, a)
    radial = d - proj[..., None] * a
    rho = jnp.sqrt(jnp.sum(radial * radial, axis=-1) + 1e-30)
    return rho - radius


def pipe_interior_sdf(
    x: jnp.ndarray, *, center: jnp.ndarray, axis: jnp.ndarray, radius: float,
) -> jnp.ndarray:
    """Negative inside the pipe bore, positive in the wall material.

    Convenience wrapper for using the *exterior* of an
    :func:`infinite_cylinder_sdf` as an IBM body.
    """
    return -infinite_cylinder_sdf(x, center=center, axis=axis, radius=radius)


def capsule_sdf(
    x: jnp.ndarray,
    *,
    p0: jnp.ndarray, p1: jnp.ndarray, radius: float,
) -> jnp.ndarray:
    """SDF of a capsule (cylinder with hemispherical caps) ``p0 → p1``."""
    pa = x - p0
    ba = p1 - p0
    h = jnp.clip(
        jnp.einsum("...i,i->...", pa, ba)
        / (jnp.sum(ba * ba) + 1e-30),
        0.0, 1.0,
    )
    seg = pa - h[..., None] * ba
    return jnp.sqrt(jnp.sum(seg * seg, axis=-1) + 1e-30) - radius


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def union_sdf(*phis: jnp.ndarray) -> jnp.ndarray:
    """Boolean union: ``min(φ_1, φ_2, ...)``.

    This makes the cell IBM "inside the union" when any constituent SDF
    is negative.
    """
    if len(phis) == 0:
        raise ValueError("union_sdf needs at least one operand")
    out = phis[0]
    for p in phis[1:]:
        out = jnp.minimum(out, p)
    return out


def intersection_sdf(*phis: jnp.ndarray) -> jnp.ndarray:
    """Boolean intersection: ``max(φ_1, φ_2, ...)``."""
    if len(phis) == 0:
        raise ValueError("intersection_sdf needs at least one operand")
    out = phis[0]
    for p in phis[1:]:
        out = jnp.maximum(out, p)
    return out


# ---------------------------------------------------------------------------
# Rigid body velocity at a point
# ---------------------------------------------------------------------------

def rigid_body_velocity(
    x: jnp.ndarray,
    *,
    pose_x: jnp.ndarray,             # body reference point (e.g. centre of mass)
    linear_velocity: jnp.ndarray,    # [dim]
    angular_velocity: jnp.ndarray | None = None,  # [3] in 3D, scalar in 2D
) -> jnp.ndarray:
    """Velocity at world point ``x`` of a rigid body with pose+twist.

    ``u(x) = v + ω × (x − pose_x)`` (3D) or ``u(x) = v + ω × (r·ê_z)``
    expressed component-wise in 2D.
    """
    r = x - pose_x
    if angular_velocity is None:
        return jnp.broadcast_to(linear_velocity, x.shape)
    if x.shape[-1] == 3:
        omega_cross_r = jnp.cross(angular_velocity, r)
    else:
        # 2D: omega is scalar (z-component); ω × r = (-ω r_y, ω r_x)
        omega = angular_velocity if jnp.ndim(angular_velocity) == 0 \
            else angular_velocity[..., 0]
        omega_cross_r = jnp.stack(
            [-omega * r[..., 1], omega * r[..., 0]], axis=-1,
        )
    return linear_velocity + omega_cross_r
