"""Diffuse-penalty immersed boundary method (Peskin-style).

Each immersed body is described by a JAX-callable SDF + an optional
JAX-callable rigid-body velocity. The IBM enforces ``u → u_body`` inside
the body via a per-cell penalty force

    f_IBM(x) = -α · H(−φ(x)) · (u(x) − u_body(x))

where ``α`` is the penalty strength and ``H`` is a smoothed Heaviside
function (cosine taper of half-width 2 cells, following Peskin 2002 §3).
The penalty force is added to the momentum equation as a generalised
body force; in :mod:`piso` it goes through the implicit-step splitting
as a *per-cell linear* term (handled exactly by the pointwise
"Brinkman" closed-form update below) so the IBM is unconditionally
stable for arbitrary ``α``.

Force / torque on a body are extracted by Newton's third law as a
masked volume reduce of the per-cell penalty force.

References
----------
- Peskin (2002) "The immersed boundary method", Acta Numerica 11.
- Goldstein, Handler & Sirovich (1993) "Modeling a no-slip flow
  boundary with an external force field", J. Comput. Phys. 105.
- Angot, Bruneau & Fabrie (1999) "A penalization method to take into
  account obstacles…", Numer. Math. 81 — Brinkman penalty motivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Smoothed Heaviside
# ---------------------------------------------------------------------------

def smoothed_indicator(phi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Smoothed indicator I = H(−φ): 1 inside body, 0 outside.

    Cosine taper over width ``2 * eps`` centred at the surface ``φ = 0``:

        I = 1                         if  φ ≤ −eps
        I = 0                         if  φ ≥ +eps
        I = 0.5 (1 − sin(π φ / 2eps)) if  |φ| < eps
    """
    inside = 1.0
    outside = 0.0
    transition = 0.5 * (1.0 - jnp.sin(jnp.pi * phi / (2.0 * eps)))
    return jnp.where(
        phi <= -eps, inside,
        jnp.where(phi >= eps, outside, transition),
    )


# ---------------------------------------------------------------------------
# Body descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IBMBody:
    """A single immersed body for the diffuse-penalty IBM.

    Attributes
    ----------
    name : str
        Identifier (e.g. ``"pipe_wall"``, ``"robot"``).
    sdf : ``Callable[[x], phi]``
        SDF as a JAX function. Must accept ``x`` of shape ``[N_cells, dim]``
        and return ``[N_cells]``.
    u_body : ``Callable[[x], u_body]`` or None
        Velocity field of the body. Returns ``[N_cells, dim]``. ``None``
        ≡ stationary (zero velocity).
    extract_force : bool
        If True, this body's force / torque will be returned by
        :func:`compute_ibm_forces`.
    ref_point : jnp.ndarray or None
        Reference point for torque (centre of mass). Required when
        ``extract_force=True``.
    """
    name: str
    sdf: Callable[[jnp.ndarray], jnp.ndarray]
    u_body: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    extract_force: bool = False
    ref_point: Optional[jnp.ndarray] = None


# ---------------------------------------------------------------------------
# Per-cell penalty body force
# ---------------------------------------------------------------------------

def ibm_body_force(
    u: jnp.ndarray,                  # [N_cells, dim]
    x: jnp.ndarray,                  # [N_cells, dim]
    bodies: Iterable[IBMBody],
    *,
    alpha: float,
    eps: float,
) -> jnp.ndarray:
    """Sum of per-cell IBM penalty forces from all ``bodies``.

    Returns ``[N_cells, dim]`` — same shape as ``u``. Per body the force
    is ``-α · I_body · (u − u_body)``. Bodies are summed, NOT unioned —
    in practice the SDFs should not overlap (e.g. pipe wall + robot).
    """
    out = jnp.zeros_like(u)
    for b in bodies:
        phi = b.sdf(x)                 # [N_cells]
        I = smoothed_indicator(phi, eps)
        if b.u_body is None:
            ub = jnp.zeros_like(u)
        else:
            ub = b.u_body(x)
        out = out - alpha * I[:, None] * (u - ub)
    return out


def ibm_brinkman_implicit_update(
    u: jnp.ndarray,                  # [N_cells, dim]
    x: jnp.ndarray,                  # [N_cells, dim]
    bodies: Iterable[IBMBody],
    *,
    alpha: float,
    eps: float,
    dt: float,
) -> jnp.ndarray:
    """Closed-form pointwise implicit IBM (Brinkman) update.

    For ``∂u/∂t = -α I (u − u_body)`` with frozen ``I, u_body`` over a
    step ``dt``, the analytical solution is

        u(t+dt) = u_body + (u(t) − u_body) · exp(−α I dt).

    For a small step (``α I dt ≪ 1``) this reduces to the linearised
    backward-Euler form ``u(t+dt) = (u + α I dt u_body) / (1 + α I dt)``.
    We use the exponential form because it is exact and unconditionally
    stable for any ``α`` (so the penalty can be made very large without
    a step-size constraint).

    When multiple bodies overlap a cell, their indicators sum
    (over-penalising) — the method assumes non-overlapping bodies, which
    is the physically meaningful case (pipe wall ∩ robot = ∅).
    """
    I_total = jnp.zeros((u.shape[0],), dtype=u.dtype)
    weighted_ub = jnp.zeros_like(u)
    for b in bodies:
        phi = b.sdf(x)
        I = smoothed_indicator(phi, eps)
        if b.u_body is None:
            ub = jnp.zeros_like(u)
        else:
            ub = b.u_body(x)
        I_total = I_total + I
        weighted_ub = weighted_ub + I[:, None] * ub
    # Effective body velocity is the indicator-weighted average
    u_body_eff = jnp.where(
        I_total[:, None] > 1e-30,
        weighted_ub / jnp.where(I_total[:, None] > 1e-30, I_total[:, None], 1.0),
        u,
    )
    decay = jnp.exp(-alpha * I_total * dt)        # [N_cells]
    return u_body_eff + (u - u_body_eff) * decay[:, None]


# ---------------------------------------------------------------------------
# Force / torque extraction (Newton's 3rd law)
# ---------------------------------------------------------------------------

def compute_ibm_forces(
    u: jnp.ndarray,                  # [N_cells, dim]
    x: jnp.ndarray,                  # [N_cells, dim]
    V: jnp.ndarray,                  # [N_cells]
    bodies: Iterable[IBMBody],
    *,
    alpha: float,
    eps: float,
    rho: float = 1.0,
    dt: float | None = None,
) -> dict:
    """Force / torque on every body marked ``extract_force=True``.

    For the **Goldstein-style** explicit IBM (small ``α dt``) the
    per-cell force on the body equals the per-cell penalty
    ``α · I · (u − u_body)`` so the integrated body force is

        F_body = ρ · ∫_V α I (u − u_body) dV.

    For the **Brinkman-style** implicit IBM with closed-form decay the
    per-step momentum sink is

        Δp/dt = ρ (u_new − u_before) / dt
              = ρ (u_body − u_before) (1 − exp(−α I dt)) / dt,

    so the integrated body force (Newton's 3rd) is

        F_body = ρ ∫_V (u − u_body) (1 − exp(−α I dt)) / dt · dV.

    For large ``α I dt`` the decay factor saturates to 1 and the formula
    reduces to ``ρ ∫_V (u − u_body) / dt · dV`` — bounded by ``dt``,
    independent of ``α``. *Pass ``dt`` to use this Brinkman-aware
    formula.*

    **Which velocity field to pass:** the *velocity that would have
    existed if the IBM weren't there* — i.e. the explicit-advection
    prediction *before* any Brinkman update touches it. PISO exposes
    this as the ``u_after_explicit`` field of the state pytree. If you
    accidentally pass ``u`` (post-everything) or ``u_pre_ibm``
    (post-projection but pre-post-Brinkman), the previous step's
    pre-Brinkman has already driven u → u_body inside the body and
    the (u − u_body) signal is gone, so the reported drag will be near
    zero.
    """
    out: dict = {}
    for b in bodies:
        if not b.extract_force:
            continue
        phi = b.sdf(x)
        I = smoothed_indicator(phi, eps)
        if b.u_body is None:
            ub = jnp.zeros_like(u)
        else:
            ub = b.u_body(x)
        # Per-cell force on the body
        if dt is None:
            f_per_cell = alpha * I[:, None] * (u - ub)
            Force = rho * jnp.sum(f_per_cell * V[:, None], axis=0)
        else:
            decay = jnp.exp(-alpha * I * dt)              # [N_cells]
            f_per_cell = (rho / dt) * (1.0 - decay)[:, None] * (u - ub)
            Force = jnp.sum(f_per_cell * V[:, None], axis=0)
        entry = {"force": Force}
        if b.ref_point is not None:
            r = x - b.ref_point
            if x.shape[-1] == 3:
                tau_cell = jnp.cross(r, f_per_cell)
            else:
                tau_cell = (
                    r[..., 0] * f_per_cell[..., 1]
                    - r[..., 1] * f_per_cell[..., 0]
                )[..., None]
            Torque = jnp.sum(tau_cell * V[:, None], axis=0)
            if dt is None:
                Torque = rho * Torque
            entry["torque"] = Torque
        out[b.name] = entry
    return out
