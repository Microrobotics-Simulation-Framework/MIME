"""Diffuse-penalty immersed boundary method (Peskin-style).

Two force-extraction methods are exposed:

* :func:`compute_ibm_forces` (legacy) — sums the per-cell Brinkman /
  Goldstein penalty momentum-sink. Captures the right SIGN of the
  drag but biases the magnitude low at moderate IBM resolution
  because the bulk momentum-sink representation under-weights the
  body-surface contribution where the actual hydrodynamic force lives.

* :func:`surface_integral_force` (preferred) — integrates the fluid
  Cauchy stress ``σ·n`` over a *shell of cells just outside the
  IBM body* (in clean fluid, past the diffuse Heaviside band). This
  is the standard surface-traction approach and is what the BEM /
  exact references compute.


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
from typing import Callable, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm.operators import grad_green_gauss


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


# ---------------------------------------------------------------------------
# Surface-integral force extraction (preferred for accuracy)
# ---------------------------------------------------------------------------

def surface_integral_force(
    u: jnp.ndarray,             # [N_cells, dim] cell-centred velocity
    p: jnp.ndarray,             # [N_cells]      cell-centred pressure
    mesh,                       # FVMMesh
    sdf_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    mu: float,
    dx: float,
    shell_inner: float = 0.5,
    shell_outer: float = 2.5,
    ref_point: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Drag (and optional torque) by surface integration of the fluid stress.

    The body is described by an SDF ``sdf_fn(x) -> phi(x)`` with the
    convention ``phi < 0`` inside, ``phi > 0`` outside. The integral

        F = ∮_S σ · n dA   with   σ = -p I + μ (∇u + ∇uᵀ)

    is approximated as a volume sum over a *shell* of cells where
    ``φ ∈ (shell_inner · dx, shell_outer · dx)`` — i.e. just outside
    the diffuse IBM band, in clean fluid. The shell-volume integral
    is converted to a surface integral by dividing by the shell
    thickness in φ space, ``Δφ_shell = (shell_outer − shell_inner) dx``.

    Parameters
    ----------
    u : ``[N_cells, dim]``
        Velocity field after the PISO step (the converged ``state['u']``,
        not ``u_after_explicit``).
    p : ``[N_cells]``
        Pressure field (state['p']).
    mesh : FVMMesh
    sdf_fn : Callable
        SDF, must be JAX-callable and differentiable for ∇φ via Green-Gauss
        (the analytical normal n = ∇φ/|∇φ| is used in the projection).
    mu : float
        Dynamic viscosity (= ρ · ν).
    dx : float
        Cell spacing (assumed isotropic). Used to scale shell thickness.
    shell_inner, shell_outer : float
        Shell location in φ units of ``dx``. Default (0.5, 2.5) — a 2-cell
        shell located 0.5 dx outside the body surface. Try (1, 3) and
        (0.5, 4) to check sensitivity; result should be robust if the
        shell sits in clean fluid.
    ref_point : ``[dim]`` or None
        Reference point for torque. None ⇒ no torque computed.

    Returns
    -------
    F : ``[dim]``
        Net hydrodynamic force on the body.
    T : ``[3]`` (3D) or ``[1]`` (2D) or None
        Net hydrodynamic torque about ``ref_point``, or None if not
        requested.

    Notes
    -----
    The quantity ``σ · n`` here is the traction the FLUID applies to
    the BODY at the surface (Cauchy convention: traction on the side
    that ``n`` points TOWARD, applied BY the side ``n`` points FROM —
    here ``n = ∇φ/|∇φ|`` points from body into fluid, so traction is
    fluid-on-body).

    For a Cartesian SDF (|∇φ| = 1) the area element is ``V_P /
    Δφ_shell``. For non-SDF implicit functions the |∇φ| factor enters
    naturally; we re-normalise n by |∇φ| anyway, so the formula handles
    both.
    """
    dim = mesh.dim
    phi = sdf_fn(mesh.x)                                # [N_cells]
    grad_phi = grad_green_gauss(phi, mesh)             # [N_cells, dim]
    norm_g = jnp.sqrt(jnp.sum(grad_phi ** 2, axis=-1) + 1e-30)
    n_hat = grad_phi / norm_g[:, None]                  # outward from body

    # Velocity gradient: grad_u[P, i, j] = ∂u_i/∂x_j (Green-Gauss on a
    # vector field returns shape [N_cells, k, dim] where k is the vector
    # component and the trailing dim is the spatial axis).
    grad_u = grad_green_gauss(u, mesh)                  # [N_cells, dim, dim]
    eps_strain = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
    sigma = (
        -p[:, None, None] * jnp.eye(dim, dtype=u.dtype)[None, :, :]
        + 2.0 * mu * eps_strain
    )                                                   # [N_cells, dim, dim]
    traction = jnp.einsum("Pij,Pj->Pi", sigma, n_hat)   # [N_cells, dim]

    shell_mask = (phi > shell_inner * dx) & (phi < shell_outer * dx)
    shell_thickness = (shell_outer - shell_inner) * dx
    weight = (mesh.V / shell_thickness) * shell_mask    # [N_cells]

    F = jnp.sum(traction * weight[:, None], axis=0)

    T = None
    if ref_point is not None:
        r = mesh.x - ref_point
        if dim == 3:
            tau_cell = jnp.cross(r, traction)
        else:
            tau_cell = (
                r[..., 0] * traction[..., 1]
                - r[..., 1] * traction[..., 0]
            )[..., None]
        T = jnp.sum(tau_cell * weight[:, None], axis=0)
    return F, T


# ---------------------------------------------------------------------------
# Momentum-deficit (control-volume) drag extraction
# ---------------------------------------------------------------------------

def momentum_deficit_drag(
    u: jnp.ndarray,                        # [N_cells, dim]
    p: jnp.ndarray,                        # [N_cells]
    mesh,                                   # FVMMesh
    *,
    sphere_centre: jnp.ndarray,             # [3]
    sphere_radius: float,                   # for the ±5a planes
    pipe_radius: float,                     # to mask out wall cells
    pipe_axis: int = 2,                     # 0=x, 1=y, 2=z
    rho: float = 1.0,
    margin_planes: float = 5.0,             # planes at z_sphere ± margin·a
    body_force: float = 0.0,                # uniform per-mass body force on this axis
    mu: float = 0.0,                         # dynamic viscosity (only needed
                                              # for periodic-z + body-force setup
                                              # to compute Hagen-Poiseuille wall shear)
) -> jnp.ndarray:
    """Drag on a static body in pipe flow via control-volume momentum balance.

    For an inflow/outflow setup, steady-state momentum balance:
        F_drag = (M_in − M_out) + (p̄_in − p̄_out) · A_pipe

    For a periodic-z body-force-driven setup, the body force adds
    momentum at rate ρ·f·V_CV between the planes, AND the pipe wall
    extracts momentum at the wall-shear rate. The momentum balance on
    the FLUID inside the CV between the two planes (excluding the
    sphere region):
        0 = (M_in − M_out) + (p̄_in − p̄_out) A_pipe
            + ρ f V_CV − F_wall_shear − F_drag

    rearranged for F_drag:
        F_drag = (M_in − M_out) + (p̄_in − p̄_out) A_pipe
                 + ρ f V_CV − F_wall_shear

    For Hagen-Poiseuille, F_wall_shear = 8πμU_mean · L_CV. We compute
    F_wall_shear using the LOCAL U_mean at the upstream plane (which
    represents the actual flow rate including sphere blockage). For
    inflow/outflow setups (body_force=0), the body-force and wall-shear
    terms drop out and the formula reduces to the standard form.

    The integration avoids the IBM body entirely → no Green-Gauss
    gradient contamination, unlike :func:`surface_integral_force`. This
    is the recommended extraction at moderate-to-high confinement
    (λ ≳ 0.15).

    Returns
    -------
    F_axis : float
        Drag force on the body along the ``pipe_axis`` direction.
    """
    if mesh.cartesian_shape is None:
        raise ValueError("momentum_deficit requires Cartesian mesh")
    shape = mesh.cartesian_shape
    spacing = mesh.cartesian_spacing
    dim = mesh.dim
    if dim != 3:
        raise NotImplementedError("momentum_deficit currently 3D only")

    # Find planes: z_sphere ± margin · a
    z_sphere = float(sphere_centre[pipe_axis])
    z_in = z_sphere - margin_planes * sphere_radius
    z_out = z_sphere + margin_planes * sphere_radius

    # Reshape to 3D
    u_3d = u.reshape(shape + (3,))
    p_3d = p.reshape(shape)
    x_3d = mesh.x.reshape(shape + (3,))

    # Get axial coordinate of each cell along pipe_axis
    axis_coords = x_3d[..., pipe_axis]
    # Find the cell index (along pipe_axis) closest to z_in / z_out
    # using the 1D coord vector.
    if pipe_axis == 0:
        coord_1d = x_3d[:, 0, 0, 0]
    elif pipe_axis == 1:
        coord_1d = x_3d[0, :, 0, 1]
    else:
        coord_1d = x_3d[0, 0, :, 2]
    iz_in = int(jnp.argmin(jnp.abs(coord_1d - z_in)))
    iz_out = int(jnp.argmin(jnp.abs(coord_1d - z_out)))

    # Pipe wall mask in cross-section (true = fluid; false = inside wall)
    cross_axes = [a for a in range(3) if a != pipe_axis]
    rho_xy_3d = jnp.sqrt(sum(x_3d[..., a] ** 2 for a in cross_axes))
    fluid_3d = rho_xy_3d < pipe_radius - spacing[0]   # exclude wall band

    # Cross-section area element
    dxa, dxb = (spacing[a] for a in cross_axes)
    dA = dxa * dxb

    def slab_quants(iz):
        # Take slab perpendicular to pipe_axis at index iz
        if pipe_axis == 0:
            u_slab = u_3d[iz, :, :, pipe_axis]
            p_slab = p_3d[iz, :, :]
            f_slab = fluid_3d[iz, :, :]
        elif pipe_axis == 1:
            u_slab = u_3d[:, iz, :, pipe_axis]
            p_slab = p_3d[:, iz, :]
            f_slab = fluid_3d[:, iz, :]
        else:
            u_slab = u_3d[:, :, iz, pipe_axis]
            p_slab = p_3d[:, :, iz]
            f_slab = fluid_3d[:, :, iz]
        f_slab_f = f_slab.astype(u.dtype)
        # Cross-section area (fluid only)
        A_fluid = jnp.sum(f_slab_f) * dA
        # Mass flux
        Q = jnp.sum(u_slab * f_slab_f) * dA
        # U_ref = mean velocity over fluid cross-section
        U_ref = Q / jnp.maximum(A_fluid, 1e-30)
        # Momentum-deficit integrand: ρ u (U_ref − u)
        deficit = rho * jnp.sum(u_slab * (U_ref - u_slab) * f_slab_f) * dA
        # Mean pressure over fluid section
        p_mean = jnp.sum(p_slab * f_slab_f) / jnp.maximum(jnp.sum(f_slab_f), 1e-30)
        return deficit, p_mean, A_fluid, U_ref, Q

    deficit_in, p_in, A_in, U_in, Q_in = slab_quants(iz_in)
    deficit_out, p_out, A_out, U_out, Q_out = slab_quants(iz_out)

    # Pressure force on the CV: (p_in - p_out) * A_pipe (averaged over fluid
    # area on each plane, multiplied by full pipe cross-section A_pipe).
    A_pipe = jnp.pi * pipe_radius ** 2
    F_pressure = (p_in - p_out) * A_pipe

    # Net momentum deficit: in - out
    F_momentum = deficit_in - deficit_out

    # Optional body-force + Hagen-Poiseuille-wall-shear corrections,
    # active when both ``body_force`` and ``mu`` are nonzero. Required
    # for a periodic-z body-force-driven setup, but the HP wall-shear
    # term is approximate (assumes the IBM cylinder wall matches an
    # ideal sharp wall, which it doesn't — the diffuse band shifts
    # the effective radius and biases this term). For best accuracy
    # use a true inflow/outflow setup (body_force=0) where the bare
    # momentum-deficit formula F = (M_in − M_out) + (P_in − P_out)·A
    # is exact.
    L_CV = jnp.abs(coord_1d[iz_out] - coord_1d[iz_in])
    V_CV = A_pipe * L_CV
    F_body = rho * body_force * V_CV
    F_wall = 8.0 * jnp.pi * mu * U_in * L_CV
    return F_momentum + F_pressure + F_body - F_wall
