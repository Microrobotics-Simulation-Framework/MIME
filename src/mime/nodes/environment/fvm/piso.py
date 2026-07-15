"""Transient PISO solver for incompressible Navier-Stokes (backward Euler).

Implements a PISO-style projection scheme (Issa 1986; Chorin 1968) on a
collocated face graph:

  * Explicit advection prediction (upwind/central blended)
  * Implicit diffusion via FFT/DST-diagonalised Helmholtz solve
    ``(I − ν dt ∇²) u* = u_n + dt · (RHS_explicit)``
  * Pressure projection to enforce continuity, with ``n_corrector``
    PISO passes per time step (default 2). Pressure correction Poisson
    is FFT-diagonalised and Rhie-Chow corrects the face flux.

The stiff diffusion term is solved exactly per time step (up to
discretisation error) so the scheme is unconditionally stable for the
diffusion part — necessary at low Reynolds numbers and small grid
spacings where Jacobi sub-iteration would converge intolerably slowly.
The advection treatment carries the usual CFL constraint.

The whole step lives inside ``jax.lax.fori_loop``/``jax.lax.scan`` and
is JIT-compiled by :func:`run_piso` and :func:`run_piso_with_history`.

References
----------
- Issa (1986) J. Comput. Phys. 62, 40–65.
- Chorin (1968) "Numerical solution of the Navier–Stokes equations",
  Math. Comp. 22.
- Versteeg & Malalasekera (2007) An Introduction to CFD §6.5.
- Moukalled, Mangani, Darwish (2016) The FVM in CFD §15.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from mime.nodes.environment.fvm.mesh import FVMMesh
from mime.nodes.environment.fvm.boundary import (
    VelocityBC, velocity_convection_boundaries,
    velocity_helmholtz_bc_source,
)
from mime.nodes.environment.fvm.operators import (
    grad_green_gauss,
    laplacian_orthogonal,
    convection_upwind_blend,
    divergence_face_flux,
    face_velocity_rhie_chow,
    momentum_diagonal_uniform_cartesian,
)
from mime.nodes.environment.fvm.pressure import (
    make_pressure_solver, make_helmholtz_solver,
    make_pressure_solver_fft, make_helmholtz_solver_fft,
    make_pressure_solver_iterative, make_helmholtz_solver_iterative,
)
from mime.nodes.environment.fvm.ibm import (
    IBMBody, ibm_brinkman_implicit_update, compute_ibm_forces,
)
from mime.nodes.environment.fvm.lifting import (
    LiftingFunction, compute_lifting_source,
)


@dataclass(frozen=True)
class PisoConfig:
    nu: float
    rho: float = 1.0
    gamma_conv: float = 0.5     # 0=upwind, 1=central
    n_corrector: int = 2        # PISO pressure corrector passes per step
    pressure_bc: str | tuple = "neumann"
    velocity_bc: str | tuple = "dirichlet"
    # IBM penalty parameters (only used when ibm_bodies are passed to step)
    ibm_alpha: float = 0.0
    # Half-width of the diffuse IBM band, in **physical length units** —
    # never rescaled by the mesh. See ``ibm_eps_cells`` below and
    # :func:`~mime.nodes.environment.fvm.ibm.smoothed_indicator`.
    #
    # ``ibm_eps`` IS PART OF THE GEOMETRY, not a numerical tolerance. The
    # body the flow actually feels is the ``eps``-wide Heaviside band, so
    # changing ``eps`` changes the effective body size and hence the drag.
    # Measured on the confined Stokes sphere (λ = 0.2, Re = 0.1) at a
    # FIXED cpr = 4, i.e. with dx and every other parameter held constant:
    #
    #     eps = 0.5 dx  ->  Cd = 168.4
    #     eps = 1.0 dx  ->  Cd = 258.0
    #     eps = 2.0 dx  ->  Cd = 500.1     (reference: 403)
    #
    # — a 3× swing that straddles the reference. So ``eps`` must be chosen
    # deliberately and reported alongside any drag number.
    #
    # THE TRAP: the idiom throughout this repo is ``ibm_eps = 1.0 * dx``,
    # which ties a *geometric* parameter to the mesh. Refining the mesh
    # then shrinks the body while it improves the discretisation, so a cpr
    # sweep varies geometry and accuracy together and is NOT a convergence
    # study — by construction, whatever it produces. For a convergence
    # study, pass an absolute ``ibm_eps`` (e.g. ``ibm_eps=0.025``) held
    # fixed while dx varies. That path needs no flag: ``ibm_eps`` has
    # always been physical.
    ibm_eps: float = 0.0
    # Opt-in: express the band as a multiple of dx and let the solver
    # resolve it against the mesh, instead of the caller computing
    # ``1.0 * dx`` by hand. Mutually exclusive with ``ibm_eps``. This
    # exists to make the resolution-tied choice *explicit and greppable*
    # rather than an accident of the call site — it does not change any
    # behaviour relative to passing ``ibm_eps = <cells> * dx`` yourself.
    ibm_eps_cells: Optional[float] = None
    # Backend for the diagonalised solvers:
    #   "dense" — dense matmul DCT/DST, O(N²) per axis. Best at N≤96
    #     on RTX 2060 (per R4-P4 measurement).
    #   "fft"   — cuFFT via jax.scipy.fft.dct, O(N log N). Worth using
    #     above the crossover (≈ 96-128 on RTX 2060; lower on H100).
    #   "auto"  — pick based on mesh.N_cells; threshold below.
    transform_backend: str = "auto"
    # R4-P4 measurement on RTX 2060: dense wins at ALL tested sizes
    # (32³ → 128³, fft/dense ratio 0.5-0.87). Crossover is above 128³,
    # beyond what fits on 6GB. The 256³ threshold defaults to dense
    # for everything on a small GPU but flips to FFT for large meshes
    # on H100-class hardware where the O(N log N) advantage matters.
    auto_fft_threshold_cells: int = 256 ** 3   # ≈ 16.8 M cells

    def resolve_ibm_eps(self, mesh: FVMMesh) -> float:
        """Physical half-width of the diffuse IBM band on ``mesh``.

        Returns :attr:`ibm_eps` unchanged (it is already physical) unless
        the caller opted into :attr:`ibm_eps_cells`, in which case the
        band is resolved against the mesh spacing. Every consumer of the
        band — the Brinkman update *and* the force extractor — must go
        through this, or they will disagree about the body's size.
        """
        if self.ibm_eps_cells is None:
            return float(self.ibm_eps)
        if self.ibm_eps:
            raise ValueError(
                "PisoConfig: ibm_eps and ibm_eps_cells are mutually "
                "exclusive (physical vs dx-relative band); got "
                f"ibm_eps={self.ibm_eps!r}, "
                f"ibm_eps_cells={self.ibm_eps_cells!r}."
            )
        if mesh.cartesian_spacing is None:
            raise ValueError(
                "PisoConfig.ibm_eps_cells needs a dx, but this mesh is not "
                "Cartesian. Pass ibm_eps in physical units instead."
            )
        return float(self.ibm_eps_cells) * float(mesh.cartesian_spacing[0])


def initial_state(mesh: FVMMesh) -> dict:
    dim = mesh.dim
    z = jnp.zeros((mesh.N_cells, dim), dtype=mesh.V.dtype)
    return {
        "u": z,
        "u_pre_ibm": z,
        "u_after_explicit": z,
        "p": jnp.zeros((mesh.N_cells,), dtype=mesh.V.dtype),
        "F": jnp.zeros((mesh.N_faces,), dtype=mesh.V.dtype),
        "t": jnp.asarray(0.0, dtype=mesh.V.dtype),
        "i_step": jnp.asarray(0, dtype=jnp.int32),
    }


def make_piso_step(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    cfg: PisoConfig,
    body_force_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ibm_bodies: list[IBMBody] | None = None,
    lifting: LiftingFunction | None = None,
):
    """Construct a JIT-compatible PISO step.

    ``body_force_fn(t)`` returns either a ``[dim]`` (uniform body force)
    or ``[N_cells, dim]`` (spatially varying) array. ``None`` ⇒ no body
    force.

    ``ibm_bodies`` is an optional list of :class:`IBMBody`. The IBM
    penalty is applied via the closed-form Brinkman update before the
    Helmholtz diffusion solve and after the projection correction —
    this preserves the no-slip enforcement that the projection step
    might otherwise smear.

    ``lifting`` is an optional :class:`LiftingFunction` that decomposes
    the velocity into ``u = u_hom + u_lift``. The PISO loop then
    evolves ``u_hom`` (which has homogeneous boundary conditions and
    matches the DST/DCT Helmholtz spectral basis exactly), and the
    physical velocity ``u_phys = u_hom + u_lift`` is reconstructed for
    IBM force extraction and for the user-facing ``state["u_pre_ibm"]``
    and ``state["u_after_explicit"]`` slots.

      * ``state["u"]`` always stores ``u_hom``.
      * IBM Brinkman pushes ``u_phys`` to the body velocity (zero for
        a static body), then we subtract ``u_lift`` to get the new
        ``u_hom`` — i.e. inside the body ``u_hom ≈ -u_lift``.
      * Inlet ``VelocityBC`` should be passed with ``u_wall = 0``
        (homogeneous) when ``lifting`` is provided. The non-zero inlet
        velocity is enforced *implicitly* by the lift.

    Returns ``step(state, dt)`` advancing one time step.
    """
    mu = cfg.rho * cfg.nu
    bF, bphi = velocity_convection_boundaries(mesh, bcs)
    bF_rho = {k: cfg.rho * v for k, v in bF.items()}
    dtype = mesh.V.dtype

    # Boundary source for the implicit diffusion solve. The diagonalised
    # Helmholtz solvers below are constructed from the BC *type*
    # (cfg.velocity_bc) alone and bake u = 0 into the basis at every
    # Dirichlet face, so both ``VelocityBC.u_wall`` (boundary data) and any
    # per-patch type that disagrees with its axis' string would otherwise be
    # dropped silently. See velocity_helmholtz_bc_source: ``bc_const`` is
    # exact, ``bc_lag`` is a deferred correction that is non-zero only on a
    # patch/axis type mismatch. Both vanish identically for homogeneous
    # no-slip walls and for the lifting decomposition.
    bc_const, bc_lag = velocity_helmholtz_bc_source(mesh, bcs, cfg.velocity_bc)
    bc_const = bc_const.astype(dtype)
    bc_lag = bc_lag.astype(dtype)

    ibm_eps = cfg.resolve_ibm_eps(mesh)

    backend = cfg.transform_backend
    if backend == "auto":
        # The FFT/dense diagonalised solvers require a Cartesian grid; an
        # unstructured / body-fitted mesh (no cartesian_shape) can only use the
        # matrix-free iterative path.
        if mesh.cartesian_shape is None:
            backend = "iterative"
        else:
            backend = "fft" if mesh.N_cells >= cfg.auto_fft_threshold_cells else "dense"
    if backend == "fft":
        pressure_solver = make_pressure_solver_fft(mesh, bc=cfg.pressure_bc)
        helmholtz_solver = make_helmholtz_solver_fft(mesh, bc=cfg.velocity_bc)
    elif backend == "dense":
        pressure_solver = make_pressure_solver(mesh, bc=cfg.pressure_bc)
        helmholtz_solver = make_helmholtz_solver(mesh, bc=cfg.velocity_bc)
    elif backend == "iterative":
        pressure_solver = make_pressure_solver_iterative(mesh, bc=cfg.pressure_bc)
        helmholtz_solver = make_helmholtz_solver_iterative(mesh, bc=cfg.velocity_bc)
    else:
        raise ValueError(f"transform_backend={cfg.transform_backend!r}")

    def step(state, dt):
        u_n = state["u"].astype(dtype)              # u_hom when lifting given
        p_n = state["p"].astype(dtype)
        F_n = state["F"].astype(dtype)
        t_n = state["t"].astype(dtype)
        t_next = t_n + dt
        i_step = state.get("i_step", jnp.asarray(0, dtype=jnp.int32))

        if body_force_fn is None:
            body = jnp.zeros_like(u_n)
        else:
            body = body_force_fn(t_next).astype(dtype)
            if body.ndim == 1:
                body = jnp.broadcast_to(body[None, :], u_n.shape)
            elif body.shape != u_n.shape:
                body = jnp.broadcast_to(body, u_n.shape)

        # ---- Lifting: f_lift body force, u_lift snapshot ----
        if lifting is None:
            u_lift = jnp.zeros_like(u_n)
            f_lift = jnp.zeros_like(u_n)
        elif lifting.omega > 0.0 and lifting.U_re is not None:
            # Analytical Womersley: reconstruct on the fly
            wt = lifting.omega * t_next
            cwt = jnp.cos(wt).astype(dtype)
            swt = jnp.sin(wt).astype(dtype)
            u_lift = (lifting.u_lift_static.astype(dtype)
                      + cwt * lifting.U_re.astype(dtype)
                      - swt * lifting.U_im.astype(dtype))
            du_lift_dt_ana = (- lifting.omega * swt * lifting.U_re.astype(dtype)
                              - lifting.omega * cwt * lifting.U_im.astype(dtype))
            f_lift = compute_lifting_source(
                u_n, u_lift, du_lift_dt_ana,
                None, None, mesh, nu=cfg.nu,
            ).astype(dtype)
        else:
            if lifting.is_time_varying:
                # Modulo-index a periodic table (cycles repeat).
                table_len = lifting.u_lift_static.shape[0]
                idx = i_step % jnp.asarray(table_len, dtype=jnp.int32)
                u_lift_ = jnp.take(lifting.u_lift_static, idx, axis=0)
                du_lift_dt_ = jnp.take(lifting.du_lift_dt, idx, axis=0)
                # Face/grad tables may be empty placeholders → None
                if lifting.u_lift_face.shape[1] == 0:
                    u_lift_face_ = None
                else:
                    u_lift_face_ = jnp.take(lifting.u_lift_face, idx, axis=0).astype(dtype)
                if lifting.grad_u_lift.shape[1] == 0:
                    grad_u_lift_ = None
                else:
                    grad_u_lift_ = jnp.take(lifting.grad_u_lift, idx, axis=0).astype(dtype)
            else:
                u_lift_ = lifting.u_lift_static
                du_lift_dt_ = lifting.du_lift_dt
                u_lift_face_ = lifting.u_lift_face.astype(dtype)
                grad_u_lift_ = lifting.grad_u_lift.astype(dtype)
            u_lift = u_lift_.astype(dtype)
            f_lift = compute_lifting_source(
                u_n, u_lift, du_lift_dt_.astype(dtype),
                u_lift_face_, grad_u_lift_,
                mesh, nu=cfg.nu,
            ).astype(dtype)

        # ---- 1. Explicit advection acceleration ----
        rhoF = cfg.rho * F_n
        conv = convection_upwind_blend(
            u_n, rhoF, mesh,
            gamma=cfg.gamma_conv,
            boundary_phi=bphi,
            boundary_F=bF_rho,
        )                                        # [N_cells, dim]

        grad_p = grad_green_gauss(p_n, mesh)
        # Body force in x-momentum is per unit mass (m/s²) — multiply by V*ρ to
        # get the same units as conv/diff/(V grad p).
        # RHS for the implicit diffusion solve, divided by the (1 - α∇²)
        # operator: u_pred = u_n + dt * (-conv/V/ρ + body − grad_p/ρ + f_lift)
        accel_explicit = (
            -conv / (cfg.rho * mesh.V[:, None])
            + body
            - grad_p / cfg.rho
            + f_lift
        )
        u_pred = u_n + dt * accel_explicit       # u_hom prediction

        # ---- 2a. IBM Brinkman pre-step (closed-form implicit) ----
        # Save the explicit-advection prediction *before* any Brinkman
        # has touched it. The IBM-force extractor consumes
        # ``u_after_explicit`` in the *physical* frame, so we add
        # ``u_lift`` here.
        u_after_explicit = u_pred + u_lift
        if ibm_bodies:
            # Brinkman acts on physical velocity (push u_phys → u_body=0)
            u_phys_pred = u_pred + u_lift
            u_phys_pred = ibm_brinkman_implicit_update(
                u_phys_pred, mesh.x, ibm_bodies,
                alpha=cfg.ibm_alpha, eps=ibm_eps, dt=dt,
            )
            u_pred = u_phys_pred - u_lift     # back to u_hom

        # ---- 2b. Implicit diffusion via Helmholtz ----
        # (I − ν dt ∇²) u* = u_pred ; the Helmholtz operator's BCs (DST
        # for Dirichlet wall, etc.) bake the *homogeneous* wall condition
        # into the basis, so u = 0 walls are handled exactly. Non-zero wall
        # data (and any patch/axis BC-type mismatch) is restored by adding
        # the difference between the physical and basis boundary fluxes:
        #   (I − α∇²) u = u_pred ⟺ (I − α∇²_s) u = u_pred + α(const + lag·u).
        alpha = cfg.nu * dt
        u_bc_src = bc_const + bc_lag[:, None] * u_pred
        u_star = helmholtz_solver(u_pred + alpha * u_bc_src, alpha)

        # ---- 3. PISO pressure correction passes ----
        # Effective momentum diagonal for the projection step is ρV/dt
        # (since the implicit diffusion has been absorbed into the
        # Helmholtz inverse). Rhie-Chow's D_face = V/a_p reduces to
        # dt/ρ — uniform on Cartesian, exactly the "fast Poisson"
        # choice (Brown, Cortez & Minion 2001).
        # We compute Rhie-Chow inline here (instead of calling
        # face_velocity_rhie_chow) to avoid a 12M-element static
        # gather a_p_cell[mesh.owner] that XLA would otherwise
        # constant-fold (multi-second compile cost at high N).
        D_bar = dt / cfg.rho
        n_hat_face = mesh.d / mesh.d_mag[:, None]   # owner → neighbour unit

        u_curr = u_star
        p_curr = p_n
        F_curr = F_n
        for _ in range(cfg.n_corrector):
            grad_p_curr = grad_green_gauss(p_curr, mesh)
            # Inline Rhie-Chow with uniform D_face = D_bar:
            #   u_f = avg(u_o, u_n) − D_bar * [Δp/|d| − avg(∇p) · n̂] n̂
            u_o = u_curr[mesh.owner]
            u_n = u_curr[mesh.neighbour]
            u_avg = 0.5 * (u_o + u_n)
            grad_p_avg = 0.5 * (
                grad_p_curr[mesh.owner] + grad_p_curr[mesh.neighbour]
            )
            dp = p_curr[mesh.neighbour] - p_curr[mesh.owner]
            grad_p_along = jnp.einsum("fd,fd->f", grad_p_avg, n_hat_face)
            corr = D_bar * (dp / mesh.d_mag - grad_p_along)
            u_face = u_avg - corr[:, None] * n_hat_face
            F_star = jnp.einsum("fd,fd->f", u_face, mesh.Sf)

            div_F = divergence_face_flux(F_star, mesh, boundary_F=bF)
            rhs_p = div_F / D_bar
            p_prime = pressure_solver(rhs_p)
            p_prime = p_prime - jnp.mean(p_prime)

            p_curr = p_curr + p_prime
            grad_pp = grad_green_gauss(p_prime, mesh)
            u_curr = u_curr - D_bar * grad_pp

            dpp = p_prime[mesh.neighbour] - p_prime[mesh.owner]
            F_curr = F_star - D_bar * mesh.area / mesh.d_mag * dpp

        # ---- 4. IBM Brinkman post-step ----
        # Re-enforce no-slip on body cells after the projection step.
        # We store u_curr (pre-post-Brinkman) as ``u_pre_ibm`` so that
        # downstream force extraction can read the IBM penalty density
        # from the *unsuppressed* field (otherwise the post-Brinkman
        # decay zeros out the diffuse band that contributes the drag).
        u_pre_ibm = u_curr + u_lift               # physical frame for output
        if ibm_bodies:
            u_phys_curr = u_curr + u_lift
            u_phys_curr = ibm_brinkman_implicit_update(
                u_phys_curr, mesh.x, ibm_bodies,
                alpha=cfg.ibm_alpha, eps=ibm_eps, dt=dt,
            )
            u_curr = u_phys_curr - u_lift          # back to u_hom

        return {
            "u": u_curr.astype(dtype),               # u_hom (or u when lifting=None)
            "u_pre_ibm": u_pre_ibm.astype(dtype),    # u_phys (or u when lifting=None)
            "u_after_explicit": u_after_explicit.astype(dtype),  # u_phys
            "p": p_curr.astype(dtype),
            "F": F_curr.astype(dtype),
            "t": t_next.astype(dtype),
            "i_step": (i_step + 1).astype(jnp.int32),
        }

    return step


def run_piso(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    cfg: PisoConfig,
    *,
    n_steps: int,
    dt: float,
    body_force_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ibm_bodies: list[IBMBody] | None = None,
    lifting: LiftingFunction | None = None,
    initial: dict | None = None,
) -> dict:
    """Advance ``n_steps`` PISO time steps. JITed via ``jax.lax.fori_loop``."""
    if initial is None:
        initial = initial_state(mesh)
    step = make_piso_step(
        mesh, bcs, cfg,
        body_force_fn=body_force_fn,
        ibm_bodies=ibm_bodies,
        lifting=lifting,
    )

    @jax.jit
    def _run(state):
        return jax.lax.fori_loop(0, n_steps, lambda i, s: step(s, dt), state)

    return _run(initial)


def run_piso_with_history(
    mesh: FVMMesh,
    bcs: Dict[str, VelocityBC],
    cfg: PisoConfig,
    *,
    n_steps: int,
    dt: float,
    body_force_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ibm_bodies: list[IBMBody] | None = None,
    lifting: LiftingFunction | None = None,
    initial: dict | None = None,
    sample_every: int = 1,
) -> tuple[dict, dict]:
    """Like :func:`run_piso` but also collect velocity history.

    Returns ``(final_state, history)`` where ``history`` is a dict of
    arrays of shape ``[n_samples, ...]``.
    """
    if initial is None:
        initial = initial_state(mesh)
    step = make_piso_step(
        mesh, bcs, cfg,
        body_force_fn=body_force_fn,
        ibm_bodies=ibm_bodies,
        lifting=lifting,
    )
    n_samples = n_steps // sample_every

    @jax.jit
    def _run(state):
        def stride_body(s, i):
            for _ in range(sample_every):
                s = step(s, dt)
            sample = {"u": s["u"], "p": s["p"], "t": s["t"]}
            return s, sample

        final, hist = jax.lax.scan(stride_body, state, jnp.arange(n_samples))
        return final, hist

    return _run(initial)
