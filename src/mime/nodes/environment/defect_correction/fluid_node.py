"""DefectCorrectionFluidNode — confined drag via defect correction.

Couples a regularised Stokeslet BEM (near-field body drag) with an LBM
(far-field vessel wall interaction) via immersed boundary force spreading
and analytical free-space subtraction. Iterates traction and wall
correction to self-consistency.

External interface identical to StokesletFluidNode:
    Inputs:  body_velocity, body_angular_velocity, body_orientation
    Outputs: drag_force, drag_torque

See docs/algorithm_guide/defect_correction.md for the full method.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from maddening.core.node import SimulationNode, BoundaryInputSpec, BoundaryFluxSpec

from mime.nodes.environment.lbm.d3q19 import (
    lbm_step_split,
    init_equilibrium,
    equilibrium,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
)
from mime.nodes.environment.lbm.immersed_boundary import (
    precompute_ib_stencil,
    spread_forces,
    interpolate_velocity,
)
from mime.nodes.environment.stokeslet.bem import (
    assemble_system_matrix,
    compute_force_torque,
)
from mime.nodes.environment.stokeslet.surface_mesh import sphere_surface_mesh
from mime.nodes.environment.stokeslet.flow_field import evaluate_velocity_field
from mime.nodes.environment.defect_correction.wall_correction import (
    compute_wall_correction,
)

logger = logging.getLogger(__name__)


class DefectCorrectionFluidNode(SimulationNode):
    """Confined drag via defect correction: BEM + IB-LBM wall correction.

    Parameters
    ----------
    name : str
    timestep : float
        Physical timestep [s].
    mu : float
        Dynamic viscosity [Pa·s].
    rho : float
        Fluid density [kg/m³].
    body_mesh : SurfaceMesh
        BEM body surface mesh.
    body_radius : float
        Characteristic body radius [m] (for eval sphere sizing).
    vessel_radius : float
        Vessel inner radius [m].
    dx : float
        LBM lattice spacing [m].
    n_lbm_spinup : int
        LBM steps for initial spin-up (first call).
    n_lbm_warmstart : int
        LBM steps per defect correction iteration (warm-start).
    max_defect_iter : int
        Maximum defect correction iterations for translation.
    alpha : float or None
        Under-relaxation parameter. None = auto from wall effect estimate.
    tol : float
        Convergence tolerance (relative drag change).
    open_bc_axis : int
        Axis for open (pressure) BCs on LBM.
    epsilon : float or None
        BEM regularisation. None = body_mesh.mean_spacing / 2.
    eval_radii_factors : tuple of float
        Eval sphere radii as multiples of body_radius.
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        mu: float,
        rho: float,
        body_mesh,
        body_radius: float,
        vessel_radius: float,
        dx: float,
        lbm_convergence_tol: float = 1e-3,
        lbm_max_spinup: int = 50000,
        lbm_max_warmstart: int = 50000,
        lbm_check_interval: int = 100,
        max_defect_iter: int = 25,
        alpha: float | None = None,
        tol: float = 0.005,
        open_bc_axis: int = 2,
        epsilon: float | None = None,
        wall_correction_method: str = "auto",
        eval_radii_factors: tuple[float, ...] = (1.15, 1.2, 1.3),
        eval_radii_factors_all: tuple[float, ...] = (1.15, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0),
        repr_radius_factor: float = 1.5,
        use_pallas: bool = True,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu, **kwargs)

        self._mu = mu
        self._rho = rho
        self._body_radius = body_radius
        self._use_pallas = use_pallas
        self._wall_correction_method = wall_correction_method
        self._lbm_conv_tol = lbm_convergence_tol
        self._lbm_max_spinup = lbm_max_spinup
        self._lbm_max_warmstart = lbm_max_warmstart
        self._lbm_check_interval = lbm_check_interval
        self._max_defect_iter = max_defect_iter
        self._tol = tol
        self._open_bc_axis = open_bc_axis

        N_b = body_mesh.n_points
        self._N_body = N_b

        if epsilon is None:
            epsilon = body_mesh.mean_spacing / 2.0
        self._epsilon = epsilon

        # ── BEM: body-only system, LU factorized ────────────────────
        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)
        self._body_pts = body_pts
        self._body_wts = body_wts

        logger.info("Assembling body-only BEM system (N=%d, ε=%.4f)...",
                     N_b, epsilon)
        A = assemble_system_matrix(body_pts, body_wts, epsilon, mu)
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        # ── LBM grid ────────────────────────────────────────────────
        domain_extent = 2.5 * vessel_radius
        N_lbm = int(np.ceil(domain_extent / dx))
        N_lbm = ((N_lbm + 7) // 8) * 8
        self._nx = self._ny = self._nz = N_lbm
        self._dx = dx
        self._tau = 0.8
        nu_lu = (self._tau - 0.5) / 3.0
        nu_phys = mu / rho
        self._dt_lbm = nu_lu * dx**2 / nu_phys

        # Unit conversion (frozen constants)
        self._force_conv = self._dt_lbm**2 / (rho * dx**4)
        self._vel_conv = dx / self._dt_lbm  # lattice → physical: u_phys = u_lu * vel_conv

        # ── Pipe wall mask (frozen at init) ─────────────────────────
        vessel_R_lu = vessel_radius / dx
        cx, cy = N_lbm / 2.0, N_lbm / 2.0
        ix = jnp.arange(N_lbm, dtype=jnp.float32)
        iy = jnp.arange(N_lbm, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        dist_2d = jnp.sqrt((gx - cx)**2 + (gy - cy)**2)
        self._pipe_wall = jnp.broadcast_to(
            (dist_2d >= vessel_R_lu)[..., None],
            (N_lbm, N_lbm, N_lbm),
        )
        self._pipe_missing = compute_missing_mask(self._pipe_wall)

        # ── IB stencil: body surface → LBM grid ────────────────────
        body_pts_lu = np.array(body_mesh.points) / dx + np.array([N_lbm / 2] * 3)
        self._body_pts_lu = body_pts_lu
        ib_idx, ib_wts = precompute_ib_stencil(body_pts_lu, (N_lbm, N_lbm, N_lbm))
        self._ib_idx = jnp.array(ib_idx)
        self._ib_wts = jnp.array(ib_wts)

        # ── Eval sphere stencils (frozen at init) ───────────────────
        # Close radii (for Richardson)
        self._eval_stencils_close = []
        d_vals_close = []
        for R_factor in eval_radii_factors:
            R_ev = R_factor * body_radius
            if R_ev >= vessel_radius - dx:
                continue
            ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
            ev_lu = ev_mesh.points / dx + np.array([N_lbm / 2] * 3)
            ei, ew = precompute_ib_stencil(ev_lu, (N_lbm, N_lbm, N_lbm))
            self._eval_stencils_close.append({
                'pts_phys': jnp.array(ev_mesh.points),
                'idx': jnp.array(ei),
                'wts': jnp.array(ew),
            })
            d_vals_close.append(R_factor - 1.0)

        # All radii (for Lamb)
        self._eval_stencils_all = []
        R_phys_all = []
        for R_factor in eval_radii_factors_all:
            R_ev = R_factor * body_radius
            if R_ev >= vessel_radius - dx:
                continue
            ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
            ev_lu = ev_mesh.points / dx + np.array([N_lbm / 2] * 3)
            ei, ew = precompute_ib_stencil(ev_lu, (N_lbm, N_lbm, N_lbm))
            self._eval_stencils_all.append({
                'pts_phys': jnp.array(ev_mesh.points),
                'idx': jnp.array(ei),
                'wts': jnp.array(ew),
            })
            R_phys_all.append(R_ev)

        # Representation formula control surface
        R_repr = repr_radius_factor * body_radius
        repr_mesh = sphere_surface_mesh(radius=R_repr, n_refine=3)
        repr_lu = repr_mesh.points / dx + np.array([N_lbm / 2] * 3)
        ri, rw = precompute_ib_stencil(repr_lu, (N_lbm, N_lbm, N_lbm))
        repr_normals = repr_mesh.points / np.linalg.norm(
            repr_mesh.points, axis=1, keepdims=True)

        # Pack eval data for dispatch
        self._eval_data = {
            'stencils_close': self._eval_stencils_close,
            'd_vals_close': jnp.array(d_vals_close),
            'stencils_all': self._eval_stencils_all,
            'R_phys_all': jnp.array(R_phys_all),
            'repr_mesh': {
                'pts_phys': jnp.array(repr_mesh.points),
                'normals': jnp.array(repr_normals),
                'weights': jnp.array(repr_mesh.weights),
            },
            'repr_stencil': {
                'idx': jnp.array(ri),
                'wts': jnp.array(rw),
            },
            'repr_f_stencil': {
                'idx': jnp.array(ri),
                'wts': jnp.array(rw),
            },
        }

        # ── Convergence check stencil (middle eval radius, ~R=1.5a) ──
        # Used by _run_lbm_until_converged to monitor wall signal arrival
        if len(self._eval_stencils_all) >= 4:
            # Use the 4th eval radius (~1.5a if using the default set)
            self._convergence_check_stencil = self._eval_stencils_all[3]
        elif len(self._eval_stencils_close) > 0:
            self._convergence_check_stencil = self._eval_stencils_close[-1]
        else:
            self._convergence_check_stencil = self._eval_stencils_all[0]

        # ── Auto-relaxation parameter ───────────────────────────────
        if alpha is None:
            # α=0.3 is stable for all directions with inline Richardson.
            # The Richardson extrapolation corrects the geometric bias,
            # so higher α is safe (unlike the polynomial approach).
            self._alpha = 0.3
        else:
            self._alpha = alpha

        # ── Velocity field stash for FluidFieldProvider ─────────────
        self._latest_velocity = None
        self._latest_traction = None
        self._first_call = True

        logger.info(
            "DefectCorrectionFluidNode: %d³ LBM, %d body pts, %d eval radii, "
            "α=%.2f, max_iter=%d",
            N_lbm, N_b, len(self._eval_stencils_all), self._alpha, max_defect_iter,
        )

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        N = self._nx
        return {
            "f": init_equilibrium(N, N, N),
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Body angular velocity [rad/s]",
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,), default=jnp.zeros(3),
                description="Body translational velocity [m/s]",
            ),
            "body_orientation": BoundaryInputSpec(
                shape=(4,), default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Body orientation quaternion",
            ),
        }

    def _lbm_full_step(self, f, force):
        """One complete LBM step: collision + Guo forcing + streaming + BB + open BCs.

        Backend selection (in priority order):
        1. "triton": Two Triton kernels (collision + stream/BB) + JAX open BCs.
           Compiles in <1s on Ampere/Hopper. Fastest.
        2. "gather": JAX gather-based (pallas_lbm). Compiles in ~2s locally,
           60+ min on H100 via XLA. Fallback when Triton unavailable.
        3. "rolls": Original JAX roll-based. Slowest compilation.

        Signature: (f: [nx,ny,nz,19], force: [nx,ny,nz,3]) -> (f, u)
        """
        if self._use_pallas:
            # Try Triton first (instant compilation)
            try:
                from mime.nodes.environment.lbm.triton_kernels import (
                    lbm_full_step_triton, TRITON_AVAILABLE,
                )
                if TRITON_AVAILABLE:
                    return lbm_full_step_triton(
                        f, force, self._tau,
                        self._pipe_wall, self._pipe_missing,
                        self._open_bc_axis,
                    )
            except (ImportError, RuntimeError):
                pass

            # Fall back to JAX gather-based
            from mime.nodes.environment.lbm.pallas_lbm import lbm_full_step_pallas
            return lbm_full_step_pallas(
                f, force, self._tau,
                self._pipe_wall, self._pipe_missing,
                self._open_bc_axis,
            )
        else:
            f_pre, f_post, rho, u = lbm_step_split(f, self._tau, force=force)
            f = apply_bounce_back(
                f_post, f_pre, self._pipe_missing, self._pipe_wall,
            )
            f = self._apply_open_bc(f, rho)
            return f, u

    def _apply_open_bc(self, f, rho):
        """Open (pressure) BCs on axial faces."""
        axis = self._open_bc_axis
        rho_0 = 1.0

        if axis == 0:
            f = f.at[-1, :, :, :].set(f[-2, :, :, :])
            f_eq = equilibrium(
                jnp.full(f.shape[1:3], rho_0),
                jnp.zeros((*f.shape[1:3], 3)),
            )
            f = f.at[0, :, :, :].set(f_eq)
        elif axis == 1:
            f = f.at[:, -1, :, :].set(f[:, -2, :, :])
            f_eq = equilibrium(
                jnp.full((f.shape[0], f.shape[2]), rho_0),
                jnp.zeros((f.shape[0], f.shape[2], 3)),
            )
            f = f.at[:, 0, :, :].set(f_eq)
        elif axis == 2:
            f = f.at[:, :, -1, :].set(f[:, :, -2, :])
            f_eq = equilibrium(
                jnp.full(f.shape[0:2], rho_0),
                jnp.zeros((*f.shape[0:2], 3)),
            )
            f = f.at[:, :, 0, :].set(f_eq)

        return f

    def _bem_solve(self, rhs):
        """BEM backsubstitution using precomputed LU factors."""
        lu = jnp.array(self._lu)
        piv = jnp.array(self._piv)
        return jax.scipy.linalg.lu_solve((lu, piv), rhs)

    def _spread_traction(self, traction):
        """Convert BEM traction to LBM force field via IB spreading."""
        point_forces = traction * self._body_wts[:, None] * self._force_conv
        N = self._nx
        return spread_forces(point_forces, self._ib_idx, self._ib_wts, (N, N, N))

    def _run_lbm_until_converged(self, f_lbm, force_field, traction, max_steps,
                                  cold_start=False):
        """Run LBM until wall correction Δu at eval sphere stabilizes.

        Monitors Δu (LBM velocity minus BEM free-space Stokeslet) at
        a single eval sphere.

        For cold starts (first run from equilibrium), enforces a minimum
        step count based on the vessel-scale diffusion time. For warm
        starts (traction updated slightly), no minimum — the flow is
        already near steady state.

        Parameters
        ----------
        traction : (N_body, 3) current BEM traction for free-space subtraction
        cold_start : bool — if True, enforce diffusion-time minimum steps
        """
        check = self._lbm_check_interval
        tol = self._lbm_conv_tol
        du_prev = None

        if cold_start:
            nu_lu = (self._tau - 0.5) / 3.0
            vessel_R_lu = self._nx * 0.4
            min_steps = max(int(vessel_R_lu**2 / (2 * nu_lu)), 500)
        else:
            min_steps = 200  # warm-start: flow is already close

        es = self._convergence_check_stencil

        for step in range(max_steps):
            f_lbm, u_lbm = self._lbm_full_step(f_lbm, force_field)

            if (step + 1) % check == 0 and (step + 1) >= min_steps:
                u_w = interpolate_velocity(
                    u_lbm, es['idx'], es['wts'],
                ) * self._dx / self._dt_lbm
                u_fs = evaluate_velocity_field(
                    es['pts_phys'], self._body_pts, self._body_wts,
                    traction, self._epsilon, self._mu,
                )
                du = jnp.mean(u_w - u_fs, axis=0)

                if du_prev is not None:
                    change = float(jnp.linalg.norm(du - du_prev))
                    mag = float(jnp.linalg.norm(du)) + 1e-30
                    rel = change / mag
                    if rel < tol:
                        logger.info(
                            "    LBM converged in %d steps (min=%d, ΔΔu/Δu=%.1e)",
                            step + 1, min_steps, rel,
                        )
                        return f_lbm, u_lbm, step + 1

                du_prev = du

        logger.info("    LBM max steps (%d, min=%d) reached", max_steps, min_steps)
        return f_lbm, u_lbm, max_steps

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Compute confined drag via defect correction iteration."""
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))

        center = jnp.zeros(3)
        N_b = self._N_body

        # Body velocity at surface points
        r = self._body_pts - center
        u_body = U + jnp.cross(omega, r)

        # Step 1: BEM body-only solve (free-space traction)
        rhs = u_body.ravel()
        traction = self._bem_solve(rhs).reshape(N_b, 3)

        # Step 2: IB spread → walled LBM → defect correction iteration
        f_lbm = state["f"]
        force_field = self._spread_traction(traction)

        is_cold = self._first_call
        max_steps = self._lbm_max_spinup if is_cold else self._lbm_max_warmstart
        self._first_call = False
        f_lbm, u_lbm, _ = self._run_lbm_until_converged(
            f_lbm, force_field, traction, max_steps, cold_start=is_cold)

        # Defect correction iterations
        method = self._wall_correction_method
        col_method = "richardson" if method == "auto" else method

        for iteration in range(self._max_defect_iter):
            delta_u = compute_wall_correction(
                col_method, u_lbm, traction,
                self._body_pts, self._body_wts,
                self._eval_data,
                self._epsilon, self._mu,
                self._dx, self._dt_lbm,
                f_dist=f_lbm,
                tau=self._tau,
                rho_phys=self._rho,
                motion_axis=2,
                body_radius=self._body_radius,
            )

            if delta_u.ndim == 1:
                delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
            else:
                delta_u_body = delta_u

            rhs_corrected = (u_body - delta_u_body).ravel()
            traction_new = self._bem_solve(rhs_corrected).reshape(N_b, 3)

            traction = (1 - self._alpha) * traction + self._alpha * traction_new

            # Update LBM with new traction (warm-start — no min_steps)
            force_field = self._spread_traction(traction)
            f_lbm, u_lbm, _ = self._run_lbm_until_converged(
                f_lbm, force_field, traction, self._lbm_max_warmstart, cold_start=False,
            )

            # Convergence check
            F, T = compute_force_torque(
                self._body_pts, self._body_wts, traction, center,
            )
            drag_mag = jnp.linalg.norm(jnp.concatenate([F, T]))
            F_new, T_new = compute_force_torque(
                self._body_pts, self._body_wts, traction_new, center,
            )
            drag_mag_new = jnp.linalg.norm(jnp.concatenate([F_new, T_new]))
            rel_change = jnp.abs(drag_mag_new - drag_mag) / (jnp.abs(drag_mag) + 1e-30)

            if float(rel_change) < self._tol:
                break

        # Extract final drag
        F, T = compute_force_torque(
            self._body_pts, self._body_wts, traction, center,
        )

        self._latest_velocity = u_lbm
        self._latest_traction = traction

        return {
            "f": f_lbm,
            "drag_force": F,
            "drag_torque": T,
        }

    def compute_resistance_matrix(self, state: dict) -> np.ndarray:
        """Compute full 6x6 confined resistance matrix.

        Batches 6 unit motions. Rotation columns (3-5) get 1 defect
        correction pass. Translation columns (0-2) get up to
        max_defect_iter passes.

        NOTE: After the batched spinup, rotation LBM states become stale
        during translation iterations. Do NOT reuse rotation LBM states
        for flow visualization. Re-run _lbm_full_step with converged
        traction if visualization is needed.

        Returns
        -------
        R : (6, 6) resistance matrix in physical units
        """
        center = jnp.zeros(3)
        e = jnp.eye(3)
        N_b = self._N_body
        N = self._nx

        R = np.zeros((6, 6))

        for col in range(6):
            U = e[col] if col < 3 else jnp.zeros(3)
            omega = e[col - 3] if col >= 3 else jnp.zeros(3)

            r = self._body_pts - center
            u_body = U + jnp.cross(omega, r)

            # BEM free-space solve
            traction = self._bem_solve(u_body.ravel()).reshape(N_b, 3)

            # IB spread + LBM cold spinup (adaptive with diffusion-time min)
            force_field = self._spread_traction(traction)
            f_lbm = state["f"]
            f_lbm, u_lbm, n_spinup = self._run_lbm_until_converged(
                f_lbm, force_field, traction, self._lbm_max_spinup,
                cold_start=True,
            )

            # Select method and iteration count per column
            method = self._wall_correction_method
            if method == "auto":
                if col >= 3:
                    # Rotation: 1 pass, any method (wall effect < 1%)
                    col_method = "richardson"
                    n_iter = 2
                elif col == self._open_bc_axis:
                    # Axial translation: inline Richardson (0.1% proven)
                    col_method = "richardson"
                    n_iter = self._max_defect_iter
                else:
                    # Transverse translation: 1-pass Lamb (2.2% at 48³)
                    # No under-relaxation for single pass — use α=1.0
                    col_method = "lamb"
                    n_iter = 1
            else:
                col_method = method
                n_iter = 2 if col >= 3 else self._max_defect_iter

            # For 1-pass methods (Lamb), use α=1 (no relaxation needed)
            alpha_col = 1.0 if n_iter == 1 else self._alpha
            prev_drag = 0.0

            for iteration in range(n_iter):
                delta_u = compute_wall_correction(
                    col_method, u_lbm, traction,
                    self._body_pts, self._body_wts,
                    self._eval_data,
                    self._epsilon, self._mu,
                    self._dx, self._dt_lbm,
                    f_dist=f_lbm,
                    tau=self._tau,
                    rho_phys=self._rho,
                    motion_axis=col if col < 3 else col - 3,
                    body_radius=self._body_radius,
                )

                # Apply correction (uniform or per-point)
                if delta_u.ndim == 1:
                    delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
                else:
                    delta_u_body = delta_u  # (N_body, 3) from representation

                traction_new = self._bem_solve(
                    (u_body - delta_u_body).ravel()
                ).reshape(N_b, 3)

                traction = (1 - alpha_col) * traction + alpha_col * traction_new

                force_field = self._spread_traction(traction)
                f_lbm, u_lbm, _ = self._run_lbm_until_converged(
                    f_lbm, force_field, traction, self._lbm_max_warmstart,
                    cold_start=False,
                )

                # Track convergence
                F_iter, T_iter = compute_force_torque(
                    self._body_pts, self._body_wts, traction, center,
                )
                drag_diag = float(F_iter[col]) if col < 3 else float(T_iter[col - 3])

                if n_iter > 1:
                    logger.info(
                        "  col %d iter %d [%s]: drag=%.4f",
                        col, iteration + 1, col_method, drag_diag,
                    )

                rel_change = abs(drag_diag - prev_drag) / (abs(drag_diag) + 1e-30)
                prev_drag = drag_diag

                if iteration > 0 and rel_change < self._tol:
                    logger.info("  col %d converged at iter %d", col, iteration + 1)
                    break

            F, T = compute_force_torque(
                self._body_pts, self._body_wts, traction, center,
            )
            R[:3, col] = np.array(F)
            R[3:, col] = np.array(T)
            logger.info(
                "R col %d done: F=[%.4f,%.4f,%.4f] T=[%.4f,%.4f,%.4f]",
                col, *[float(x) for x in F], *[float(x) for x in T],
            )

        return R

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "drag_force": BoundaryFluxSpec(
                shape=(3,),
                description="Hydrodynamic drag force [N]",
                output_units="N",
            ),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,),
                description="Hydrodynamic drag torque [N·m]",
                output_units="N*m",
            ),
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
        }

    # ── FluidFieldProvider protocol ─────────────────────────────────

    def get_composited_flow_field(self) -> np.ndarray | None:
        """Return composited velocity field (BEM inside, LBM outside).

        Uses the converged LBM velocity field and BEM traction from
        the most recent update(). Returns None if no solve has run.

        For Tier A/B/C visualization.
        """
        if self._latest_velocity is None:
            return None
        return np.asarray(self._latest_velocity) * self._vel_conv

    def get_midplane_velocity(self, resolution: tuple[int, int]):
        """Z-midplane velocity magnitude."""
        if self._latest_velocity is None:
            return None
        vel_np = np.asarray(self._latest_velocity) * self._vel_conv
        nz = vel_np.shape[2]
        mid = vel_np[:, :, nz // 2, :]
        mag = np.linalg.norm(mid, axis=-1)
        target_nx, target_ny = resolution
        if mag.shape[0] != target_nx or mag.shape[1] != target_ny:
            sx = max(mag.shape[0] // target_nx, 1)
            sy = max(mag.shape[1] // target_ny, 1)
            mag = mag[::sx, ::sy][:target_nx, :target_ny]
        return mag
