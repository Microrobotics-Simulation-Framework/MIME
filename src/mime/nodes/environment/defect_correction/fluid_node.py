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
        n_lbm_spinup: int = 500,
        n_lbm_warmstart: int = 200,
        max_defect_iter: int = 10,
        alpha: float | None = None,
        tol: float = 0.01,
        open_bc_axis: int = 2,
        epsilon: float | None = None,
        eval_radii_factors: tuple[float, ...] = (1.25, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0),
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu, **kwargs)

        self._mu = mu
        self._rho = rho
        self._body_radius = body_radius
        self._n_lbm_spinup = n_lbm_spinup
        self._n_lbm_warmstart = n_lbm_warmstart
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
        self._eval_stencils = []
        x_vals = []
        for R_factor in eval_radii_factors:
            R_ev = R_factor * body_radius
            if R_ev >= vessel_radius - dx:
                continue  # too close to wall
            ev_mesh = sphere_surface_mesh(radius=R_ev, n_refine=2)
            ev_lu = ev_mesh.points / dx + np.array([N_lbm / 2] * 3)
            ei, ew = precompute_ib_stencil(ev_lu, (N_lbm, N_lbm, N_lbm))
            self._eval_stencils.append({
                'pts_phys': jnp.array(ev_mesh.points),
                'idx': jnp.array(ei),
                'wts': jnp.array(ew),
            })
            x_vals.append(body_radius / R_ev)
        self._x_vals = jnp.array(x_vals)

        # ── Auto-relaxation parameter ───────────────────────────────
        if alpha is None:
            # Haberman-Sayre estimate for sphere-in-cylinder wall effect
            kappa = body_radius / vessel_radius
            W_est = 2.1044 * kappa / (1 - 2.1044 * kappa) if kappa < 0.4 else 3.0
            self._alpha = 0.8 / (1.0 + W_est)
        else:
            self._alpha = alpha

        # ── Velocity field stash for FluidFieldProvider ─────────────
        self._latest_velocity = None
        self._latest_traction = None
        self._first_call = True

        logger.info(
            "DefectCorrectionFluidNode: %d³ LBM, %d body pts, %d eval radii, "
            "α=%.2f, max_iter=%d",
            N_lbm, N_b, len(self._eval_stencils), self._alpha, max_defect_iter,
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

        PALLAS BOUNDARY: This entire function body is the target for
        Pallas kernel fusion. When replacing, the signature stays the same:
        (f: [nx,ny,nz,19], force: [nx,ny,nz,3]) -> (f: [nx,ny,nz,19], u: [nx,ny,nz,3])

        The Pallas kernel fuses: BGK collision + Guo forcing + streaming +
        bounce-back (using self._pipe_wall) + open BCs (on self._open_bc_axis).
        """
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

        n_steps = self._n_lbm_spinup if self._first_call else self._n_lbm_warmstart
        self._first_call = False

        for step in range(n_steps):
            f_lbm, u_lbm = self._lbm_full_step(f_lbm, force_field)

        # Defect correction iterations
        for iteration in range(self._max_defect_iter):
            # Wall correction: Δu at body surface
            delta_u = compute_wall_correction(
                u_lbm, traction,
                self._body_pts, self._body_wts,
                self._eval_stencils, self._x_vals,
                self._epsilon, self._mu,
                self._dx, self._dt_lbm,
            )

            # BEM re-solve with wall correction
            delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
            rhs_corrected = (u_body - delta_u_body).ravel()
            traction_new = self._bem_solve(rhs_corrected).reshape(N_b, 3)

            # Under-relaxation
            traction = (1 - self._alpha) * traction + self._alpha * traction_new

            # Update LBM with new traction (warm-start)
            force_field = self._spread_traction(traction)
            for step in range(self._n_lbm_warmstart):
                f_lbm, u_lbm = self._lbm_full_step(f_lbm, force_field)

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

            # IB spread + LBM spinup
            force_field = self._spread_traction(traction)
            f_lbm = state["f"]
            for step in range(self._n_lbm_spinup):
                f_lbm, u_lbm = self._lbm_full_step(f_lbm, force_field)

            # Rotation: 1 defect correction pass
            # Translation: iterate up to max_defect_iter
            n_iter = 1 if col >= 3 else self._max_defect_iter

            for iteration in range(n_iter):
                delta_u = compute_wall_correction(
                    u_lbm, traction,
                    self._body_pts, self._body_wts,
                    self._eval_stencils, self._x_vals,
                    self._epsilon, self._mu,
                    self._dx, self._dt_lbm,
                )
                delta_u_body = jnp.broadcast_to(delta_u, (N_b, 3))
                traction_new = self._bem_solve(
                    (u_body - delta_u_body).ravel()
                ).reshape(N_b, 3)

                traction = (1 - self._alpha) * traction + self._alpha * traction_new

                force_field = self._spread_traction(traction)
                for step in range(self._n_lbm_warmstart):
                    f_lbm, u_lbm = self._lbm_full_step(f_lbm, force_field)

            F, T = compute_force_torque(
                self._body_pts, self._body_wts, traction, center,
            )
            R[:3, col] = np.array(F)
            R[3:, col] = np.array(T)

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
