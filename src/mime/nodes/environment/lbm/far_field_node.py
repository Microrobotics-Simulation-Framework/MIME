"""LBMFarFieldNode — LBM far-field solver for Schwarz domain decomposition.

Solves the far-field flow in a cylindrical vessel driven by immersed
boundary forces from the BEM near-field solver. No resolved body geometry
in the LBM domain — the body's influence enters entirely through the
Guo forcing term via Peskin delta spreading.

Paired with StokesletFluidNode via MADDENING's CouplingGroup for
Dirichlet-Neumann Schwarz coupling:
    BEM → (traction → IB spreading) → LBM force field
    LBM → (IB interpolation) → BEM background velocity

References:
    Guo, Zheng & Shi (2002), Phys. Rev. E 65:046308 — Guo forcing
    Peskin (2002), Acta Numerica 11:479-517 — IB method
    Tian et al. (2011), J. Comput. Phys. 230:7266-7283 — IB-LBM
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
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
    interpolate_velocity,
)

logger = logging.getLogger(__name__)


class LBMFarFieldNode(SimulationNode):
    """LBM far-field solver with IB force coupling.

    The LBM domain is a cylindrical vessel with no-slip walls and
    open (pressure) BCs on the axial faces. No solid body inside —
    the body's influence is a distributed force field from IB spreading.

    Parameters
    ----------
    name : str
    timestep : float
        Physical timestep [s].
    nx, ny, nz : int
        Lattice dimensions.
    tau : float
        BGK relaxation time.
    vessel_radius_lu : float
        Pipe wall radius in lattice units.
    body_points_lu : np.ndarray
        (N_body, 3) BEM body surface points in lattice units.
        Used for IB velocity interpolation (LBM → BEM).
    open_bc_axis : int or None
        Axis for open (pressure) BCs. None = periodic (default).
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        nx: int,
        ny: int,
        nz: int,
        tau: float,
        vessel_radius_lu: float,
        body_points_lu: np.ndarray,
        open_bc_axis: int | None = None,
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            nx=nx, ny=ny, nz=nz, tau=tau,
            vessel_radius_lu=vessel_radius_lu,
            **kwargs,
        )

        self._n_body = len(body_points_lu)
        self._open_bc_axis = open_bc_axis

        # Static pipe wall mask (cylinder along Z)
        cx, cy = nx / 2.0, ny / 2.0
        ix = jnp.arange(nx, dtype=jnp.float32)
        iy = jnp.arange(ny, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        self._pipe_wall = jnp.broadcast_to(
            (dist_2d >= vessel_radius_lu)[..., None], (nx, ny, nz),
        )
        self._pipe_missing = compute_missing_mask(self._pipe_wall)

        # Precompute IB stencil for velocity interpolation at body points
        self._ib_indices, self._ib_weights = precompute_ib_stencil(
            np.array(body_points_lu), (nx, ny, nz),
        )
        self._ib_indices_jax = jnp.array(self._ib_indices)
        self._ib_weights_jax = jnp.array(self._ib_weights)

        # Velocity field stash for FluidFieldProvider
        self._latest_velocity = None

        logger.info(
            "LBMFarFieldNode: %dx%dx%d, tau=%.2f, vessel_R=%.1f lu, "
            "N_body=%d, open_bc=%s",
            nx, ny, nz, tau, vessel_radius_lu,
            self._n_body, open_bc_axis,
        )

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        return {
            "f": init_equilibrium(nx, ny, nz),
            "velocity_at_body": jnp.zeros((self._n_body, 3)),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        return {
            "ib_force_field": BoundaryInputSpec(
                shape=(nx, ny, nz, 3),
                default=jnp.zeros((nx, ny, nz, 3)),
                description="IB body force field [lattice units]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        tau = self.params["tau"]
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]

        force_field = boundary_inputs.get(
            "ib_force_field",
            jnp.zeros((nx, ny, nz, 3)),
        )

        # LBM step with Guo forcing from IB-spread body force
        f_pre, f_post, rho, u = lbm_step_split(state["f"], tau, force=force_field)
        self._latest_velocity = u

        # Bounce-back: vessel wall only (no body boundary)
        f = apply_bounce_back(
            f_post, f_pre, self._pipe_missing, self._pipe_wall,
            wall_velocity=None,
        )

        # Open BCs on axial faces
        if self._open_bc_axis is not None:
            f = self._apply_open_bc(f, rho)

        # IB interpolation: LBM velocity at body surface points
        vel_at_body = interpolate_velocity(
            u, self._ib_indices_jax, self._ib_weights_jax,
        )

        return {
            "f": f,
            "velocity_at_body": vel_at_body,
        }

    def _apply_open_bc(
        self,
        f: jnp.ndarray,
        rho: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply open (pressure) BCs at axial faces.

        Non-equilibrium extrapolation: outlet copies from interior,
        inlet set to equilibrium at reference density.
        """
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

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "velocity_at_body": BoundaryFluxSpec(
                shape=(self._n_body, 3),
                description="LBM velocity at body surface [lattice units]",
                output_units="lattice",
            ),
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "velocity_at_body": state["velocity_at_body"],
        }

    # -- FluidFieldProvider protocol -----------------------------------------

    def get_midplane_velocity(self, resolution: tuple[int, int]):
        if self._latest_velocity is None:
            return None
        vel_np = np.asarray(self._latest_velocity)
        nz = vel_np.shape[2]
        mid = vel_np[:, :, nz // 2, :]
        mag = np.linalg.norm(mid, axis=-1)
        target_nx, target_ny = resolution
        if mag.shape[0] != target_nx or mag.shape[1] != target_ny:
            sx = max(mag.shape[0] // target_nx, 1)
            sy = max(mag.shape[1] // target_ny, 1)
            mag = mag[::sx, ::sy][:target_nx, :target_ny]
        return mag

    def get_full_velocity_field(self):
        if self._latest_velocity is None:
            return None
        return np.asarray(self._latest_velocity)
