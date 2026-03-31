"""LBMFarFieldNode — LBM far-field solver for Schwarz domain decomposition.

A simplified LBM node that solves the far-field flow (vessel + interface
sphere) without any resolved robot geometry. The interface sphere acts
as a Bouzidi IBB boundary with prescribed velocity from the BEM near-field
solver.

This node is paired with StokesletFluidNode via MADDENING's CouplingGroup
for Schwarz-type domain decomposition coupling.

Key differences from IBLBMFluidNode:
- No UMR geometry, no helix mask, no rotating body
- Interface sphere = simple spherical solid with prescribed velocity
- No Mach constraint (sphere moves slowly)
- Outputs LBM velocity at interface points (for BEM background flow)
"""

from __future__ import annotations

import logging

import jax
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

logger = logging.getLogger(__name__)


class LBMFarFieldNode(SimulationNode):
    """LBM far-field solver with interface sphere boundary.

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
    interface_center_lu : tuple[float, float, float]
        Interface sphere center in lattice units.
    interface_radius_lu : float
        Interface sphere radius in lattice units.
    interface_points_physical : np.ndarray
        (N_interface, 3) interface mesh points in physical coordinates.
        Used for velocity interpolation (LBM → BEM).
    body_points_physical : np.ndarray or None
        (N_body, 3) body surface points in physical coordinates.
        If provided, the LBM evaluates its velocity at these points
        to provide background flow to the BEM.
    dx_physical : float
        Physical lattice spacing [m].
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
        interface_center_lu: tuple[float, float, float],
        interface_radius_lu: float,
        interface_points_physical: np.ndarray,
        body_points_physical: np.ndarray | None = None,
        dx_physical: float = 1.0,
        open_bc_axis: int | None = None,
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            nx=nx, ny=ny, nz=nz,
            tau=tau,
            vessel_radius_lu=vessel_radius_lu,
            interface_center_lu=interface_center_lu,
            interface_radius_lu=interface_radius_lu,
            dx_physical=dx_physical,
            **kwargs,
        )

        cx, cy, cz = interface_center_lu
        self._center_lu = jnp.array([cx, cy, cz])
        self._interface_radius_lu = interface_radius_lu
        self._open_bc_axis = open_bc_axis
        self._dx = dx_physical
        self._n_interface = len(interface_points_physical)
        self._n_body = len(body_points_physical) if body_points_physical is not None else 0

        # Static pipe wall mask (cylinder along Z)
        pipe_cx, pipe_cy = nx / 2.0, ny / 2.0
        ix = jnp.arange(nx, dtype=jnp.float32)
        iy = jnp.arange(ny, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        dist_2d = jnp.sqrt((gx - pipe_cx) ** 2 + (gy - pipe_cy) ** 2)
        self._pipe_wall = jnp.broadcast_to(
            (dist_2d >= vessel_radius_lu)[..., None], (nx, ny, nz),
        )
        self._pipe_missing = compute_missing_mask(self._pipe_wall)

        # Interface sphere mask
        ix3 = jnp.arange(nx, dtype=jnp.float32)
        iy3 = jnp.arange(ny, dtype=jnp.float32)
        iz3 = jnp.arange(nz, dtype=jnp.float32)
        gx3, gy3, gz3 = jnp.meshgrid(ix3, iy3, iz3, indexing='ij')
        dist_3d = jnp.sqrt(
            (gx3 - cx) ** 2 + (gy3 - cy) ** 2 + (gz3 - cz) ** 2
        )
        self._sphere_mask = dist_3d <= interface_radius_lu
        self._sphere_missing = compute_missing_mask(self._sphere_mask)

        # Precompute interpolation map: sample LBM velocity OUTSIDE the
        # interface sphere (offset outward by 1 lattice spacing).
        # Must be outside the solid sphere but close to the boundary
        # to minimize the velocity decay error from the offset.
        # At 1.0 lu offset, the velocity error is ~O(dx/R_iface).
        iface_lu = interface_points_physical / dx_physical
        center_arr = np.array([cx, cy, cz])
        normals_outward = (iface_lu - center_arr)
        normals_outward /= np.linalg.norm(normals_outward, axis=1, keepdims=True)
        sample_offset = 1.0  # lattice spacings outward
        sample_pts_lu = iface_lu + sample_offset * normals_outward

        self._interp_indices, self._interp_weights = (
            self._build_interpolation_map(sample_pts_lu, (nx, ny, nz))
        )

        # Precompute reverse map: for each lattice node, nearest interface point
        # Used for spreading interface velocity to lattice boundary nodes
        from scipy.spatial import cKDTree
        tree = cKDTree(np.array(iface_lu))
        # Generate all lattice coordinates
        all_coords = np.stack(np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij',
        ), axis=-1).reshape(-1, 3)
        _, self._lattice_to_iface = tree.query(all_coords)
        self._lattice_to_iface = self._lattice_to_iface.reshape(nx, ny, nz)

        # Build interpolation map for body surface points (inside sphere)
        # These are inside the solid region, so the LBM velocity there is
        # unphysical. We interpolate from nearby FLUID nodes instead.
        # Offset body points outward to the nearest fluid layer.
        if body_points_physical is not None:
            body_lu = body_points_physical / dx_physical
            # Body points are inside the sphere. Move them to just outside
            # the sphere (in the fluid region) for interpolation.
            body_r = body_lu - center_arr
            body_dist = np.linalg.norm(body_r, axis=1, keepdims=True)
            body_dir = body_r / (body_dist + 1e-30)
            # Place sample points at interface_radius + 1 lattice spacing
            sample_r = interface_radius_lu + 1.0
            body_sample_lu = center_arr + body_dir * sample_r
            self._body_interp_indices, self._body_interp_weights = (
                self._build_interpolation_map(body_sample_lu, (nx, ny, nz))
            )

        # Store for velocity stashing
        self._latest_velocity = None

        logger.info(
            "LBMFarFieldNode: %dx%dx%d, tau=%.2f, "
            "vessel_R=%.1f lu, interface_R=%.1f lu, "
            "N_interface=%d, N_body=%d",
            nx, ny, nz, tau, vessel_radius_lu,
            interface_radius_lu, self._n_interface, self._n_body,
        )

    @staticmethod
    def _build_interpolation_map(
        points_lu: np.ndarray,
        lattice_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build trilinear interpolation map from interface points to lattice."""
        N = len(points_lu)
        nx, ny, nz = lattice_shape

        frac = np.array(points_lu, dtype=np.float64)
        i0 = np.floor(frac).astype(int)
        f = frac - i0

        i0 = np.clip(i0, 0, np.array([nx - 2, ny - 2, nz - 2]))
        f = np.clip(f, 0.0, 1.0)

        offsets = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ])

        indices = np.zeros((N, 8), dtype=int)
        weights = np.zeros((N, 8), dtype=np.float64)

        for c, (di, dj, dk) in enumerate(offsets):
            ii = i0[:, 0] + di
            jj = i0[:, 1] + dj
            kk = i0[:, 2] + dk
            indices[:, c] = ii * ny * nz + jj * nz + kk

            wx = f[:, 0] if di else (1.0 - f[:, 0])
            wy = f[:, 1] if dj else (1.0 - f[:, 1])
            wz = f[:, 2] if dk else (1.0 - f[:, 2])
            weights[:, c] = wx * wy * wz

        return indices, weights

    @property
    def requires_halo(self) -> bool:
        return False  # single-device

    def initial_state(self) -> dict:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]

        state = {
            "f": init_equilibrium(nx, ny, nz),
            "interface_background_velocity": jnp.zeros((self._n_interface, 3)),
        }
        if self._n_body > 0:
            state["background_flow_at_body"] = jnp.zeros((self._n_body, 3))
        return state

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "interface_velocity": BoundaryInputSpec(
                shape=(self._n_interface, 3),
                default=jnp.zeros((self._n_interface, 3)),
                description="Prescribed velocity at interface sphere [lattice units]",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        tau = self.params["tau"]
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]

        interface_vel = boundary_inputs.get(
            "interface_velocity",
            jnp.zeros((self._n_interface, 3)),
        )

        # Convert interface velocity from physical to lattice units
        # u_lattice = u_physical * dt_physical / dx_physical
        # But for the LBM, the velocity at the boundary is already
        # in the natural units of the coupling. The BEM outputs m/s,
        # the edge transform handles conversion.

        # Build wall velocity field for the interface sphere
        # Map interface point velocities to a full lattice velocity field
        # using nearest-point assignment for the boundary links
        wall_vel = self._build_sphere_wall_velocity(
            interface_vel, nx, ny, nz,
        )

        # Combined solid mask: pipe wall + interface sphere
        solid_mask = self._pipe_wall | self._sphere_mask

        # LBM step
        f_pre, f_post, rho, u = lbm_step_split(state["f"], tau)
        self._latest_velocity = u

        # Bounce-back pass 1: pipe wall (static, no velocity)
        f = apply_bounce_back(
            f_post, f_pre, self._pipe_missing, solid_mask,
            wall_velocity=None,
        )

        # Bounce-back pass 2: interface sphere (with prescribed velocity)
        f = apply_bounce_back(
            f, f_pre, self._sphere_missing, solid_mask,
            wall_velocity=wall_vel,
        )

        # Open BCs on axial faces (if enabled)
        if self._open_bc_axis is not None:
            f = self._apply_open_bc(f, rho)

        # Interpolate LBM velocity at interface points (outside sphere)
        bg_vel = self._interpolate_velocity_at_interface(u)

        result = {
            "f": f,
            "interface_background_velocity": bg_vel,
        }

        # Interpolate LBM velocity at body surface points (for BEM bg flow)
        if self._n_body > 0:
            result["background_flow_at_body"] = self._interpolate_velocity_at_body(u)

        return result

    def _build_sphere_wall_velocity(
        self,
        interface_vel: jnp.ndarray,
        nx: int, ny: int, nz: int,
    ) -> jnp.ndarray:
        """Map interface point velocities to lattice velocity field.

        Creates a (nx, ny, nz, 3) velocity field where each lattice node
        gets the velocity of its nearest interface mesh point. Only values
        at boundary nodes (where missing_mask is True) matter; the rest
        are ignored by bounce-back.
        """
        mapping = jnp.array(self._lattice_to_iface)  # (nx, ny, nz)
        vel_field = interface_vel[mapping]  # (nx, ny, nz, 3)
        return vel_field

    def _apply_open_bc(
        self,
        f: jnp.ndarray,
        rho: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply open (pressure) BCs at axial faces.

        Uses non-equilibrium extrapolation: at the outlet face, copy
        distributions from the second-to-last layer. At the inlet face,
        set to equilibrium at reference density ρ₀=1 and zero velocity
        (quiescent far-field).

        This is first-order accurate but stable and trivial to implement.
        For the Schwarz coupling where quantitative drag comes from the
        BEM, the LBM accuracy at the boundaries is secondary.

        Parameters
        ----------
        f : (nx, ny, nz, 19) post-bounce-back distributions
        rho : (nx, ny, nz) density field (for reference)

        Returns
        -------
        f : (nx, ny, nz, 19) with open BCs applied
        """
        axis = self._open_bc_axis
        rho_0 = 1.0  # reference density

        if axis == 0:  # x-axis
            # Outlet: x = nx-1 ← copy from x = nx-2
            f = f.at[-1, :, :, :].set(f[-2, :, :, :])
            # Inlet: x = 0 ← equilibrium at rho_0, u=0
            f_eq_slice = equilibrium(
                jnp.full(f.shape[1:3], rho_0),
                jnp.zeros((*f.shape[1:3], 3)),
            )
            f = f.at[0, :, :, :].set(f_eq_slice)
        elif axis == 1:  # y-axis
            f = f.at[:, -1, :, :].set(f[:, -2, :, :])
            f_eq_slice = equilibrium(
                jnp.full((f.shape[0], f.shape[2]), rho_0),
                jnp.zeros((f.shape[0], f.shape[2], 3)),
            )
            f = f.at[:, 0, :, :].set(f_eq_slice)
        elif axis == 2:  # z-axis
            # Outlet: z = nz-1 ← copy from z = nz-2
            f = f.at[:, :, -1, :].set(f[:, :, -2, :])
            # Inlet: z = 0 ← equilibrium at rho_0, u=0
            f_eq_slice = equilibrium(
                jnp.full(f.shape[0:2], rho_0),
                jnp.zeros((*f.shape[0:2], 3)),
            )
            f = f.at[:, :, 0, :].set(f_eq_slice)

        return f

    def _apply_open_bc_pressure(
        self,
        f: jnp.ndarray,
        rho_in: float = 1.0,
        rho_out: float = 1.0,
    ) -> jnp.ndarray:
        """Apply pressure BCs at axial faces (for Poiseuille validation).

        At each face, set density to the prescribed value and compute
        unknown distributions via non-equilibrium bounce-back (Zou-He
        simplified for zero tangential velocity).

        Parameters
        ----------
        f : (nx, ny, nz, 19)
        rho_in : float, inlet density
        rho_out : float, outlet density
        """
        axis = self._open_bc_axis
        if axis != 2:
            raise NotImplementedError("Pressure BC only for z-axis currently")

        # For D3Q19 with z-axis:
        # At z=nz-1 (outlet): unknown are distributions with ez < 0
        #   Indices: 6 (-z), 13 (+x-z), 14 (-x-z), 17 (+y-z), 18 (-y-z)
        # At z=0 (inlet): unknown are distributions with ez > 0
        #   Indices: 5 (+z), 11 (+x+z), 12 (-x+z), 15 (+y+z), 16 (-y+z)

        # Outlet face (z = nz-1): set to equilibrium at rho_out, u from interior
        u_out = f[:, :, -2, :].sum(axis=-1)  # not right, use proper macroscopic
        # Simpler: extrapolation + density correction
        f_out = f[:, :, -2, :]  # copy from interior
        # Scale to match target density
        rho_current = jnp.sum(f_out, axis=-1, keepdims=True)
        f = f.at[:, :, -1, :].set(f_out * (rho_out / (rho_current + 1e-30)))

        # Inlet face (z = 0): extrapolation + density correction
        f_in = f[:, :, 1, :]
        rho_current_in = jnp.sum(f_in, axis=-1, keepdims=True)
        f = f.at[:, :, 0, :].set(f_in * (rho_in / (rho_current_in + 1e-30)))

        return f

    def _interpolate_velocity_at_interface(
        self, u: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate LBM velocity field at interface mesh points.

        Uses precomputed trilinear interpolation weights.
        Returns velocity in lattice units.
        """
        # Flatten velocity field
        u_flat = u.reshape(-1, 3)  # (nx*ny*nz, 3)

        indices = jnp.array(self._interp_indices)  # (N_interface, 8)
        weights = jnp.array(self._interp_weights)  # (N_interface, 8)

        # Trilinear interpolation
        result = jnp.zeros((self._n_interface, 3))
        for c in range(8):
            idx = indices[:, c]
            w = weights[:, c:c + 1]  # (N, 1)
            result = result + w * u_flat[idx]

        return result

    def _interpolate_velocity_at_body(
        self, u: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate LBM velocity at body surface sample points.

        Body points are inside the solid sphere, so we sample from
        fluid nodes just outside the interface sphere (precomputed
        at init). Returns velocity in lattice units.
        """
        u_flat = u.reshape(-1, 3)
        indices = jnp.array(self._body_interp_indices)
        weights = jnp.array(self._body_interp_weights)

        result = jnp.zeros((self._n_body, 3))
        for c in range(8):
            idx = indices[:, c]
            w = weights[:, c:c + 1]
            result = result + w * u_flat[idx]

        return result

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        spec = {
            "interface_background_velocity": BoundaryFluxSpec(
                shape=(self._n_interface, 3),
                description="LBM velocity at interface points [lattice units]",
                output_units="lattice",
            ),
        }
        if self._n_body > 0:
            spec["background_flow_at_body"] = BoundaryFluxSpec(
                shape=(self._n_body, 3),
                description="LBM velocity at body surface points [lattice units]",
                output_units="lattice",
            )
        return spec

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        fluxes = {
            "interface_background_velocity": state["interface_background_velocity"],
        }
        if self._n_body > 0:
            fluxes["background_flow_at_body"] = state["background_flow_at_body"]
        return fluxes

    # -- FluidFieldProvider protocol -----------------------------------------

    def get_midplane_velocity(
        self,
        resolution: tuple[int, int],
    ):
        """Return (nx, ny) velocity magnitude at the Z-midplane."""
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
        """Return the full (nx, ny, nz, 3) velocity field in lattice units."""
        if self._latest_velocity is None:
            return None
        return np.asarray(self._latest_velocity)
