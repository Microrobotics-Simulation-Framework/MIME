"""StokesletFluidNode — regularised Stokeslet BEM fluid solver.

A quasi-static Stokes flow solver for confined microrobot FSI.
Computes drag force and torque on a rigid body via a precomputed
6×6 resistance matrix (standalone mode) or via LU backsubstitution
with background flow correction (Schwarz coupling mode).

No Mach number constraint — operates at any rotation frequency.

Standalone mode:
    Resistance matrix R computed at init. update() is a 6×6 matvec.

Schwarz coupling mode (interface_mesh provided):
    Body-only BEM system assembled and LU-factorized at init.
    update() builds the RHS as (U_body - u_background), backsubstitutes
    for body traction, then evaluates the Stokeslet velocity field at
    the interface sphere points. This velocity is the BEM's contribution
    to the flow at the interface — sent to the LBM as Dirichlet BC.

    The interface sphere is an EVALUATION GRID, not a BEM surface.
    Wall confinement enters through the LBM background flow, not
    through the BEM system matrix. This is correct Schwarz coupling:
    each solver sees the other's influence through interface conditions.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from maddening.core.node import SimulationNode, BoundaryInputSpec, BoundaryFluxSpec
from maddening.core.edge import EdgeSpec

from .surface_mesh import SurfaceMesh
from .resistance import compute_resistance_matrix, compute_confined_resistance_matrix
from .bem import (
    assemble_system_matrix,
    assemble_rhs_rigid_motion,
    compute_force_torque,
)
from .flow_field import evaluate_velocity_field

logger = logging.getLogger(__name__)


class StokesletFluidNode(SimulationNode):
    """Regularised Stokeslet BEM fluid solver.

    Parameters
    ----------
    name : str
    timestep : float
        Simulation timestep [s]. Only used for graph scheduling —
        the Stokes solve is quasi-static (no time dependence).
    mu : float
        Dynamic viscosity [Pa·s].
    body_mesh : SurfaceMesh
        Body surface mesh for BEM.
    wall_mesh : SurfaceMesh or None
        Vessel wall mesh (None = unconfined). Used in standalone mode.
    interface_mesh : SurfaceMesh or None
        Interface sphere mesh for Schwarz coupling. When provided,
        the BEM evaluates its velocity field at these points after
        solving, and accepts background_flow at body surface points.
    epsilon : float or None
        Regularisation parameter. Default: body mesh spacing / 2.
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        mu: float,
        body_mesh: SurfaceMesh,
        wall_mesh: SurfaceMesh | None = None,
        interface_mesh: SurfaceMesh | None = None,
        epsilon: float | None = None,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu, **kwargs)

        self._body_mesh = body_mesh
        self._wall_mesh = wall_mesh
        self._interface_mesh = interface_mesh
        self._mu = mu
        self._schwarz_mode = interface_mesh is not None

        if epsilon is None:
            epsilon = body_mesh.mean_spacing / 2.0
        self._epsilon = epsilon

        if self._schwarz_mode:
            self._init_schwarz(body_mesh, interface_mesh, epsilon, mu)
        else:
            self._init_standalone(body_mesh, wall_mesh, epsilon, mu)

    def _init_standalone(self, body_mesh, wall_mesh, epsilon, mu):
        """Standalone mode: precompute 6×6 resistance matrix."""
        logger.info(
            "Standalone mode: computing resistance matrix "
            "(N_body=%d, N_wall=%d, ε=%.4f)...",
            body_mesh.n_points,
            wall_mesh.n_points if wall_mesh else 0,
            epsilon,
        )

        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)
        center = jnp.zeros(3)

        if wall_mesh is not None:
            wall_pts = jnp.array(wall_mesh.points)
            wall_wts = jnp.array(wall_mesh.weights)
            self._R = np.array(compute_confined_resistance_matrix(
                body_pts, body_wts, wall_pts, wall_wts,
                center, epsilon, mu,
            ))
        else:
            self._R = np.array(compute_resistance_matrix(
                body_pts, body_wts, center, epsilon, mu,
            ))

        logger.info("Resistance matrix computed: R shape %s", self._R.shape)

    def _init_schwarz(self, body_mesh, interface_mesh, epsilon, mu):
        """Schwarz mode: assemble body-only BEM system, LU-factorize.

        The interface sphere is NOT part of the BEM system — it is an
        evaluation grid only. The BEM solves for body traction given
        the body velocity minus background flow. After solving, the
        BEM velocity field is evaluated at the interface sphere points.
        """
        N_b = body_mesh.n_points
        N_i = interface_mesh.n_points

        logger.info(
            "Schwarz mode: assembling body-only BEM system "
            "(N_body=%d, N_interface_eval=%d, ε=%.4f)...",
            N_b, N_i, epsilon,
        )

        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)

        # Assemble body-only system matrix
        A = assemble_system_matrix(body_pts, body_wts, epsilon, mu)

        # LU-factorize once — reused for all Schwarz iterations
        logger.info("LU-factorizing %d×%d body-only system...", A.shape[0], A.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        # Store dimensions and meshes
        self._N_body = N_b
        self._N_interface = N_i
        self._body_pts_jax = body_pts
        self._body_wts_jax = body_wts
        self._iface_pts_jax = jnp.array(interface_mesh.points)

        logger.info("Schwarz BEM system ready: %d body DOF, %d interface eval points",
                     3 * N_b, N_i)

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        state = {
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
        }
        if self._schwarz_mode:
            state["interface_velocity"] = jnp.zeros((self._N_interface, 3))
        return state

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        spec = {
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body angular velocity [rad/s]",
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body translational velocity [m/s]",
            ),
            "body_orientation": BoundaryInputSpec(
                shape=(4,),
                default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Body orientation quaternion",
            ),
        }
        if self._schwarz_mode:
            spec["background_flow"] = BoundaryInputSpec(
                shape=(self._N_interface, 3),
                default=jnp.zeros((self._N_interface, 3)),
                description="Far-field velocity at interface sphere points [m/s]",
            )
        return spec

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        if self._schwarz_mode:
            return self._update_schwarz(state, boundary_inputs, dt)
        else:
            return self._update_standalone(state, boundary_inputs, dt)

    def _update_standalone(self, state, boundary_inputs, dt):
        """Standalone: R @ [U, ω] → [F, T]."""
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))

        R = jnp.array(self._R)
        motion = jnp.concatenate([U, omega])
        response = R @ motion

        return {
            "drag_force": response[:3],
            "drag_torque": response[3:],
        }

    def _update_schwarz(self, state, boundary_inputs, dt):
        """Schwarz: body-only BEM with background flow from LBM.

        The exchange happens at the interface sphere only:
        1. Receive u_background at INTERFACE points from LBM
        2. Interpolate u_background to body surface points (smooth far-field)
        3. Solve body-only BEM: RHS = U_body - u_bg_at_body
        4. Evaluate TOTAL velocity at interface: u_bg + BEM perturbation
        5. Send total interface velocity to LBM as Dirichlet BC
        """
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        bg_flow_iface = boundary_inputs.get(
            "background_flow",
            jnp.zeros((self._N_interface, 3)),
        )

        center = jnp.zeros(3)
        N_b = self._N_body

        # Interpolate bg flow from interface sphere to body surface.
        # Far-field flow varies on vessel scale >> body scale, so
        # linear interpolation is accurate.
        bg_flow_body = self._interpolate_iface_to_body(bg_flow_iface)

        # RHS = rigid body velocity - background flow at body points
        r = self._body_pts_jax - center
        u_body = U + jnp.cross(omega, r)  # (N_b, 3)
        rhs = (u_body - bg_flow_body).ravel()  # (3*N_b,)

        # Backsubstitute using precomputed LU factors
        lu = jnp.array(self._lu)
        piv = jnp.array(self._piv)
        solution = jax.scipy.linalg.lu_solve((lu, piv), rhs)

        # Extract body traction → force/torque
        body_traction = solution.reshape(N_b, 3)
        F, T = compute_force_torque(
            self._body_pts_jax, self._body_wts_jax,
            body_traction, center,
        )

        # Evaluate TOTAL velocity at interface sphere:
        # u_total = u_background_at_iface + perturbation from body
        perturbation_vel = evaluate_velocity_field(
            self._iface_pts_jax,
            self._body_pts_jax, self._body_wts_jax, body_traction,
            self._epsilon, self._mu,
        )
        total_iface_vel = bg_flow_iface + perturbation_vel

        return {
            "drag_force": F,
            "drag_torque": T,
            "interface_velocity": total_iface_vel,
        }

    def _interpolate_iface_to_body(
        self, bg_flow_iface: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate background flow from interface sphere to body points.

        The far-field flow is smooth (varies on vessel scale, not body scale).
        Fit a linear model u(x) = u_0 + G·x to the interface velocity data,
        then evaluate at each body surface point.
        """
        x_i = self._iface_pts_jax  # (N_i, 3)
        u_i = bg_flow_iface         # (N_i, 3)

        # Design matrix: [1, x, y, z]
        ones = jnp.ones((self._N_interface, 1))
        A = jnp.concatenate([ones, x_i], axis=1)  # (N_i, 4)

        # Least squares: coeffs = (A^T A)^{-1} A^T u
        AtA = A.T @ A  # (4, 4)
        Atu = A.T @ u_i  # (4, 3)
        coeffs = jnp.linalg.solve(AtA, Atu)  # (4, 3)

        # Evaluate at body points
        A_body = jnp.concatenate(
            [jnp.ones((self._N_body, 1)), self._body_pts_jax], axis=1,
        )  # (N_b, 4)
        return A_body @ coeffs  # (N_b, 3)

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        spec = {
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
        if self._schwarz_mode:
            spec["interface_velocity"] = BoundaryFluxSpec(
                shape=(self._N_interface, 3),
                description="BEM perturbation velocity at interface [m/s]",
                output_units="m/s",
            )
        return spec

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        fluxes = {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
        }
        if self._schwarz_mode:
            fluxes["interface_velocity"] = state["interface_velocity"]
        return fluxes

    # -- FluidFieldProvider protocol -----------------------------------------

    def get_midplane_velocity(self, resolution: tuple[int, int]):
        """Evaluate velocity at a grid of points on the Z=0 plane."""
        return None


def make_stokeslet_rigid_body_edges(
    stokeslet_node_name: str,
    rigid_body_node_name: str,
) -> list[EdgeSpec]:
    """Return EdgeSpecs wiring StokesletFluidNode to RigidBodyNode.

    No unit transforms needed — Stokeslet operates in SI directly.
    """
    return [
        EdgeSpec(
            source_node=stokeslet_node_name,
            target_node=rigid_body_node_name,
            source_field="drag_force",
            target_field="drag_force",
            additive=True,
            source_units="N",
            target_units="N",
        ),
        EdgeSpec(
            source_node=stokeslet_node_name,
            target_node=rigid_body_node_name,
            source_field="drag_torque",
            target_field="drag_torque",
            additive=True,
            source_units="N*m",
            target_units="N*m",
        ),
        EdgeSpec(
            source_node=rigid_body_node_name,
            target_node=stokeslet_node_name,
            source_field="angular_velocity",
            target_field="body_angular_velocity",
        ),
        EdgeSpec(
            source_node=rigid_body_node_name,
            target_node=stokeslet_node_name,
            source_field="velocity",
            target_field="body_velocity",
        ),
        EdgeSpec(
            source_node=rigid_body_node_name,
            target_node=stokeslet_node_name,
            source_field="orientation",
            target_field="body_orientation",
        ),
    ]


def make_schwarz_coupling_edges(
    bem_node_name: str,
    lbm_node_name: str,
) -> list[EdgeSpec]:
    """Return EdgeSpecs for BEM ↔ LBM Schwarz coupling.

    BEM → LBM: interface_velocity (total vel as Dirichlet BC)
    LBM → BEM: interface_background_velocity → background_flow
    """
    return [
        EdgeSpec(
            source_node=bem_node_name,
            target_node=lbm_node_name,
            source_field="interface_velocity",
            target_field="interface_velocity",
        ),
        EdgeSpec(
            source_node=lbm_node_name,
            target_node=bem_node_name,
            source_field="interface_background_velocity",
            target_field="background_flow",
        ),
    ]
