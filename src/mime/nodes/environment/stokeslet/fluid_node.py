"""StokesletFluidNode — regularised Stokeslet BEM fluid solver.

A quasi-static Stokes flow solver for confined microrobot FSI.
Computes drag force and torque on a rigid body via a precomputed
6×6 resistance matrix (standalone mode) or via LU backsubstitution
with an interface boundary (Schwarz coupling mode).

No Mach number constraint — operates at any rotation frequency.

Standalone mode:
    Resistance matrix R computed at init. update() is a 6×6 matvec.

Schwarz coupling mode (interface_mesh provided):
    BEM system matrix assembled over body + interface surfaces at init.
    LU-factorized once. update() builds the RHS (body: rigid body
    velocity, interface: background flow from far-field solver) and
    backsubstitutes. Force/torque extracted from body traction only.
    Interface velocity evaluated via Stokeslet sum for sending to
    the far-field solver.
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
from .nearest_neighbour import (
    compute_nearest_neighbour_map,
    assemble_nn_confined_system,
)
from .bem import assemble_rhs_rigid_motion, assemble_rhs_confined, compute_force_torque
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
        the BEM system includes body + interface as two surfaces.
        The interface acts as the BEM's outer boundary with velocity
        prescribed by the far-field solver.
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
        """Schwarz mode: assemble body+interface BEM system, LU-factorize."""
        N_b = body_mesh.n_points
        N_i = interface_mesh.n_points

        logger.info(
            "Schwarz mode: assembling BEM system "
            "(N_body=%d, N_interface=%d, ε=%.4f)...",
            N_b, N_i, epsilon,
        )

        # Use NN method: coarse force points = mesh points,
        # fine quadrature = refined mesh
        # For now, use the mesh points as both force and quad
        # (standard Nyström — NN refinement can be added later)
        body_force_pts = jnp.array(body_mesh.points)
        body_quad_pts = jnp.array(body_mesh.points)
        body_quad_wts = jnp.array(body_mesh.weights)
        body_nn = compute_nearest_neighbour_map(
            np.array(body_mesh.points), np.array(body_mesh.points),
        )

        iface_force_pts = jnp.array(interface_mesh.points)
        iface_quad_pts = jnp.array(interface_mesh.points)
        iface_quad_wts = jnp.array(interface_mesh.weights)
        iface_nn = compute_nearest_neighbour_map(
            np.array(interface_mesh.points), np.array(interface_mesh.points),
        )

        # Assemble system: same structure as body + wall confined system
        # [A_bb  A_bi] [f_body    ]   [u_body     ]
        # [A_ib  A_ii] [f_interface] = [u_interface]
        A = assemble_nn_confined_system(
            body_force_pts, body_quad_pts, body_quad_wts, body_nn,
            iface_force_pts, iface_quad_pts, iface_quad_wts, iface_nn,
            epsilon, mu,
        )

        # LU-factorize once — reused for all Schwarz iterations
        logger.info("LU-factorizing %d×%d system...", A.shape[0], A.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        # Store dimensions
        self._N_body = N_b
        self._N_interface = N_i

        # Store meshes as JAX arrays for update()
        self._body_pts_jax = body_force_pts
        self._body_wts_jax = jnp.array(body_mesh.weights)
        self._iface_pts_jax = iface_force_pts
        self._iface_wts_jax = jnp.array(interface_mesh.weights)

        logger.info("Schwarz BEM system ready: %d DOF", A.shape[0])

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
                description="Far-field velocity at interface points [m/s]",
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
        """Schwarz: backsubstitute with updated interface RHS."""
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        bg_flow = boundary_inputs.get(
            "background_flow",
            jnp.zeros((self._N_interface, 3)),
        )

        center = jnp.zeros(3)
        N_b = self._N_body
        N_i = self._N_interface

        # Build RHS: body portion = rigid body velocity
        rhs_body = assemble_rhs_rigid_motion(
            self._body_pts_jax, center, U, omega,
        )  # (3*N_b,)

        # Interface portion = background flow from LBM
        rhs_interface = bg_flow.ravel()  # (3*N_i,)

        rhs = jnp.concatenate([rhs_body, rhs_interface])

        # Backsubstitute using precomputed LU factors
        lu = jnp.array(self._lu)
        piv = jnp.array(self._piv)
        solution = jax.scipy.linalg.lu_solve((lu, piv), rhs)

        # Extract body traction → force/torque
        body_traction = solution[:3 * N_b].reshape(N_b, 3)
        F, T = compute_force_torque(
            self._body_pts_jax, self._body_wts_jax,
            body_traction, center,
        )

        # Evaluate BEM velocity at interface points
        # (Stokeslet sum over ALL tractions — body + interface)
        all_pts = jnp.concatenate([self._body_pts_jax, self._iface_pts_jax])
        all_wts = jnp.concatenate([self._body_wts_jax, self._iface_wts_jax])
        all_traction = solution.reshape(N_b + N_i, 3)

        interface_vel = evaluate_velocity_field(
            self._iface_pts_jax,
            all_pts, all_wts, all_traction,
            self._epsilon, self._mu,
        )

        return {
            "drag_force": F,
            "drag_torque": T,
            "interface_velocity": interface_vel,
        }

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
                description="BEM velocity at interface points [m/s]",
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
        # TODO: implement using stored traction from last solve
        return None


def make_stokeslet_rigid_body_edges(
    stokeslet_node_name: str,
    rigid_body_node_name: str,
) -> list[EdgeSpec]:
    """Return EdgeSpecs wiring StokesletFluidNode to RigidBodyNode.

    No unit transforms needed — Stokeslet operates in SI directly.

    Forward: drag_force, drag_torque → rigid body
    Back: angular_velocity, velocity, orientation → Stokeslet
    """
    return [
        # Forward: BEM drag → RigidBody (SI units, no transform)
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
        # Back-edges: RigidBody state → Stokeslet boundary inputs
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

    BEM → LBM: interface_velocity (Dirichlet BC for LBM)
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
