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
    for body traction, and extracts force/torque. The traction is
    output for IB force spreading to the LBM far-field solver.

    The background flow comes from IB interpolation of the LBM velocity
    at the body surface points. At convergence, the BEM traction and
    LBM velocity are self-consistent: the spread traction produces
    the flow that the LBM computes.
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
from .bem import assemble_system_matrix, compute_force_torque

logger = logging.getLogger(__name__)


class StokesletFluidNode(SimulationNode):
    """Regularised Stokeslet BEM fluid solver.

    Parameters
    ----------
    name : str
    timestep : float
        Simulation timestep [s].
    mu : float
        Dynamic viscosity [Pa·s].
    body_mesh : SurfaceMesh
        Body surface mesh for BEM.
    wall_mesh : SurfaceMesh or None
        Vessel wall mesh for standalone confined mode.
    interface_mesh : SurfaceMesh or None
        When provided, enables Schwarz coupling mode.
        The mesh points are used as the Lagrangian positions for
        IB force spreading / velocity interpolation. Typically
        the same as body_mesh (the BEM surface IS the IB surface).
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
        self._mu = mu
        self._schwarz_mode = interface_mesh is not None

        if epsilon is None:
            epsilon = body_mesh.mean_spacing / 2.0
        self._epsilon = epsilon

        if self._schwarz_mode:
            self._init_schwarz(body_mesh, epsilon, mu)
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

    def _init_schwarz(self, body_mesh, epsilon, mu):
        """Schwarz mode: assemble body-only BEM system, LU-factorize.

        The BEM solves for body traction given body velocity minus
        background flow from the LBM. The traction is output for
        IB force spreading. No interface mesh in the BEM system.
        """
        N_b = body_mesh.n_points

        logger.info(
            "Schwarz mode: assembling body-only BEM system "
            "(N_body=%d, ε=%.4f)...",
            N_b, epsilon,
        )

        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)

        A = assemble_system_matrix(body_pts, body_wts, epsilon, mu)

        logger.info("LU-factorizing %d×%d body-only system...",
                     A.shape[0], A.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        self._N_body = N_b
        self._body_pts_jax = body_pts
        self._body_wts_jax = body_wts

        logger.info("Schwarz BEM system ready: %d body DOF", 3 * N_b)

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self) -> dict:
        state = {
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
        }
        if self._schwarz_mode:
            state["body_traction"] = jnp.zeros((self._N_body, 3))
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
                shape=(self._N_body, 3),
                default=jnp.zeros((self._N_body, 3)),
                description="LBM velocity at body surface [m/s]",
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
        """Schwarz: body-only BEM with IB background flow.

        1. RHS = U_body - u_background at each body surface point
        2. Solve body-only BEM for traction
        3. Extract force/torque from traction
        4. Output traction for IB force spreading to LBM
        """
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        bg_flow = boundary_inputs.get(
            "background_flow",
            jnp.zeros((self._N_body, 3)),
        )

        center = jnp.zeros(3)
        N_b = self._N_body

        # RHS = rigid body velocity - background flow at body points
        r = self._body_pts_jax - center
        u_body = U + jnp.cross(omega, r)  # (N_b, 3)
        rhs = (u_body - bg_flow).ravel()  # (3*N_b,)

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

        return {
            "drag_force": F,
            "drag_torque": T,
            "body_traction": body_traction,
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
            spec["body_traction"] = BoundaryFluxSpec(
                shape=(self._N_body, 3),
                description="BEM body surface traction [Pa]",
                output_units="Pa",
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
            fluxes["body_traction"] = state["body_traction"]
        return fluxes

    # -- FluidFieldProvider protocol -----------------------------------------

    def get_midplane_velocity(self, resolution: tuple[int, int]):
        return None


def make_stokeslet_rigid_body_edges(
    stokeslet_node_name: str,
    rigid_body_node_name: str,
) -> list[EdgeSpec]:
    """Return EdgeSpecs wiring StokesletFluidNode to RigidBodyNode."""
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
