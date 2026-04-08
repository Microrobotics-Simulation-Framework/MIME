"""StokesletFluidNode — regularised Stokeslet BEM fluid solver.

A quasi-static Stokes flow solver for confined microrobot FSI.
Computes drag force and torque on a rigid body via a precomputed
6×6 resistance matrix (standalone mode) or via LU backsubstitution
with background flow correction (Schwarz coupling mode).

No Mach number constraint — operates at any rotation frequency.

Architecture: hybrid BEM + volumetric solver
-------------------------------------------------
Each solver covers the other's weakness:

**BEM + G_wall** (this node) resolves body drag exactly — no Mach
constraint, direction-independent, <4% accuracy for sphere κ=0.3.
But as a surface method it cannot compute volumetric effects:
pulsatile background flow, acoustic streaming, or multi-robot wake
coupling.

**LBM** (IBLBMFluidNode) resolves volumetric flow but cannot
resolve the robot body at clinical frequencies (Ma ≈ 14 for a
d=1.74 mm UMR at 128 Hz in water). The Peskin delta kernel used
in IB-LBM coupling also creates direction-dependent velocity
transfer errors.

**The hybrid** sidesteps both: BEM + G_wall handles body drag.
LBM handles volumetric flow with NO robot body in the LBM domain
— the BEM traction is spread as volume force density (Force
Coupling Method, Lomholt & Maxey 2003), avoiding both Mach number
and IB kernel problems.

Modes
-----
Standalone (no interface_mesh):
    Resistance matrix R computed at init. update() is a 6×6 matvec.
    Variants: unconfined, explicit-wall, or cylinder-confined (via
    Liron-Shahar wall table).

Schwarz coupling (interface_mesh provided):
    Body BEM system (optionally including G_wall) assembled and
    LU-factorized at init. update() builds the RHS as
    (U_body - u_background), backsubstitutes for body traction, and
    extracts force/torque.

    When wall_table is provided in Schwarz mode, the BEM system is
    A_body + G_wall. There is NO double-counting with the LBM wall:
    the LBM computes the volumetric background flow (pulsatile,
    wakes); the BEM computes the body's perturbation relative to that
    background. G_wall captures how the cylinder wall modifies this
    perturbation. The LBM has the physical wall BCs; G_wall has the
    perturbation image. They compute different quantities.
"""

from __future__ import annotations

import logging
import warnings

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
    compute_dlp_rhs_correction,
    compute_force_torque,
    solve_bem_multi_rhs,
)

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
        Vessel wall mesh for explicit-wall standalone mode.
    interface_mesh : SurfaceMesh or None
        When provided, enables Schwarz coupling mode.
    wall_table : WallTable or None
        Precomputed Liron-Shahar cylindrical Green's function table.
        When provided, the analytical wall correction is baked into
        the BEM system matrix at init. Requires ``R_cyl``.
    R_cyl : float or None
        Cylinder radius [m]. Required when ``wall_table`` is provided.
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
        wall_table=None,
        R_cyl: float | None = None,
        epsilon: float | None = None,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu, **kwargs)

        self._body_mesh = body_mesh
        self._wall_mesh = wall_mesh
        self._mu = mu
        self._schwarz_mode = interface_mesh is not None
        self._N_body = body_mesh.n_points

        if epsilon is None:
            epsilon = body_mesh.mean_spacing / 2.0
        self._epsilon = epsilon

        if wall_table is not None and R_cyl is None:
            raise ValueError("R_cyl is required when wall_table is provided")

        # ── Mode dispatch ─────────────────────────────────────────
        # The wall_table provides the Liron-Shahar cylindrical Green's
        # function image correction. It handles the wall's hydrodynamic
        # effect on the body perturbation flow analytically, without
        # discretizing the wall.
        #
        # When combined with LBM background flow (Schwarz mode), there
        # is NO double-counting: the LBM computes the volumetric
        # background (pulsatile, wakes) and the BEM computes the body's
        # perturbation relative to that background. G_wall captures how
        # the cylinder wall modifies the perturbation. The LBM has the
        # physical wall BCs; G_wall has the perturbation image.
        if self._schwarz_mode:
            if wall_table is not None:
                self._init_confined_schwarz(
                    body_mesh, wall_table, R_cyl, epsilon, mu)
            else:
                self._init_schwarz(body_mesh, epsilon, mu)
        else:
            if wall_table is not None:
                self._init_confined_standalone(
                    body_mesh, wall_table, R_cyl, epsilon, mu)
            elif wall_mesh is not None:
                self._init_standalone(body_mesh, wall_mesh, epsilon, mu)
            else:
                self._init_standalone(body_mesh, None, epsilon, mu)

    # ── Init methods ──────────────────────────────────────────────

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

    def _init_confined_standalone(self, body_mesh, wall_table, R_cyl,
                                  epsilon, mu):
        """Cylinder-confined standalone: Liron-Shahar wall table → R (6×6).

        Assembles A_confined = A_body_BEM + G_wall from the precomputed
        wall table, solves 6 BEM problems with DLP correction, and
        extracts the 6×6 resistance matrix.
        """
        from .cylinder_wall_table import assemble_image_correction_matrix_from_table

        N_b = body_mesh.n_points
        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)
        body_nml = jnp.array(body_mesh.normals)
        center = jnp.zeros(3)

        self._check_centering(np.array(body_pts), R_cyl)

        logger.info(
            "Confined standalone mode: N_body=%d, R_cyl=%.3f, ε=%.4f",
            N_b, R_cyl, epsilon,
        )

        # Free-space BEM + analytical wall correction
        A_body = assemble_system_matrix(body_pts, body_wts, epsilon, mu)
        G_wall = assemble_image_correction_matrix_from_table(
            np.array(body_pts), np.array(body_wts), R_cyl, mu, wall_table,
        )
        A_conf = A_body + jnp.array(G_wall)

        # Solve 6 BEM problems → 6×6 R
        e = jnp.eye(3)
        zero = jnp.zeros(3)
        rhs_cols = []
        for i in range(3):
            r = body_pts - center
            vel = e[i] + jnp.cross(zero, r)
            rhs_cols.append(compute_dlp_rhs_correction(
                body_pts, body_nml, body_wts, vel, epsilon))
        for i in range(3):
            r = body_pts - center
            vel = zero + jnp.cross(e[i], r)
            rhs_cols.append(compute_dlp_rhs_correction(
                body_pts, body_nml, body_wts, vel, epsilon))

        rhs_matrix = jnp.stack(rhs_cols, axis=1)
        solutions = solve_bem_multi_rhs(A_conf, rhs_matrix)

        R = jnp.zeros((6, 6))
        for col in range(6):
            trac = solutions[:, col].reshape(N_b, 3)
            F, T = compute_force_torque(body_pts, body_wts, trac, center)
            R = R.at[:3, col].set(F)
            R = R.at[3:, col].set(T)

        self._R = np.array(R)
        logger.info("Confined resistance matrix computed: R shape %s",
                     self._R.shape)

    def _init_schwarz(self, body_mesh, epsilon, mu):
        """Schwarz mode: assemble body-only BEM system, LU-factorize."""
        N_b = body_mesh.n_points
        logger.info(
            "Schwarz mode: assembling body-only BEM system "
            "(N_body=%d, ε=%.4f)...", N_b, epsilon,
        )

        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)

        A = assemble_system_matrix(body_pts, body_wts, epsilon, mu)

        logger.info("LU-factorizing %d×%d body-only system...",
                     A.shape[0], A.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        self._body_pts_jax = body_pts
        self._body_wts_jax = body_wts

        logger.info("Schwarz BEM system ready: %d body DOF", 3 * N_b)

    def _init_confined_schwarz(self, body_mesh, wall_table, R_cyl,
                               epsilon, mu):
        """Confined Schwarz: A_body + G_wall LU-factored for runtime solve.

        Same as Schwarz mode but with the Liron-Shahar wall correction
        baked into the system matrix. The update() method is identical
        to _update_schwarz — the LU factors already include the wall.
        """
        from .cylinder_wall_table import assemble_image_correction_matrix_from_table

        N_b = body_mesh.n_points
        body_pts = jnp.array(body_mesh.points)
        body_wts = jnp.array(body_mesh.weights)

        self._check_centering(np.array(body_pts), R_cyl)

        logger.info(
            "Confined Schwarz mode: N_body=%d, R_cyl=%.3f, ε=%.4f",
            N_b, R_cyl, epsilon,
        )

        A_body = assemble_system_matrix(body_pts, body_wts, epsilon, mu)
        G_wall = assemble_image_correction_matrix_from_table(
            np.array(body_pts), np.array(body_wts), R_cyl, mu, wall_table,
        )
        A_conf = A_body + jnp.array(G_wall)

        logger.info("LU-factorizing %d×%d confined system...",
                     A_conf.shape[0], A_conf.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A_conf)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        self._body_pts_jax = body_pts
        self._body_wts_jax = body_wts

        logger.info("Confined Schwarz BEM system ready: %d body DOF",
                     3 * N_b)

    @staticmethod
    def _check_centering(body_pts_np, R_cyl):
        """Assert body is inside cylinder and warn if off-axis."""
        rho = np.sqrt(body_pts_np[:, 0]**2 + body_pts_np[:, 1]**2)
        if np.any(rho >= R_cyl):
            raise ValueError(
                f"Body extends outside cylinder: max(ρ)={rho.max():.3f} "
                f"≥ R_cyl={R_cyl:.3f}"
            )
        centroid_offset = np.sqrt(
            np.mean(body_pts_np[:, 0])**2 +
            np.mean(body_pts_np[:, 1])**2
        )
        if centroid_offset > 0.05 * R_cyl:
            warnings.warn(
                f"Body centroid is {centroid_offset:.3f} off cylinder axis "
                f"(R_cyl={R_cyl:.3f}). Wall table is only valid for "
                f"centered bodies. Use Level 2+ (LBM background) for "
                f"off-axis motion.",
                stacklevel=3,
            )

    # ── State and ports ───────────────────────────────────────────

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
            # Stub: returns zeros when not connected. Wired at Level 3
            # for Force Coupling Method spreading into LBM.
            "body_force_density": BoundaryFluxSpec(
                shape=(self._N_body, 3),
                description="Force density for LBM spreading [N/m³]",
                output_units="N/m^3",
            ),
        }
        if self._schwarz_mode:
            spec["body_traction"] = BoundaryFluxSpec(
                shape=(self._N_body, 3),
                description="BEM body surface traction [Pa]",
                output_units="Pa",
            )
        return spec

    # ── Update ────────────────────────────────────────────────────

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
        """Schwarz: body BEM with background flow correction.

        Works identically for unconfined and confined modes — the
        LU factors already include the wall correction if wall_table
        was provided at init.
        """
        omega = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U = boundary_inputs.get("body_velocity", jnp.zeros(3))
        bg_flow = boundary_inputs.get(
            "background_flow",
            jnp.zeros((self._N_body, 3)),
        )

        center = jnp.zeros(3)
        N_b = self._N_body

        r = self._body_pts_jax - center
        u_body = U + jnp.cross(omega, r)
        rhs = (u_body - bg_flow).ravel()

        lu = jnp.array(self._lu)
        piv = jnp.array(self._piv)
        solution = jax.scipy.linalg.lu_solve((lu, piv), rhs)

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

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        fluxes = {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
            # Stub: zeros until Level 3 force spreading is implemented.
            # The transform (traction → volume force density) sits on
            # the edge, not in this node.
            "body_force_density": jnp.zeros((self._N_body, 3)),
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
