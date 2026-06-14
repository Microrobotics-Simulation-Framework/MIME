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
import os
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
        length_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(name, timestep, mu=mu, **kwargs)

        self._body_mesh = body_mesh
        self._wall_mesh = wall_mesh
        self._mu = mu
        self._schwarz_mode = interface_mesh is not None
        self._N_body = body_mesh.n_points

        # SI-native confined mode: the wall TABLE is dimensionless (body radius ≡ 1,
        # R_cyl ≡ confinement ratio). When ``length_scale`` (the body radius, in the
        # SI units of ``body_mesh``/``mu``) is given, the body is normalised by it
        # for the table-based assembly, while the velocity BC and the output
        # force/torque/traction stay in SI — so the node is SI in, SI out and the
        # rest of the graph never sees the non-dimensionalisation. ``R_cyl`` is then
        # the table's dimensionless value (= R_vessel / length_scale). Default 1.0
        # keeps the legacy non-dimensional behaviour.
        self._L = float(length_scale) if length_scale is not None else 1.0

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
        self._R_si = self._extract_resistance_si()

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
        L = self._L
        body_pts_si = jnp.array(body_mesh.points)
        body_wts_si = jnp.array(body_mesh.weights)
        # Assemble in body-radius-normalised units (lengths/L, areas/L², ε/L) so the
        # dimensionless wall table applies; μ stays SI (the benchmark convention).
        body_pts_nd = body_pts_si / L
        body_wts_nd = body_wts_si / (L * L)
        epsilon_nd = epsilon / L

        self._check_centering(np.array(body_pts_nd), R_cyl)   # R_cyl is dimensionless

        logger.info(
            "Confined Schwarz mode: N_body=%d, R_cyl=%.3f, ε_nd=%.4f, L=%.4g",
            N_b, R_cyl, epsilon_nd, L,
        )

        A_body = assemble_system_matrix(body_pts_nd, body_wts_nd, epsilon_nd, mu)
        G_wall = assemble_image_correction_matrix_from_table(
            np.array(body_pts_nd), np.array(body_wts_nd), R_cyl, mu, wall_table,
        )
        A_conf = A_body + jnp.array(G_wall)

        logger.info("LU-factorizing %d×%d confined system...",
                     A_conf.shape[0], A_conf.shape[1])
        self._lu, self._piv = jax.scipy.linalg.lu_factor(A_conf)
        self._lu = np.array(self._lu)
        self._piv = np.array(self._piv)

        # SI points/weights drive the velocity BC (u = U + ω×r_SI) and the SI
        # force/torque extraction; the LU above is in normalised units (factor L).
        self._body_pts_jax = body_pts_si
        self._body_wts_jax = body_wts_si
        self._R_si = self._extract_resistance_si()

        # Reusable pieces for the OFF-CENTER resistance grid (resistance_grid_si):
        # A_body is translation-invariant (assembled once); only G_wall re-assembles
        # at each radial offset. Stored in body-radius-nd units (matching the table).
        self._A_body_nd = A_body
        self._body_pts_nd = body_pts_nd
        self._body_wts_nd = body_wts_nd
        self._wall_table = wall_table
        self._R_cyl = R_cyl

        logger.info("Confined Schwarz BEM system ready: %d body DOF",
                     3 * N_b)

    def _extract_resistance_si(self, lu=None, piv=None) -> np.ndarray:
        """SI 6×6 resistance matrix in the **body frame** from the factored
        Schwarz system.

        ``lu``/``piv`` default to the centered system's factors; pass an
        off-center system's factors (``resistance_grid_si``) to extract R(offset).

        Mirrors :meth:`_update_schwarz` (background flow = 0, identity
        orientation) for the six unit rigid motions — three unit translations
        and three unit rotations ``ω × r`` — reusing the LU factors. This is the
        same algebra the de-risk (``test_si_confined_bem``) and
        ``scripts/dejongh_benchmark.compute_R_matrix`` use; it recovers the
        validated de Jongh swim speed.

        Constant for a centred, axis-aligned rigid body (the wall table's regime
        of validity), so it can drive an **overdamped mobility body**: the
        physically-correct microswimmer model solves ``[V;ω] = R⁻¹·L_ext``
        algebraically each step (no inertia → first-order → the screw locks to
        the magnetic drive instead of librating). ``[F;T] = R·[U;ω]`` is the
        *reaction* convention (the force ON the body is ``−R·[U;ω]``).
        """
        lu = jnp.array(self._lu if lu is None else lu)
        piv = jnp.array(self._piv if piv is None else piv)
        pts = self._body_pts_jax
        wts = self._body_wts_jax
        center = jnp.zeros(3)
        r = pts - center
        e = jnp.eye(3)
        motions = [jnp.broadcast_to(e[i], pts.shape) for i in range(3)]
        motions += [jnp.cross(jnp.broadcast_to(e[i], pts.shape), r)
                    for i in range(3)]

        R = np.zeros((6, 6))
        for col, u_body in enumerate(motions):
            sol = jax.scipy.linalg.lu_solve((lu, piv), u_body.ravel())
            trac = sol.reshape(-1, 3) / self._L
            F, T = compute_force_torque(pts, wts, trac, center)
            R[:3, col] = np.asarray(F)
            R[3:, col] = np.asarray(T)
        # Symmetrize (Stokes reciprocity; Fourier-Bessel truncation breaks it ~1%),
        # matching scripts/dejongh_benchmark.compute_R_matrix.
        return 0.5 * (R + R.T)

    def resistance_grid_si(self, d_knots=None, n_knots: int = 8,
                           d_max: float | None = None,
                           cache_path: str | None = None):
        """Precompute SI 6×6 resistance matrices R(d) over RADIAL OFFSETS d.

        For a dense screw riding the tube floor (off-axis, near the wall), the
        CENTERED wall table underestimates propulsion. Exploiting the cylinder's
        axisymmetry, R at any lateral offset equals R(|offset|) rotated by the
        azimuth (handled at the consumer); here we tabulate the canonical R(d) for
        offsets along **body-x** (the cylinder axis is body-z).

        ``A_body`` is translation-invariant (reused), so each knot only re-assembles
        ``G_wall`` at the shifted points and re-factors — the
        ``dejongh_benchmark.compute_R_matrix`` template. ``d`` is in body-radius
        (``length_scale``) units, matching the table. Returns ``(d_knots, grid)``
        with ``grid.shape == (len(d_knots), 6, 6)`` (SI, body frame, symmetrized).

        Cost: ``len(d_knots)`` extra LU factorisations at call time (one-time);
        pass ``cache_path`` to persist/restore the grid.
        """
        if not self._schwarz_mode or not hasattr(self, "_wall_table"):
            raise RuntimeError(
                "resistance_grid_si requires confined Schwarz mode")
        from .cylinder_wall_table import (
            assemble_image_correction_matrix_from_table)

        pts_nd = np.asarray(self._body_pts_nd)
        wts_nd = np.asarray(self._body_wts_nd)
        # Body's max radial (x-y) extent in nd units → max safe offset to the wall.
        r_body_max = float(np.max(np.hypot(pts_nd[:, 0], pts_nd[:, 1])))
        if d_max is None:
            d_max = max(0.0, (self._R_cyl - r_body_max) * 0.95)
        if d_knots is None:
            u = np.linspace(0.0, 1.0, int(n_knots))
            d_knots = d_max * (2.0 * u - u * u)          # clustered toward the wall
        d_knots = np.asarray(d_knots, dtype=float)

        if cache_path is not None and os.path.exists(cache_path):
            cached = np.load(cache_path)
            if (cached["d_knots"].shape == d_knots.shape
                    and np.allclose(cached["d_knots"], d_knots)):
                logger.info("resistance_grid_si: loaded cache %s", cache_path)
                return cached["d_knots"], cached["grid"]

        A_body = self._A_body_nd
        grid = np.zeros((len(d_knots), 6, 6))
        for k, d in enumerate(d_knots):
            if d <= 0.0:
                grid[k] = self._R_si                      # centered (already have it)
                continue
            pts_shifted = pts_nd + np.array([float(d), 0.0, 0.0])
            G = assemble_image_correction_matrix_from_table(
                pts_shifted, wts_nd, self._R_cyl, self._mu, self._wall_table)
            A_conf = A_body + jnp.array(G)
            lu, piv = jax.scipy.linalg.lu_factor(A_conf)
            grid[k] = self._extract_resistance_si(lu=np.array(lu), piv=np.array(piv))
            logger.info("resistance_grid_si: d=%.4f (%d/%d)", d, k + 1, len(d_knots))

        if cache_path is not None:
            np.savez(cache_path, d_knots=d_knots, grid=grid)
        return d_knots, grid

    def resistance_matrix_si(self) -> np.ndarray:
        """The SI 6×6 body-frame resistance matrix.

        Standalone mode returns the precomputed ``self._R``; Schwarz mode the
        resistance extracted from the factored interface system at init. Used to
        build the overdamped mobility body (``M = R⁻¹``)."""
        if self._schwarz_mode:
            return np.array(self._R_si)
        return np.array(self._R)

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
    def static_data(self) -> dict:
        """BEM operators closed over by :meth:`update`, declared on the
        v0.2 ``static_data`` channel.

        Standalone mode exposes the 6×6 resistance matrix; Schwarz mode
        exposes the LU factors of the body BEM system plus the body
        surface points and quadrature weights.  All
        ``replication="replicate"`` — a regularised-Stokeslet BEM node
        is a single dense solve and is never sharded.

        ``static_data`` is not checkpointed; these operators are
        rebuilt in ``__init__`` from the body / wall meshes and wall
        table passed to the constructor.
        """
        from maddening.core.static_data import StaticArray
        if self._schwarz_mode:
            return {
                "bem_lu": StaticArray(self._lu),
                "bem_piv": StaticArray(self._piv),
                "body_points": StaticArray(self._body_pts_jax),
                "body_weights": StaticArray(self._body_wts_jax),
            }
        return {
            "resistance_matrix": StaticArray(self._R),
        }

    def initial_state(self) -> dict:
        state = {
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
        }
        if self._schwarz_mode:
            state["body_traction"] = jnp.zeros((self._N_body, 3))
            # Force/torque ON the body from the background flow alone (body held
            # at rest). Lets an overdamped mobility body solve the force balance
            # [V;ω] = R⁻¹·(L_ext + L_bg) one-shot — see _update_schwarz.
            state["background_force"] = jnp.zeros(3)
            state["background_torque"] = jnp.zeros(3)
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
            spec["background_force"] = BoundaryFluxSpec(
                shape=(3,),
                description="Force on body from background flow alone [N]",
                output_units="N",
            )
            spec["background_torque"] = BoundaryFluxSpec(
                shape=(3,),
                description="Torque on body from background flow alone [N·m]",
                output_units="N*m",
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

        **Frame-aware:** the BEM (and its confined wall table) is built in the BODY
        frame, where the body's long axis is body-z = the cylinder axis. The graph
        speaks WORLD frame, so the rigid-body velocity, angular velocity and the
        sampled background flow are rotated world→body by ``body_orientation`` before
        the solve, and the resulting traction / drag-force / drag-torque rotated
        body→world after. With identity orientation (body=world) this is a no-op, so
        the legacy axis-aligned behaviour (and TASK A / the de-risk) is unchanged —
        but a vessel along world-x (ar4 axes), with the screw rotated body-z→world-x,
        is now handled correctly.
        """
        from mime.core.quaternion import rotate_vector, rotate_vector_inverse

        omega_w = boundary_inputs.get("body_angular_velocity", jnp.zeros(3))
        U_w = boundary_inputs.get("body_velocity", jnp.zeros(3))
        bg_w = boundary_inputs.get(
            "background_flow",
            jnp.zeros((self._N_body, 3)),
        )
        q = boundary_inputs.get("body_orientation", jnp.array([1.0, 0.0, 0.0, 0.0]))

        # world → body (no-op when q is identity)
        U = rotate_vector_inverse(q, U_w)
        omega = rotate_vector_inverse(q, omega_w)
        bg_flow = jax.vmap(lambda v: rotate_vector_inverse(q, v))(bg_w)

        center = jnp.zeros(3)
        N_b = self._N_body

        r = self._body_pts_jax - center
        u_body = U + jnp.cross(omega, r)
        rhs = (u_body - bg_flow).ravel()

        lu = jnp.array(self._lu)
        piv = jnp.array(self._piv)
        solution = jax.scipy.linalg.lu_solve((lu, piv), rhs)

        # The LU is in body-radius-normalised units: A_nd = A_phys / L, so the raw
        # solve gives t_solve = L·t_phys. Dividing by L recovers SI traction [Pa];
        # then compute_force_torque on the SI points/weights yields SI F [N], T [N·m].
        # (L = 1 in the legacy non-dimensional mode — a no-op there.)
        body_traction = solution.reshape(N_b, 3) / self._L
        F, T = compute_force_torque(
            self._body_pts_jax, self._body_wts_jax,
            body_traction, center,
        )

        # Background-only load (force/torque ON the body from the ambient flow,
        # body held at rest): solve A·t_bg = bg_flow and integrate. The full
        # reaction decomposes as R·[V;ω] − F_bg, so the force on the body splits
        # into the −R·[V;ω] resistance and this +F_bg background load — exposing
        # F_bg lets an overdamped body solve [V;ω] = R⁻¹·(L_ext + F_bg) one-shot
        # (no dependence on its own velocity → no coupling-iteration oscillation,
        # unlike folding it into the motion-coupled drag).
        sol_bg = jax.scipy.linalg.lu_solve((lu, piv), bg_flow.ravel())
        trac_bg = sol_bg.reshape(N_b, 3) / self._L
        F_bg, T_bg = compute_force_torque(
            self._body_pts_jax, self._body_wts_jax, trac_bg, center)

        # body → world (no-op when q is identity): the drag/torque go back to the
        # body and the traction back to the FVM forcing, both world-frame.
        F = rotate_vector(q, F)
        T = rotate_vector(q, T)
        body_traction = jax.vmap(lambda v: rotate_vector(q, v))(body_traction)
        F_bg = rotate_vector(q, F_bg)
        T_bg = rotate_vector(q, T_bg)

        return {
            "drag_force": F,
            "drag_torque": T,
            "body_traction": body_traction,
            "background_force": F_bg,
            "background_torque": T_bg,
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
            fluxes["background_force"] = state["background_force"]
            fluxes["background_torque"] = state["background_torque"]
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
