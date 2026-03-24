"""IBLBMFluidNode -- 3D LBM fluid solver as a MADDENING SimulationNode.

Wraps the existing LBM utility functions (d3q19.py, bounce_back.py,
helix_geometry.py) as a proper MimeNode for node-graph coupling via
GraphManager. Replaces the manual loop in run_confinement_sweep.py:run_single()
with a single update() call per timestep.

Two-pass bounce-back: pipe wall (static, simple BB) then UMR body
(rotating, Bouzidi IBB or simple BB depending on use_bouzidi flag).

Reference: docs/architecture/iblbm_fluid_node_spec.md
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)

from mime.nodes.environment.lbm.d3q19 import (
    lbm_step_split,
    init_equilibrium,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    apply_bounce_back,
    apply_bouzidi_bounce_back,
    compute_q_values_sdf_sparse,
    compute_momentum_exchange_force,
    compute_momentum_exchange_torque,
)
from mime.nodes.environment.lbm.rotating_body import _rotation_velocity_field
from mime.nodes.robot.helix_geometry import create_umr_mask, umr_sdf


@stability(StabilityLevel.EXPERIMENTAL)
class IBLBMFluidNode(MimeNode):
    """3D IB-LBM fluid solver with Bouzidi IBB for confined microrobot flows.

    Wraps existing LBM functions as a node-graph component. Receives
    body angular velocity as a boundary input (from RigidBodyNode via edges)
    and outputs drag force/torque as boundary fluxes.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep (physical units).
    nx, ny, nz : int
        Lattice dimensions.
    tau : float
        BGK relaxation time.
    vessel_radius_lu : float
        Pipe wall radius in lattice units.
    body_geometry_params : dict
        Keyword arguments for ``create_umr_mask`` (must include nx, ny, nz
        and geometry dimensions in lattice units).
    use_bouzidi : bool
        If True, use Bouzidi interpolated bounce-back for the UMR surface.
        Pipe wall always uses simple bounce-back.
    dx_physical : float
        Physical lattice spacing [m] for unit conversion.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-010",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "3D IB-LBM fluid solver with Bouzidi IBB for confined "
            "microrobot flows"
        ),
        governing_equations=(
            "BGK-LBM D3Q19, Bouzidi interpolated bounce-back, "
            "momentum exchange force/torque"
        ),
        discretization="D3Q19 lattice Boltzmann with BGK collision operator",
        assumptions=(
            "Incompressible flow (Ma << 1 at all lattice nodes)",
            "Newtonian fluid",
            "Rigid body (no deformation)",
            "Single-device execution (no halo exchange)",
            "Rotation about z-axis only (body_angular_velocity[2])",
        ),
        limitations=(
            "No multi-GPU support (requires_halo=True blocks sharding)",
            "Per-step q-value recomputation (~0.1s at 192^3 on H100)",
            "First step triggers JAX compilation (30-60s at 192^3)",
        ),
        validated_regimes=(
            ValidatedRegime("Ma_tip", 0.0, 0.1, "",
                            "Mach number at fin tips must be < 0.1"),
            ValidatedRegime("tau", 0.55, 1.5, "",
                            "BGK relaxation time stability range"),
        ),
        references=(
            Reference("Bouzidi2001",
                       "Bouzidi et al. (2001) Phys. Fluids 13(11)"),
            Reference("Ladd1994",
                       "Ladd (1994) J. Fluid Mech. 271, 285-309"),
        ),
        hazard_hints=(
            "If max_boundary_links_per_dir is too small, jnp.nonzero "
            "silently truncates boundary links — accuracy degrades "
            "without error. Verify boundary link counts after geometry "
            "changes.",
        ),
        implementation_map={
            "BGK collision + streaming": (
                "mime.nodes.environment.lbm.d3q19.lbm_step_split"
            ),
            "Two-pass bounce-back": (
                "mime.nodes.environment.lbm.fluid_node."
                "IBLBMFluidNode.update"
            ),
            "Momentum exchange": (
                "mime.nodes.environment.lbm.bounce_back."
                "compute_momentum_exchange_torque"
            ),
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="iliac artery (confined UMR)",
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=3e-3, viscosity_max_pa_s=4e-3,
                temperature_min_c=36.0, temperature_max_c=38.0,
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        nx: int,
        ny: int,
        nz: int,
        tau: float,
        vessel_radius_lu: float,
        body_geometry_params: dict,
        use_bouzidi: bool = False,
        dx_physical: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            name, timestep,
            nx=nx, ny=ny, nz=nz,
            tau=tau,
            vessel_radius_lu=vessel_radius_lu,
            body_geometry_params=body_geometry_params,
            use_bouzidi=use_bouzidi,
            dx_physical=dx_physical,
            **kwargs,
        )

        cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
        self._center = (cx, cy, cz)

        # Static pipe wall mask
        ix = jnp.arange(nx, dtype=jnp.float32)
        iy = jnp.arange(ny, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        dist_2d = jnp.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        self._pipe_wall = jnp.broadcast_to(
            (dist_2d >= vessel_radius_lu)[..., None], (nx, ny, nz),
        )
        self._pipe_missing = compute_missing_mask(self._pipe_wall)

        # SDF kwargs (body_geometry_params without grid dimensions)
        self._sdf_kwargs = {
            k: v for k, v in body_geometry_params.items()
            if k not in ('nx', 'ny', 'nz')
        }

        # Precompute max_boundary_links_per_dir for Bouzidi path.
        # Must be computed at construction time (not inside JIT) because
        # it requires int() on a concrete JAX array.
        if use_bouzidi:
            initial_umr = create_umr_mask(
                center=self._center, rotation_angle=0.0,
                **body_geometry_params,
            )
            umr_missing = compute_missing_mask(initial_umr)
            counts = jnp.sum(umr_missing, axis=(1, 2, 3))
            max_count = int(jnp.max(counts))
            # 1.5x margin handles angle-to-angle variation for fixed geometry
            self._max_boundary_links_per_dir = int(max_count * 1.5) + 1
        else:
            self._max_boundary_links_per_dir = 0

    @property
    def requires_halo(self) -> bool:
        return True

    def initial_state(self) -> dict:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        geom = self.params["body_geometry_params"]

        initial_umr = create_umr_mask(
            center=self._center, rotation_angle=0.0, **geom,
        )
        return {
            "f": init_equilibrium(nx, ny, nz),
            "solid_mask": self._pipe_wall | initial_umr,
            "body_angle": jnp.array(0.0),
            "drag_force": jnp.zeros(3),
            "drag_torque": jnp.zeros(3),
        }

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body angular velocity [rad/s] from RigidBodyNode",
            ),
            "body_orientation": BoundaryInputSpec(
                shape=(4,),
                default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Body orientation quaternion from RigidBodyNode",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        tau = self.params["tau"]
        use_bouzidi = self.params["use_bouzidi"]
        geom = self.params["body_geometry_params"]
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        center = self._center

        omega_vec = boundary_inputs.get(
            "body_angular_velocity", jnp.zeros(3),
        )
        omega_z = omega_vec[2]

        # 1. Update angle (dt_lbm = 1 in lattice units)
        new_angle = state["body_angle"] + omega_z

        # 2. Generate UMR mask at new angle
        umr_mask = create_umr_mask(
            center=center, rotation_angle=new_angle, **geom,
        )
        umr_missing = compute_missing_mask(umr_mask)
        solid_mask = self._pipe_wall | umr_mask

        # 3. Wall velocity (omega x r)
        wall_vel = _rotation_velocity_field(
            (nx, ny, nz), omega_z, (0, 0, 1), center,
        )

        # 4. LBM collision + streaming
        f_pre, f_post, rho, u = lbm_step_split(state["f"], tau)

        # 5. Two-pass bounce-back
        # Pass 1: pipe wall (static, no wall velocity)
        f = apply_bounce_back(
            f_post, f_pre, self._pipe_missing, solid_mask,
            wall_velocity=None,
        )

        # Pass 2: UMR body (rotating)
        if use_bouzidi:
            sdf_kw = self._sdf_kwargs

            def sdf_func(pts):
                return umr_sdf(
                    pts, rotation_angle=new_angle, center=center,
                    **sdf_kw,
                )

            q_values = compute_q_values_sdf_sparse(
                umr_missing, sdf_func,
                max_boundary_links_per_dir=self._max_boundary_links_per_dir,
            )
            f = apply_bouzidi_bounce_back(
                f, f_pre, umr_missing, solid_mask,
                q_values, wall_velocity=wall_vel,
            )
        else:
            f = apply_bounce_back(
                f, f_pre, umr_missing, solid_mask,
                wall_velocity=wall_vel,
            )

        # 6. Momentum exchange force/torque
        body_center = jnp.array(center, dtype=jnp.float32)
        force = compute_momentum_exchange_force(f_pre, f, umr_missing)
        torque = compute_momentum_exchange_torque(
            f_pre, f, umr_missing, body_center,
        )

        return {
            "f": f,
            "solid_mask": solid_mask,
            "body_angle": new_angle,
            "drag_force": force,
            "drag_torque": torque,
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
        }
