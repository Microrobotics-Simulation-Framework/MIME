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

import jax
import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, BoundaryFluxSpec
from maddening.core.edge import EdgeSpec
from maddening.core.transforms import lbm_to_si_force, lbm_to_si_torque
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
    compute_macroscopic,
    collide_bgk,
    stream_padded,
)
from mime.nodes.environment.lbm.bounce_back import (
    compute_missing_mask,
    compute_missing_mask_sharded,
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
            "No multi-GPU support (non-empty halo_width blocks sharding)",
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
        multigpu_shard_axis: int | None = None,
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
            multigpu_shard_axis=multigpu_shard_axis,
            **kwargs,
        )
        # Multi-GPU sharding (fit-up §8 Step 5, requires maddening>=0.2.1):
        # when set, this is the spatial axis (0=x / 1=y / 2=z) along which a
        # wrapping ShardedStencilNode decomposes the lattice. The pipe-wall
        # masks declare `replication="shard"` on the corresponding axis;
        # `update_padded()` reads per-shard slabs from `static_padded` and
        # the slab's coordinate offset from `shard_info`.
        self._multigpu_shard_axis = multigpu_shard_axis

        cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
        self._center = (cx, cy, cz)
        self._latest_velocity = None  # FluidFieldProvider: stashed each step

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

    def halo_width(self) -> dict[int, int]:
        """D3Q19 streaming reads one neighbour per spatial axis."""
        return {0: 1, 1: 1, 2: 1}

    @property
    def static_data(self) -> dict:
        """Non-evolving lattice masks closed over by :meth:`update` /
        :meth:`update_padded`.

        The pipe-wall occupancy mask and its D3Q19 missing-link mask are
        fixed vessel geometry — they never change once the node is built.
        Exposing them via the v0.2 ``static_data`` channel keeps them out
        of every ``scan`` / checkpoint pass.

        When ``multigpu_shard_axis`` is set the masks are declared
        ``replication="shard"`` along that axis — under MADDENING v0.2.1's
        ``ShardedStencilNode`` each device receives its per-slab slice +
        halos via ``update_padded``'s ``static_padded`` kwarg. Otherwise
        the masks are replicated (the single-device path).

        Both masks are rebuilt from ``self.params`` in ``__init__``; a
        checkpoint/restore round-trip reconstructs them from the persisted
        params, since ``static_data`` itself is not checkpointed.
        """
        from maddening.core.static_data import StaticArray
        ax = self._multigpu_shard_axis
        if ax is None:
            return {
                "pipe_wall": StaticArray(self._pipe_wall),
                "pipe_missing": StaticArray(self._pipe_missing),
            }
        # Sharded path: only ``pipe_wall`` is sharded. It is ``(nx,ny,nz)`` so
        # its array ``shard_axis`` equals the spatial axis ``ax`` —
        # ShardedStencilNode requires a static's shard_axis to be one of the
        # spatial axes it shards (per ``axis_map.values()``). The precomputed
        # ``pipe_missing`` is ``(19,nx,ny,nz)`` with the Q-axis leading, so its
        # spatial axis would be ``ax+1`` — which is *not* a sharded spatial
        # axis and the wrapper rejects it. Rather than carry a transposed
        # copy, ``update_padded`` recomputes the pipe (and UMR) missing-link
        # masks per slab from the halo-exchanged ``pipe_wall`` via
        # :func:`compute_missing_mask_sharded`.
        return {
            "pipe_wall": StaticArray(
                self._pipe_wall,
                replication="shard", shard_axis=ax,
            ),
        }

    def initial_state(self) -> dict:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]

        # The pipe wall lives in ``static_data``; the UMR occupancy is
        # recomputed from ``body_angle`` each step inside ``update()``.
        # Neither belongs in the state pytree.
        return {
            "f": init_equilibrium(nx, ny, nz),
            "body_angle": jnp.array(0.0, dtype=jnp.float32),
            "drag_force": jnp.zeros(3, dtype=jnp.float32),
            "drag_torque": jnp.zeros(3, dtype=jnp.float32),
        }

    def state_fields(self) -> list[str]:
        """The evolving state pytree (fit-up §8 Step 5).

        ``drag_force`` / ``drag_torque`` live in ``initial_state`` for
        single-device compatibility, but they are *outputs* of each step
        (domain integrals over the lattice), not evolving state. Excluding
        them here tells MADDENING's ``ShardedStencilNode`` to treat them
        through :meth:`domain_integral_fields` instead of halo-stripping
        them as spatial state.
        """
        return ["f", "body_angle"]

    def domain_integral_fields(self) -> set[str]:
        """Output fields ``ShardedStencilNode`` cross-device-``psum``s.

        ``drag_force`` and ``drag_torque`` are sums of a momentum-exchange
        field over every lattice cell. Under sharding each device produces
        a partial sum; MADDENING v0.2.1's ``ShardedStencilNode`` reads this
        declaration and ``lax.psum``s the named fields across the device
        mesh after :meth:`update_padded` returns.
        """
        return {"drag_force", "drag_torque"}

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body angular velocity [rad/step] in lattice units",
                expected_units="lattice",
            ),
            "body_orientation": BoundaryInputSpec(
                shape=(4,),
                default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description="Body orientation quaternion (dimensionless)",
                expected_units="lattice",
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

        # The LBM solver is float32 by design (see init_equilibrium's
        # explicit dtype).  Cast the boundary input so a graph running
        # under jax_enable_x64 cannot silently promote wall_vel — and
        # therefore the streamed f — to float64.
        omega_vec = boundary_inputs.get(
            "body_angular_velocity", jnp.zeros(3),
        ).astype(jnp.float32)
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
        self._latest_velocity = u  # FluidFieldProvider: stash for get_midplane_velocity

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
            "body_angle": new_angle,
            "drag_force": force,
            "drag_torque": torque,
        }

    def update_padded(
        self,
        state_padded: dict,
        boundary_inputs: dict,
        dt: float,
        *,
        static_padded: dict | None = None,
        shard_info: dict | None = None,
    ) -> dict:
        """Halo-aware step for ShardedStencilNode wrapping (fit-up §8 Step 5).

        Conforms to MADDENING v0.2.1's sharded-stencil contract: a
        halo-padded ``f`` (per :meth:`halo_width` = ``{0:1, 1:1, 2:1}``),
        per-shard pipe-mask slabs in ``static_padded``, and the slab's
        coordinate offset in ``shard_info``; returns the same padded shape
        for ``f`` plus partial-sum ``drag_force`` / ``drag_torque`` that
        ``ShardedStencilNode`` ``lax.psum``s across the device mesh per
        :meth:`domain_integral_fields`.

        **Single-device fallback.** Called outside ``ShardedStencilNode``
        (no ``static_padded`` / ``shard_info``): strip halos, delegate to
        :meth:`update` on the un-padded interior, then re-pad ``f``
        periodically (matches ``lbm_step_split``'s ``jnp.roll``).

        **Multi-device path** (``shard_info`` / ``static_padded`` present):
        collide on the full halo-padded ``f``; stream via the slice-based
        :func:`stream_padded` (halos carry the neighbour slab's
        post-collision populations); recompute the pipe + UMR missing-link
        masks per slab from the halo-exchanged ``pipe_wall`` and the UMR
        geometry rebuilt on the slab's *global* coordinate range
        (``shard_info`` offset); two-pass bounce-back on the interior; and
        partial-sum ``drag_force`` / ``drag_torque`` that the wrapper
        ``lax.psum``s across the mesh. See :meth:`_update_padded_sharded`.

        Bouzidi IBB on the sharded path needs per-slab SDF q-value
        recomputation and is deferred (simple bounce-back only).
        """
        if self.params.get("use_bouzidi", False):
            raise NotImplementedError(
                "IBLBMFluidNode.update_padded does not yet support "
                "use_bouzidi=True; use simple bounce-back for sharded "
                "execution"
            )

        if shard_info is not None or static_padded is not None:
            return self._update_padded_sharded(
                state_padded, boundary_inputs, dt, static_padded, shard_info,
            )

        # --- Single-device fallback ---------------------------------------
        h = 1
        interior_f = state_padded["f"][h:-h, h:-h, h:-h, :]
        state_unpadded = {
            "f": interior_f,
            "body_angle": state_padded["body_angle"],
            "drag_force": jnp.zeros(3, dtype=jnp.float32),
            "drag_torque": jnp.zeros(3, dtype=jnp.float32),
        }
        new = self.update(state_unpadded, boundary_inputs, dt)
        f_out = jnp.pad(
            new["f"], ((h, h), (h, h), (h, h), (0, 0)), mode="wrap",
        )
        return {
            "f": f_out,
            "body_angle": new["body_angle"],
            "drag_force": new["drag_force"],
            "drag_torque": new["drag_torque"],
        }

    def _update_padded_sharded(
        self, state_padded, boundary_inputs, dt, static_padded, shard_info,
    ) -> dict:
        """Multi-device halo-aware step (simple bounce-back).

        Mirrors :meth:`update`'s collision → streaming → two-pass bounce-back
        → momentum-exchange pipeline, but on a halo-padded slab decomposed
        along ``self._multigpu_shard_axis``. The pipe wall arrives as a
        halo-exchanged ``static_padded["pipe_wall"]`` slab; the UMR body and
        its rotation-velocity field are rebuilt on the slab's global
        coordinate range (from ``shard_info``); ``drag_force`` /
        ``drag_torque`` are per-slab partial sums that ``ShardedStencilNode``
        ``lax.psum``s across the device mesh.
        """
        halo = 1
        ax = self._multigpu_shard_axis
        if ax is None:
            raise ValueError(
                "IBLBMFluidNode.update_padded reached the sharded path but "
                "multigpu_shard_axis is None — construct the node with "
                "multigpu_shard_axis set to the sharded spatial axis."
            )
        tau = self.params["tau"]
        geom = self.params["body_geometry_params"]
        center = self._center
        nx, ny, nz = self.params["nx"], self.params["ny"], self.params["nz"]

        # Slab geometry along the shard axis: global offset of the first
        # interior cell (a traced scalar) and the unpadded local extent.
        offset, extent = shard_info[ax]

        f_pad = state_padded["f"]  # halo-padded on all 3 spatial axes

        omega_vec = boundary_inputs.get(
            "body_angular_velocity", jnp.zeros(3),
        ).astype(jnp.float32)
        omega_z = omega_vec[2]
        new_angle = state_padded["body_angle"] + omega_z

        # 1. Collision (local) on the full padded array, then halo-aware
        #    streaming (reads neighbour populations from the exchanged halos).
        density_pad, velocity_pad = compute_macroscopic(f_pad)
        f_post_collision_pad = collide_bgk(f_pad, density_pad, velocity_pad, tau)
        f_post_stream = stream_padded(f_post_collision_pad, halo=halo)  # interior
        interior = (slice(halo, -halo),) * 3 + (slice(None),)
        f_pre = f_post_collision_pad[interior]  # interior post-collision

        # 2. Per-slab masks. pipe_wall arrives halo-padded (edge boundary) on
        #    the shard axis; the UMR body is rebuilt on the slab's global
        #    coordinates (its first padded cell is at global offset-halo).
        pipe_wall_pad = static_padded["pipe_wall"].astype(jnp.bool_)
        pad_shape = [nx, ny, nz]
        pad_shape[ax] = extent + 2 * halo
        umr_origin = [0.0, 0.0, 0.0]
        umr_origin[ax] = offset - halo
        # ``geom`` carries the full-grid nx/ny/nz; override them with the
        # padded-slab dimensions so create_umr_mask builds the slab's piece.
        geom_slab = {
            **geom, "nx": pad_shape[0], "ny": pad_shape[1], "nz": pad_shape[2],
        }
        umr_pad = create_umr_mask(
            center=center, rotation_angle=new_angle,
            origin=tuple(umr_origin), **geom_slab,
        )

        pipe_missing = compute_missing_mask_sharded(pipe_wall_pad, ax, halo)
        umr_missing = compute_missing_mask_sharded(umr_pad, ax, halo)

        # Combined interior solid (vestigial arg to apply_bounce_back).
        pipe_int = jax.lax.slice_in_dim(pipe_wall_pad, halo, halo + extent, axis=ax)
        umr_int = jax.lax.slice_in_dim(umr_pad, halo, halo + extent, axis=ax)
        solid_int = pipe_int | umr_int

        # 3. Wall velocity (omega x r) at interior cells, in global coords.
        interior_shape = [nx, ny, nz]
        interior_shape[ax] = extent
        wall_origin = [0.0, 0.0, 0.0]
        wall_origin[ax] = offset
        wall_vel = _rotation_velocity_field(
            tuple(interior_shape), omega_z, (0, 0, 1), center,
            origin=tuple(wall_origin),
        )

        # 4. Two-pass bounce-back on the interior (pipe static, UMR rotating).
        f_bb = apply_bounce_back(
            f_post_stream, f_pre, pipe_missing, solid_int, wall_velocity=None,
        )
        f_bb = apply_bounce_back(
            f_bb, f_pre, umr_missing, solid_int, wall_velocity=wall_vel,
        )

        # 5. Momentum exchange — per-slab partials; wrapper psums them.
        body_center = jnp.array(center, dtype=jnp.float32)
        force = compute_momentum_exchange_force(f_pre, f_bb, umr_missing)
        torque = compute_momentum_exchange_torque(
            f_pre, f_bb, umr_missing, body_center, origin=tuple(wall_origin),
        )

        # 6. Re-pad f so the wrapper strips halos uniformly across fields.
        f_out = f_pad.at[interior].set(f_bb)

        return {
            "f": f_out,
            "body_angle": new_angle,
            "drag_force": force,
            "drag_torque": torque,
        }

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "drag_force": BoundaryFluxSpec(
                shape=(3,),
                description="Momentum exchange force on body",
                output_units="lattice",
            ),
            "drag_torque": BoundaryFluxSpec(
                shape=(3,),
                description="Momentum exchange torque on body",
                output_units="lattice",
            ),
        }

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {
            "drag_force": state["drag_force"],
            "drag_torque": state["drag_torque"],
        }

    # -- FluidFieldProvider protocol -----------------------------------------

    def get_midplane_velocity(
        self,
        resolution: tuple[int, int],
    ) -> "np.ndarray | None":
        """Return (nx, ny) velocity magnitude at the Z-midplane.

        Uses the velocity field stashed during the most recent update().
        Downsamples to the requested resolution if needed.
        """
        if self._latest_velocity is None:
            return None
        import numpy as np
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


def _make_si_to_lattice_omega(dt_physical: float, omega_max_lattice: float = 0.005):
    """Create a transform converting angular velocity from rad/s to rad/step.

    Includes a safety clamp to prevent Ma > 0.1 in the LBM. At step 0 of
    FSI coupling, the back-edge carries zero drag, so the overdamped
    RigidBodyNode can produce an unphysically large omega. The clamp
    prevents this from destabilising the LBM.

    Parameters
    ----------
    dt_physical : float
        Physical timestep per LBM step [s].
    omega_max_lattice : float
        Maximum allowed angular velocity in lattice units [rad/step].
        Default 0.005 keeps Ma < 0.1 at typical fin radii.
    """
    factor = float(dt_physical)
    clamp = float(omega_max_lattice)

    def _convert(omega_si):
        omega_lat = omega_si * factor
        return jnp.clip(omega_lat, -clamp, clamp)

    _convert.__qualname__ = (
        f"si_to_lattice_omega(dt={dt_physical}, "
        f"clamp={omega_max_lattice})"
    )
    return _convert


def make_iblbm_rigid_body_edges(
    lbm_node_name: str,
    rigid_body_node_name: str,
    dx_physical: float,
    dt_physical: float,
    rho_physical: float = 1060.0,
    omega_max_lattice: float = 0.005,
) -> list[EdgeSpec]:
    """Return EdgeSpecs for wiring IBLBMFluidNode to RigidBodyNode.

    Forward edges carry drag force/torque from LBM to rigid body with
    LBM-to-SI unit transforms. Back-edges carry angular velocity and
    orientation from rigid body back to LBM (no transform needed —
    angular velocity in rad/step is handled by the caller's unit mapping,
    and orientation is dimensionless).

    Back-edges are detected automatically by GraphManager during
    ``compile()`` based on topological ordering — no explicit flag needed.

    Parameters
    ----------
    lbm_node_name : str
        Name of the IBLBMFluidNode instance.
    rigid_body_node_name : str
        Name of the RigidBodyNode instance.
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].
    rho_physical : float
        Reference fluid density [kg/m^3]. Default: blood (1060).

    Returns
    -------
    list[EdgeSpec]
        Four edges: drag_force, drag_torque (forward, with transform),
        angular_velocity, orientation (back-edges, no transform).
    """
    return [
        # Forward: LBM drag → RigidBody (with unit conversion)
        EdgeSpec(
            source_node=lbm_node_name,
            target_node=rigid_body_node_name,
            source_field="drag_force",
            target_field="drag_force",
            transform=lbm_to_si_force(dx_physical, dt_physical, rho_physical),
            additive=True,
            source_units="lattice",
            target_units="N",
        ),
        EdgeSpec(
            source_node=lbm_node_name,
            target_node=rigid_body_node_name,
            source_field="drag_torque",
            target_field="drag_torque",
            transform=lbm_to_si_torque(dx_physical, dt_physical, rho_physical),
            additive=True,
            source_units="lattice",
            target_units="N*m",
        ),
        # Back-edges: RigidBody state → LBM boundary inputs
        # (auto-detected as back-edges during GraphManager.compile())
        # Angular velocity: RigidBody outputs rad/s, LBM expects rad/step.
        # Convert: omega_lattice = omega_SI * dt_physical
        EdgeSpec(
            source_node=rigid_body_node_name,
            target_node=lbm_node_name,
            source_field="angular_velocity",
            target_field="body_angular_velocity",
            transform=_make_si_to_lattice_omega(dt_physical, omega_max_lattice),
            source_units="rad/s",
            target_units="rad/step",
        ),
        EdgeSpec(
            source_node=rigid_body_node_name,
            target_node=lbm_node_name,
            source_field="orientation",
            target_field="body_orientation",
        ),
    ]
