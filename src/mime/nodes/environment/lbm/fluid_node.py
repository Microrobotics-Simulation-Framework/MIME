"""IBLBMFluidNode -- 3D LBM fluid solver as a MADDENING SimulationNode.

Wraps the existing LBM utility functions (d3q19.py, bounce_back.py,
helix_geometry.py) as a proper MimeNode for node-graph coupling via
GraphManager. Replaces the manual loop in run_confinement_sweep.py:run_single()
with a single update() call per timestep.

Two-pass bounce-back: pipe wall (static, simple BB) then the immersed body
(Bouzidi IBB or simple BB depending on use_bouzidi flag).

Geometry hook
-------------
The immersed body is **not** hard-wired. It is built each step by a
``LatticeBodyFactory`` (see :mod:`mime.nodes.environment.lbm.body`) from the
body's pose, which arrives on the shared fluid-node contract inputs
``body_position`` / ``body_velocity`` / ``body_angular_velocity`` /
``body_orientation`` (``src/mime/nodes/environment/FLUID_NODE_CONTRACT.md``).
This mirrors ``FVMFluidNode``'s ``dynamic_body_factories``, so the two solvers
are interchangeable for the same experiment. Two geometries ship:
``make_umr_body_factory`` (the de Boer rotating finned UMR — the original
hard-wired body) and ``make_sphere_body_factory``.

Passing the legacy ``body_geometry_params`` still selects the UMR and, with
``body_position`` left unwired, reproduces the pre-hook behaviour exactly
(the body sits at the grid centre and only spins about z).

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
from mime.nodes.environment.lbm.body import (
    LatticeBody,
    LatticeBodyFactory,
    LatticeGrid,
    make_sphere_body_factory,
    make_umr_body_factory,
    rigid_body_velocity_field,
)


@stability(StabilityLevel.EXPERIMENTAL)
class IBLBMFluidNode(MimeNode):
    """3D IB-LBM fluid solver with Bouzidi IBB for confined microrobot flows.

    Wraps existing LBM functions as a node-graph component. Receives the
    immersed body's pose and twist as boundary inputs (from a RigidBodyNode
    via edges) and outputs drag force/torque as boundary fluxes.

    The body geometry is supplied by a ``LatticeBodyFactory``; the node itself
    is geometry-agnostic and supports an arbitrary rigid body translating and
    rotating through the lattice.

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
    body_factory : LatticeBodyFactory, optional
        ``(pose, grid) -> LatticeBody``; rebuilt inside ``update()`` each step
        so the mask tracks the body's pose. See
        :mod:`mime.nodes.environment.lbm.body` for the shipped factories
        (``make_umr_body_factory``, ``make_sphere_body_factory``). Exactly one
        of ``body_factory`` / ``body_geometry_params`` must be given.
    body_geometry_params : dict, optional
        **Legacy UMR path.** Keyword arguments for ``create_umr_mask`` (may
        include nx/ny/nz, which are ignored in favour of the node's lattice)
        in lattice units. Equivalent to passing
        ``body_factory=make_umr_body_factory(**geom)``; kept so existing
        callers (``experiments/umr_confinement``) work unchanged.
    use_bouzidi : bool
        If True, use Bouzidi interpolated bounce-back for the body surface
        (requires the body to provide an SDF). Pipe wall always uses simple
        bounce-back.
    dx_physical : float
        Physical lattice spacing [m] for unit conversion.
    body_reference_position_lu : (3,) tuple, optional
        The pose the body sits at when ``body_position`` is not wired, and the
        pose used to pre-size the Bouzidi boundary-link buffer. Defaults to
        the grid centre — the pre-hook hard-wired behaviour.
    boundary_links_margin : float
        Safety factor on the at-init worst-case boundary-link count used to
        size ``jnp.nonzero``'s static buffer on the Bouzidi path. **A
        translating body changes its link count as it moves across the
        lattice** (staircasing is sub-cell-position dependent), and
        ``jnp.nonzero`` truncates SILENTLY. The default 1.5 covers the
        rotation of a fixed-centre UMR; raise it for a translating body, or
        call :meth:`count_boundary_links` to measure.
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-010",
        algorithm_version="1.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "3D IB-LBM fluid solver with Bouzidi IBB for an arbitrary rigid "
            "body translating/rotating in a confined flow"
        ),
        governing_equations=(
            "BGK-LBM D3Q19, Bouzidi interpolated bounce-back, "
            "momentum exchange force/torque"
        ),
        discretization="D3Q19 lattice Boltzmann with BGK collision operator",
        assumptions=(
            "Weakly-compressible flow (Ma << 1 at all lattice nodes); the "
            "incompressible limit is approached as Ma -> 0 with error O(Ma^2)",
            "Newtonian fluid",
            "Rigid body (no deformation)",
            "Single-device execution unless wrapped in ShardedStencilNode",
            "Body kinematics enter via the shared fluid-node contract inputs "
            "(body_position / body_velocity / body_angular_velocity); the "
            "geometry is supplied by a LatticeBodyFactory",
            "A geometry whose mask is parameterised about a single axis "
            "projects the angular velocity onto it (the UMR: z only). The "
            "sphere consumes the full angular-velocity vector.",
        ),
        limitations=(
            "No ambient-flow / inflow boundary condition: relative flow must "
            "be produced by translating the body through the lattice. There "
            "is no co-moving-frame or uniform-inflow option, so a body driven "
            "far along the vessel needs a correspondingly long lattice.",
            "Streaming is periodic (jnp.roll) on every axis; a body translated "
            "past the lattice edge is NOT wrapped by the geometry (masks use "
            "unwrapped coordinates) and will silently leave the domain.",
            "body_orientation is plumbed to the geometry hook but no shipped "
            "geometry consumes it (the UMR uses the node's integrated z-angle; "
            "the sphere is symmetric). A full-orientation body mask is not "
            "implemented.",
            "Re for a translating body is bounded by the Ma-Re-resolution "
            "coupling, not by the algorithm: D_lu >= sqrt(3)*Re*(tau-0.5)/"
            "(3*Ma_max), i.e. D_lu >= 0.29*Re at tau=0.55, Ma=0.1. Re~360 on "
            "a 1.7 mm sphere in a 6.3 mm tube therefore needs D_lu ~ 104 and "
            "an O(1e8)-cell lattice (~25 GB for f alone in float32) — an 80 GB "
            "accelerator or multi-GPU. Re~100 needs D_lu ~ 29 and is cheap "
            "(~4e6 cells). See the algorithm guide's cost table.",
            "Bouzidi IBB is not supported on the multi-GPU sharded path "
            "(needs per-slab SDF q-value recomputation)",
            "Per-step q-value recomputation (~0.1s at 192^3 on H100)",
            "First step triggers JAX compilation (30-60s at 192^3)",
        ),
        validated_regimes=(
            ValidatedRegime("Ma", 0.0, 0.1, "",
                            "Lattice Mach number (max over all nodes, incl. "
                            "fin tips) must be < 0.1"),
            ValidatedRegime("tau", 0.55, 1.5, "",
                            "BGK relaxation time stability range"),
            ValidatedRegime("Re_rot", 0.0, 30.0, "",
                            "Rotational Reynolds number; the UMR replication "
                            "cases (umr_confinement)"),
            ValidatedRegime("Re_translation", 0.0, 1.0, "",
                            "Translating-body Reynolds number. Drag on a "
                            "translating sphere is verified against Stokes "
                            "drag with the Haberman-Sayre wall correction "
                            "(MIME-VER-012) in the Stokes regime ONLY. "
                            "Higher Re is admissible and unvalidated — see "
                            "the resolution constraint in `limitations`."),
        ),
        references=(
            Reference("Bouzidi2001",
                       "Bouzidi et al. (2001) Phys. Fluids 13(11)"),
            Reference("Ladd1994",
                       "Ladd (1994) J. Fluid Mech. 271, 285-309"),
            Reference("HabermanSayre1958",
                       "Haberman & Sayre (1958) David Taylor Model Basin "
                       "Report 1143 — sphere-in-tube drag correction"),
            Reference("SchillerNaumann1933",
                       "Schiller & Naumann (1933) Z. Ver. Dtsch. Ing. 77 — "
                       "Cd = (24/Re)(1 + 0.15 Re^0.687), the finite-Re "
                       "reference the node is NOT yet validated against"),
        ),
        hazard_hints=(
            "If max_boundary_links_per_dir is too small, jnp.nonzero "
            "silently truncates boundary links — accuracy degrades "
            "without error. It is pre-sized at the body's reference pose "
            "times boundary_links_margin. A TRANSLATING body's link count "
            "varies with its sub-cell position, so the reference-pose count "
            "is not an upper bound: use count_boundary_links() to measure "
            "over the intended trajectory and raise boundary_links_margin.",
            "Re, Ma and resolution are not independent in weakly-compressible "
            "LBM. Raising Re at fixed grid raises Ma (compressibility error "
            "~Ma^2) or lowers tau toward the 0.5 stability singularity. "
            "Resolve Re by refining the grid, not by pushing Ma.",
        ),
        implementation_map={
            "BGK collision + streaming": (
                "mime.nodes.environment.lbm.d3q19.lbm_step_split"
            ),
            "Body geometry hook (mask + SDF from pose)": (
                "mime.nodes.environment.lbm.body"
            ),
            "Rigid-body wall velocity v + omega x r": (
                "mime.nodes.environment.lbm.body.rigid_body_velocity_field"
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
                anatomy="iliac artery (confined rotating UMR)",
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=0.1,
                viscosity_min_pa_s=3e-3, viscosity_max_pa_s=4e-3,
                temperature_min_c=36.0, temperature_max_c=38.0,
                notes=(
                    "The rotating-UMR replication regime (umr_confinement): "
                    "translational Re ~ 0, rotational Re up to ~30."
                ),
            ),
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="large vessel / in-vitro tube (translating millibot)",
                # STAGNANT, not a driven flow: the node has no inflow/ambient-
                # flow BC, so the relative velocity comes from the body moving
                # through otherwise-quiescent fluid (see `limitations`).
                flow_regime=FlowRegime.STAGNANT,
                re_min=0.0, re_max=360.0,
                viscosity_min_pa_s=0.7e-3, viscosity_max_pa_s=4e-3,
                temperature_min_c=20.0, temperature_max_c=38.0,
                notes=(
                    "Particle-Reynolds regime of a force-driven body "
                    "translating in a tube (e.g. a ~1.7 mm sphere in 6.3 mm "
                    "ID water). AFFORDABILITY, NOT VALIDATION: the solver "
                    "admits this range only at the resolution the Ma-Re "
                    "coupling demands (D_lu >= 0.29*Re at tau=0.55, Ma=0.1) — "
                    "Re~360 is an O(1e8)-cell run. Verified drag is Stokes-"
                    "regime only (MIME-VER-012); the finite-Re wake "
                    "(Schiller-Naumann, shedding above Re~270) is NOT yet "
                    "verified in this node."
                ),
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
        body_geometry_params: dict | None = None,
        use_bouzidi: bool = False,
        dx_physical: float = 1.0,
        multigpu_shard_axis: int | None = None,
        body_factory: LatticeBodyFactory | None = None,
        body_reference_position_lu: tuple[float, float, float] | None = None,
        boundary_links_margin: float = 1.5,
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

        # ---- Geometry hook ------------------------------------------------
        # Exactly one of body_factory / body_geometry_params selects the body.
        # body_geometry_params is the legacy UMR path, kept working verbatim.
        if (body_factory is None) == (body_geometry_params is None):
            raise ValueError(
                "IBLBMFluidNode needs exactly one of body_factory (the "
                "geometry hook — see mime.nodes.environment.lbm.body) or "
                "body_geometry_params (the legacy UMR shorthand); "
                f"got body_factory={body_factory!r}, "
                f"body_geometry_params={body_geometry_params!r}."
            )
        if body_factory is None:
            # nx/ny/nz in body_geometry_params are redundant with the node's
            # lattice (and wrong on a sharded slab) — the grid supplies them.
            geom = {
                k: v for k, v in body_geometry_params.items()
                if k not in ("nx", "ny", "nz")
            }
            body_factory = make_umr_body_factory(**geom)
        self._body_factory = body_factory

        cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
        # The pose the body holds when body_position is not wired. Defaulting
        # to the grid centre reproduces the pre-hook frozen-centre behaviour.
        if body_reference_position_lu is None:
            body_reference_position_lu = (cx, cy, cz)
        self._reference_position = jnp.asarray(
            body_reference_position_lu, dtype=jnp.float32)
        self._center = tuple(float(v) for v in body_reference_position_lu)
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

        # Precompute max_boundary_links_per_dir for the Bouzidi path.
        # Must be computed at construction time (not inside JIT) because
        # it requires int() on a concrete JAX array.
        self._boundary_links_margin = float(boundary_links_margin)
        if use_bouzidi:
            # Fail fast: Bouzidi needs an SDF to locate the wall along each
            # missing link. Check at construction, not on the first update().
            self._check_sdf(self._body_at())
            max_count = self.count_boundary_links()
            # Margin covers pose-to-pose variation of the link count (the
            # rotation of a fixed-centre UMR at 1.5x; a translating body's
            # sub-cell position also modulates it — see the hazard hint).
            self._max_boundary_links_per_dir = (
                int(max_count * self._boundary_links_margin) + 1
            )
        else:
            self._max_boundary_links_per_dir = 0

    @staticmethod
    def _check_sdf(body: LatticeBody) -> LatticeBody:
        if body.sdf is None:
            raise ValueError(
                f"use_bouzidi=True but body {body.name!r} provides no SDF; "
                "Bouzidi interpolated bounce-back needs one to locate the "
                "wall along each missing link. Supply LatticeBody.sdf or use "
                "simple bounce-back (use_bouzidi=False)."
            )
        return body

    def _body_at(
        self,
        position_lu: tuple[float, float, float] | None = None,
        angle: float = 0.0,
    ) -> LatticeBody:
        """Build the body at a concrete pose (construction-time helper).

        Not for use inside ``update()`` — that path goes through
        :meth:`_build_body` with the traced boundary inputs.
        """
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        pos = (self._reference_position if position_lu is None
               else jnp.asarray(position_lu, dtype=jnp.float32))
        pose = {
            "position": pos,
            "velocity": jnp.zeros(3, dtype=jnp.float32),
            "angular_velocity": jnp.zeros(3, dtype=jnp.float32),
            "orientation": jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            "angle": jnp.asarray(angle, dtype=jnp.float32),
        }
        return self._body_factory(pose, LatticeGrid((nx, ny, nz)))

    def count_boundary_links(
        self,
        position_lu: tuple[float, float, float] | None = None,
        angle: float = 0.0,
    ) -> int:
        """Worst-case D3Q19 boundary-link count for the body at one pose.

        Returns ``max_q sum_x missing_mask[q, x]`` — the quantity
        ``max_boundary_links_per_dir`` must exceed, or
        :func:`compute_q_values_sdf_sparse`'s ``jnp.nonzero`` **silently
        truncates** links on the Bouzidi path and drag degrades with no error.

        The constructor calls this at the body's reference pose. For a body
        that translates, sweep the intended trajectory::

            worst = max(node.count_boundary_links(position_lu=p)
                        for p in trajectory)

        and size ``boundary_links_margin`` so that
        ``max_count_at_reference * margin > worst``.

        Concrete (not traced) — for pre-flight sizing, never inside ``update``.
        """
        missing = compute_missing_mask(self._body_at(position_lu, angle).mask)
        return int(jnp.max(jnp.sum(missing, axis=(1, 2, 3))))

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
        """The shared fluid-node contract's ``body_*`` inputs, in lattice units.

        ``body_position`` defaults to the body's reference position (the grid
        centre unless overridden) rather than the origin, so a graph that does
        not wire it leaves the body exactly where the pre-geometry-hook node
        froze it. Wire it via ``make_iblbm_rigid_body_edges(...,
        wire_translation=True)``, which carries the SI->lattice transform.
        """
        return {
            "body_position": BoundaryInputSpec(
                shape=(3,),
                default=self._reference_position,
                description=(
                    "Body centroid position [lattice cells]. Defaults to the "
                    "body reference position (grid centre), NOT the origin — "
                    "an unwired graph holds the body at its reference pose."
                ),
                expected_units="lattice",
            ),
            "body_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body linear velocity [cells/step] in lattice units",
                expected_units="lattice",
            ),
            "body_angular_velocity": BoundaryInputSpec(
                shape=(3,),
                default=jnp.zeros(3),
                description="Body angular velocity [rad/step] in lattice units",
                expected_units="lattice",
            ),
            "body_orientation": BoundaryInputSpec(
                shape=(4,),
                default=jnp.array([1.0, 0.0, 0.0, 0.0]),
                description=(
                    "Body orientation quaternion (w,x,y,z), dimensionless. "
                    "Passed to the geometry hook; no shipped geometry reads "
                    "it (the UMR mask uses the node's integrated z-angle, the "
                    "sphere is symmetric)."
                ),
                expected_units="lattice",
            ),
        }

    def _angular_velocity(self, boundary_inputs: dict) -> jnp.ndarray:
        """``body_angular_velocity`` [rad/step], float32-cast (see :meth:`_pose`)."""
        v = boundary_inputs.get("body_angular_velocity")
        if v is None:
            return jnp.zeros(3, dtype=jnp.float32)
        return jnp.asarray(v).astype(jnp.float32)

    def _pose(self, boundary_inputs: dict, angle: jnp.ndarray) -> dict:
        """Assemble the traced pose dict handed to the geometry factory.

        Every value is cast to float32: the LBM solver is float32 by design
        (see ``init_equilibrium``'s explicit dtype), and a graph running under
        ``jax_enable_x64`` must not silently promote the wall velocity — and
        therefore the streamed ``f`` — to float64.

        Defaults mirror :meth:`boundary_input_spec` exactly, so an unwired
        input behaves the same whether or not the GraphManager materialises
        the declared default.
        """
        def _get(key, default):
            v = boundary_inputs.get(key)
            return (default if v is None
                    else jnp.asarray(v).astype(jnp.float32))

        return {
            "position": _get("body_position", self._reference_position),
            "velocity": _get("body_velocity", jnp.zeros(3, jnp.float32)),
            "angular_velocity": self._angular_velocity(boundary_inputs),
            "orientation": _get(
                "body_orientation",
                jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)),
            "angle": angle,
        }

    def _build_body(
        self, boundary_inputs: dict, angle: jnp.ndarray, grid: LatticeGrid,
    ) -> LatticeBody:
        """Rebuild the immersed body from the current pose (traceable)."""
        body = self._body_factory(self._pose(boundary_inputs, angle), grid)
        if self.params["use_bouzidi"]:
            # Normally already caught at construction; re-checked here for a
            # factory that returns an SDF only for some poses.
            self._check_sdf(body)
        return body

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        tau = self.params["tau"]
        use_bouzidi = self.params["use_bouzidi"]
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]

        omega_z = self._angular_velocity(boundary_inputs)[2]

        # 1. Update the integrated z-angle (dt_lbm = 1 in lattice units).
        #    Kept as node state (rather than derived from body_orientation)
        #    because the UMR mask is parameterised by this angle; a geometry
        #    that ignores it (the sphere) is unaffected.
        new_angle = state["body_angle"] + omega_z

        # 2. Rebuild the body at the current pose via the geometry hook.
        body = self._build_body(
            boundary_inputs, new_angle, LatticeGrid((nx, ny, nz)),
        )
        body_missing = compute_missing_mask(body.mask)
        solid_mask = self._pipe_wall | body.mask

        # 3. Wall velocity: v + omega x (x - ref_point)
        wall_vel = rigid_body_velocity_field(
            (nx, ny, nz), body.ref_point, body.velocity, body.angular_velocity,
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

        # Pass 2: immersed body (moving)
        if use_bouzidi:
            q_values = compute_q_values_sdf_sparse(
                body_missing, body.sdf,
                max_boundary_links_per_dir=self._max_boundary_links_per_dir,
            )
            f = apply_bouzidi_bounce_back(
                f, f_pre, body_missing, solid_mask,
                q_values, wall_velocity=wall_vel,
            )
        else:
            f = apply_bounce_back(
                f, f_pre, body_missing, solid_mask,
                wall_velocity=wall_vel,
            )

        # 6. Momentum exchange force/torque about the body's reference point
        force = compute_momentum_exchange_force(f_pre, f, body_missing)
        torque = compute_momentum_exchange_torque(
            f_pre, f, body_missing, body.ref_point,
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
        post-collision populations); recompute the pipe + body missing-link
        masks per slab from the halo-exchanged ``pipe_wall`` and the body
        rebuilt by the geometry hook on the slab's *global* coordinate range
        (the ``LatticeGrid`` origin from ``shard_info``); two-pass bounce-back
        on the interior; and partial-sum ``drag_force`` / ``drag_torque`` that
        the wrapper ``lax.psum``s across the mesh. See
        :meth:`_update_padded_sharded`.

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
        halo-exchanged ``static_padded["pipe_wall"]`` slab; the body and its
        rigid-body velocity field are rebuilt on the slab's global coordinate
        range (the ``LatticeGrid`` origin carries the ``shard_info`` offset);
        ``drag_force`` / ``drag_torque`` are per-slab partial sums that
        ``ShardedStencilNode`` ``lax.psum``s across the device mesh.

        The geometry hook makes this geometry-agnostic for free: any
        ``LatticeBodyFactory`` that honours ``LatticeGrid.origin`` shards.
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
        nx, ny, nz = self.params["nx"], self.params["ny"], self.params["nz"]

        # Slab geometry along the shard axis: global offset of the first
        # interior cell (a traced scalar) and the unpadded local extent.
        offset, extent = shard_info[ax]

        f_pad = state_padded["f"]  # halo-padded on all 3 spatial axes

        omega_z = self._angular_velocity(boundary_inputs)[2]
        new_angle = state_padded["body_angle"] + omega_z

        # 1. Collision (local) on the full padded array, then halo-aware
        #    streaming (reads neighbour populations from the exchanged halos).
        density_pad, velocity_pad = compute_macroscopic(f_pad)
        f_post_collision_pad = collide_bgk(f_pad, density_pad, velocity_pad, tau)
        f_post_stream = stream_padded(f_post_collision_pad, halo=halo)  # interior
        interior = (slice(halo, -halo),) * 3 + (slice(None),)
        f_pre = f_post_collision_pad[interior]  # interior post-collision

        # 2. Per-slab masks. pipe_wall arrives halo-padded (edge boundary) on
        #    the shard axis; the body is rebuilt on the slab's global
        #    coordinates (its first padded cell is at global offset-halo) via
        #    the geometry hook, with the padded-slab extent as the grid.
        pipe_wall_pad = static_padded["pipe_wall"].astype(jnp.bool_)
        pad_shape = [nx, ny, nz]
        pad_shape[ax] = extent + 2 * halo
        body_origin = [0.0, 0.0, 0.0]
        body_origin[ax] = offset - halo
        body = self._build_body(
            boundary_inputs, new_angle,
            LatticeGrid(tuple(pad_shape), tuple(body_origin)),
        )
        body_pad = body.mask

        pipe_missing = compute_missing_mask_sharded(pipe_wall_pad, ax, halo)
        body_missing = compute_missing_mask_sharded(body_pad, ax, halo)

        # Combined interior solid (vestigial arg to apply_bounce_back).
        pipe_int = jax.lax.slice_in_dim(pipe_wall_pad, halo, halo + extent, axis=ax)
        body_int = jax.lax.slice_in_dim(body_pad, halo, halo + extent, axis=ax)
        solid_int = pipe_int | body_int

        # 3. Wall velocity v + omega x r at interior cells, in global coords.
        interior_shape = [nx, ny, nz]
        interior_shape[ax] = extent
        wall_origin = [0.0, 0.0, 0.0]
        wall_origin[ax] = offset
        wall_vel = rigid_body_velocity_field(
            tuple(interior_shape), body.ref_point,
            body.velocity, body.angular_velocity,
            origin=tuple(wall_origin),
        )

        # 4. Two-pass bounce-back on the interior (pipe static, body moving).
        f_bb = apply_bounce_back(
            f_post_stream, f_pre, pipe_missing, solid_int, wall_velocity=None,
        )
        f_bb = apply_bounce_back(
            f_bb, f_pre, body_missing, solid_int, wall_velocity=wall_vel,
        )

        # 5. Momentum exchange — per-slab partials; wrapper psums them.
        force = compute_momentum_exchange_force(f_pre, f_bb, body_missing)
        torque = compute_momentum_exchange_torque(
            f_pre, f_bb, body_missing, body.ref_point, origin=tuple(wall_origin),
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


def _make_si_to_lattice_position(
    dx_physical: float, origin_lu: tuple[float, float, float],
):
    """Create a transform converting a position from metres to lattice cells.

    ``x_lu = x_SI / dx + origin_lu`` — ``origin_lu`` is the lattice coordinate
    the body's SI origin maps to (usually the grid centre, so a body at SI
    ``(0,0,0)`` sits mid-lattice).
    """
    inv_dx = 1.0 / float(dx_physical)
    origin = jnp.asarray(origin_lu, dtype=jnp.float32)

    def _convert(pos_si):
        return jnp.asarray(pos_si) * inv_dx + origin

    _convert.__qualname__ = (
        f"si_to_lattice_position(dx={dx_physical}, origin_lu={origin_lu})"
    )
    return _convert


def _make_si_to_lattice_velocity(
    dx_physical: float, dt_physical: float,
    u_max_lattice: float | None = 0.0577,
):
    """Create a transform converting a velocity from m/s to cells/step.

    ``u_lu = u_SI * dt / dx``, with the same step-0 safety clamp rationale as
    :func:`_make_si_to_lattice_omega`: at step 0 of FSI coupling the back-edge
    carries zero drag, so the body can produce an unphysically large velocity
    that would drive the LBM past its Mach limit.

    ``u_max_lattice`` defaults to 0.0577 = 0.1/sqrt(3), i.e. the Ma = 0.1
    bound. Pass ``None`` to disable the clamp (and lose that protection).
    """
    factor = float(dt_physical) / float(dx_physical)
    clamp = None if u_max_lattice is None else float(u_max_lattice)

    def _convert(u_si):
        u_lat = jnp.asarray(u_si) * factor
        if clamp is None:
            return u_lat
        return jnp.clip(u_lat, -clamp, clamp)

    _convert.__qualname__ = (
        f"si_to_lattice_velocity(dx={dx_physical}, dt={dt_physical}, "
        f"clamp={u_max_lattice})"
    )
    return _convert


def make_iblbm_rigid_body_edges(
    lbm_node_name: str,
    rigid_body_node_name: str,
    dx_physical: float,
    dt_physical: float,
    rho_physical: float = 1060.0,
    omega_max_lattice: float = 0.005,
    wire_translation: bool = False,
    position_origin_lu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    u_max_lattice: float | None = 0.0577,
) -> list[EdgeSpec]:
    """Return EdgeSpecs for wiring IBLBMFluidNode to RigidBodyNode.

    Forward edges carry drag force/torque from LBM to rigid body with
    LBM-to-SI unit transforms. Back-edges carry the body kinematics from the
    rigid body back to the LBM's contract ``body_*`` inputs, with SI->lattice
    transforms (orientation is dimensionless and needs none).

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
    omega_max_lattice : float
        Angular-velocity clamp [rad/step]; see
        :func:`_make_si_to_lattice_omega`.
    wire_translation : bool
        **Opt-in.** When True, also wire ``position`` -> ``body_position`` and
        ``velocity`` -> ``body_velocity``, so the LBM body translates with the
        rigid body. Default False: the LBM body stays pinned at its reference
        position and only spins — the behaviour every pre-existing caller
        (``experiments/umr_confinement``) was built and validated against.
        Turning this on changes those results, so it is not the default.
    position_origin_lu : (3,) tuple
        Lattice coordinate that SI position ``(0,0,0)`` maps to. Only used
        when ``wire_translation=True``; should normally be the node's
        ``body_reference_position_lu`` (the grid centre by default), so a
        body at the SI origin starts mid-lattice.
    u_max_lattice : float or None
        Linear-velocity clamp [cells/step] on the translation back-edge;
        default 0.0577 (Ma = 0.1). ``None`` disables it.

    Returns
    -------
    list[EdgeSpec]
        Four edges (drag_force, drag_torque forward with transforms;
        angular_velocity, orientation back), plus body_position and
        body_velocity back-edges when ``wire_translation=True``.
    """
    edges = [
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
    if wire_translation:
        edges.extend([
            EdgeSpec(
                source_node=rigid_body_node_name,
                target_node=lbm_node_name,
                source_field="position",
                target_field="body_position",
                transform=_make_si_to_lattice_position(
                    dx_physical, position_origin_lu),
                source_units="m",
                target_units="lattice",
            ),
            EdgeSpec(
                source_node=rigid_body_node_name,
                target_node=lbm_node_name,
                source_field="velocity",
                target_field="body_velocity",
                transform=_make_si_to_lattice_velocity(
                    dx_physical, dt_physical, u_max_lattice),
                source_units="m/s",
                target_units="lattice",
            ),
        ])
    return edges
