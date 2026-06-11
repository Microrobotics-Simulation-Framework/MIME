"""FVMFluidNode — graph-native FVM fluid solver as a MADDENING/MIME node.

This is the M3 deliverable: the FVM stack (mesh + operators + PISO +
IBM) wrapped as a :class:`MimeNode` so it can be composed in a
GraphManager with rigid-body, magnetic-actuation, and other MIME
nodes.

Interface decisions
-------------------
1. **State is an explicit JAX pytree**, not a Python object: ``u``, ``p``,
   ``F``, ``t``, ``u_pre_ibm``. Every field is a JAX array with static
   shape. This is what makes ``jax.lax.scan`` / ``jit`` / ``vmap``
   transparent through the node.
2. **Robot pose enters as a boundary input** (not as a parameter), so a
   GraphManager can drive the fluid from a coupled rigid-body node via
   edges. The robot's IBM body is rebuilt inside ``update()`` from the
   pose so SDF gradients with respect to pose are differentiable.
3. **Drag force / torque are boundary fluxes**, computed each step via
   the Brinkman-aware IBM force formula (see :mod:`.ibm`) on the
   *pre-Brinkman* velocity field — the post-Brinkman field has had its
   IBM region zeroed and would give a misleading-low force.
4. **Static bodies (pipe wall) are constructed once at node init**;
   only the dynamic robot body is rebuilt per step. This keeps the
   per-step cost from growing with the number of static obstacles.
5. **Mesh + cfg are static-shape pytrees**; the solver step itself does
   not reference Python-level state. Vmap over parameter sets (``ν``,
   ``ρ``, IBM penalty, etc.) is therefore possible without retracing.
6. **Body kinematics** (position, orientation, linear/angular velocity)
   come in via ``boundary_inputs`` exactly the way the existing
   :class:`IBLBMFluidNode` exposes them; this keeps coupling-edge code
   reusable across the LBM, BEM, and FVM solvers.

The interface here is intentionally a *generalisation* of the existing
LBM fluid-node interface — once stable it can be used to refactor LBM
and BEM to a unified contract. Differences from
:class:`IBLBMFluidNode` (recorded for the unification work):

* No `body_orientation` quaternion in inputs (we accept rotation matrix
  via `body_quaternion` if needed; default identity). The capsule SDF
  in this version uses two endpoint positions which already encode
  orientation implicitly. Adding quaternion inputs is a one-line
  addition once a consumer needs it.
* Output flux units are SI ``N`` and ``N·m`` directly (no LBM →SI
  conversion edge needed); this assumes the user has set ``rho`` and
  ``nu`` in physical units.

References
----------
- Issa (1986) "Solution of the implicitly discretised fluid flow
  equations by operator splitting", J. Comput. Phys. 62.
- Peskin (2002) "The immersed boundary method", Acta Numerica 11.
- Du et al. (2024) "DiFVM: A differentiable finite volume method",
  arXiv:2603.15920.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, BoundaryFluxSpec
from maddening.core.compliance.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.compliance.stability import stability

from mime.core.node import MimeNode
from mime.core.metadata import (
    MimeNodeMeta, NodeRole,
    AnatomicalRegimeMeta, AnatomicalCompartment, FlowRegime,
)

from mime.nodes.environment.fvm.mesh import FVMMesh
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import (
    PisoConfig, make_piso_step, initial_state as piso_initial_state,
)
from mime.nodes.environment.fvm.ibm import (
    IBMBody, compute_ibm_forces, surface_integral_force,
    momentum_deficit_drag,
)
from mime.nodes.environment.fvm.lifting import LiftingFunction
from mime.nodes.environment.fvm.integrator import (
    sample_velocity_at_points, spread_point_forces,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf, rigid_body_velocity


# Type alias: a function pose -> IBMBody, for building dynamic bodies
# from boundary inputs each step.
BodyFactory = Callable[[Dict[str, jnp.ndarray]], IBMBody]


def make_sphere_body_factory(
    name: str, radius: float, *, extract_force: bool = True,
) -> BodyFactory:
    """Build an :class:`IBMBody` factory for a moving sphere.

    The returned callable closes over ``radius`` and produces, given
    boundary inputs containing ``"position"`` and ``"linear_velocity"``
    (and optionally ``"angular_velocity"``), an IBMBody whose SDF and
    rigid-body velocity field track the input pose. Differentiable in
    pose.
    """
    def factory(inputs: dict) -> IBMBody:
        pos = inputs["position"]
        v_lin = inputs.get("linear_velocity", jnp.zeros(3))
        omega = inputs.get("angular_velocity", None)

        def sdf(x):
            return sphere_sdf(x, center=pos, radius=radius)

        def u_body(x):
            return rigid_body_velocity(
                x, pose_x=pos,
                linear_velocity=v_lin,
                angular_velocity=omega,
            )
        return IBMBody(
            name=name,
            sdf=sdf,
            u_body=u_body,
            extract_force=extract_force,
            ref_point=pos,
        )
    return factory


@stability(StabilityLevel.EXPERIMENTAL)
class FVMFluidNode(MimeNode):
    """Graph-native FVM fluid solver as a MADDENING simulation node.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Time step in physical units (s). The solver integrates one PISO
        time step per ``update()`` call.
    mesh : FVMMesh
        Pre-built FVM mesh (Cartesian-structured for now). Static for
        the lifetime of the node.
    bcs : dict[str, VelocityBC]
        Boundary-condition map keyed by patch name.
    cfg : PisoConfig
        PISO solver configuration (ν, ρ, IBM penalty, BC types, ...).
    static_bodies : list[IBMBody]
        Bodies that don't move during the simulation (pipe wall etc.).
    dynamic_body_factories : list[(name, BodyFactory)]
        Factories for bodies whose pose comes from ``boundary_inputs``
        each step. Each factory's ``IBMBody`` will be re-constructed
        inside ``update()`` so SDF gradients w.r.t. pose are available.
    body_force_fn : optional ``Callable[[jnp.ndarray], jnp.ndarray]``
        Time-dependent body force (m/s²). Returns a ``[dim]`` vector
        (uniform) or a ``[N_cells, dim]`` array (spatially varying).
    """

    meta = NodeMeta(
        algorithm_id="MIME-NODE-020",
        algorithm_version="0.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Graph-native FVM solver for incompressible Navier-Stokes "
            "with diffuse-penalty IBM."
        ),
        governing_equations=(
            "Incompressible Navier-Stokes; PISO/projection time stepping; "
            "Goldstein-style diffuse-penalty IBM."
        ),
        discretization=(
            "Cell-centred finite volume on a Cartesian face graph "
            "(gather→compute→scatter); FFT/DST/DCT-diagonalised pressure "
            "Poisson and Helmholtz; Rhie-Chow face flux."
        ),
        assumptions=(
            "Incompressible Newtonian fluid",
            "No-slip walls (Dirichlet) or periodic BCs only",
            "Single-device execution (no halo exchange)",
            "IBM penalty large enough to enforce no-slip on body "
            "(α·dt ≫ 1 recommended)",
        ),
        limitations=(
            "Cartesian mesh only (unstructured-mesh extension is "
            "scoped but not implemented)",
            "Body-force-driven flow only (inflow/outflow BCs are "
            "implemented in operators but not wired through the node)",
            "Single density / viscosity per node",
        ),
        validated_regimes=(
            ValidatedRegime("Re_pipe", 0.0, 500.0, "",
                            "Pipe Reynolds; tested up to ~200 in M1/M2"),
            ValidatedRegime("Wo", 0.0, 10.0, "",
                            "Womersley number; tested at 7 in M1"),
        ),
        references=(
            Reference("Issa1986",
                      "Issa (1986) J. Comput. Phys. 62, 40-65."),
            Reference("Peskin2002",
                      "Peskin (2002) Acta Numerica 11."),
            Reference("DiFVM",
                      "Du et al. (2024) arXiv:2603.15920."),
            Reference("Womersley1955",
                      "Womersley (1955) J. Physiol. 127:553-563."),
        ),
        hazard_hints=(
            "IBM diffuse zone (~1.5 cells) shrinks the effective "
            "obstacle radius — drag force is biased low at coarse "
            "resolution. Mesh refinement is the only fix.",
            "Body-force-driven periodic flow needs a few diffusion "
            "timescales (R²/ν) to reach periodic steady state. Initial "
            "transient dynamics are not the steady response.",
        ),
        implementation_map={
            "Mesh + face graph": "mime.nodes.environment.fvm.mesh",
            "Gather/compute/scatter operators": "mime.nodes.environment.fvm.operators",
            "FFT/DST pressure + Helmholtz": "mime.nodes.environment.fvm.pressure",
            "PISO time stepping": "mime.nodes.environment.fvm.piso",
            "Diffuse penalty IBM": "mime.nodes.environment.fvm.ibm",
            "SDFs (sphere/cylinder/capsule)": "mime.nodes.environment.fvm.sdf",
        },
    )

    mime_meta = MimeNodeMeta(
        role=NodeRole.ENVIRONMENT,
        anatomical_regimes=(
            AnatomicalRegimeMeta(
                compartment=AnatomicalCompartment.BLOOD,
                anatomy="iliac artery (millibot)",
                flow_regime=FlowRegime.OSCILLATORY,
                re_min=0.0, re_max=500.0,
                viscosity_min_pa_s=3e-3, viscosity_max_pa_s=4e-3,
                temperature_min_c=36.0, temperature_max_c=38.0,
            ),
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        *,
        mesh: FVMMesh,
        bcs: Dict[str, VelocityBC],
        cfg: PisoConfig,
        static_bodies: List[IBMBody] | None = None,
        dynamic_body_factories: List[Tuple[str, BodyFactory]] | None = None,
        body_force_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        lifting: LiftingFunction | None = None,
        force_method: str = "brinkman",
        force_shell: tuple[float, float] = (1.5, 3.5),
        n_sample_points: int | None = None,
        n_forcing_points: int | None = None,
        forcing_sigma: float | None = None,
        **kwargs,
    ):
        """``force_method``: ``"brinkman"`` (legacy per-cell penalty
        sink) or ``"surface_integral"`` (preferred — Cauchy stress
        integrated on a shell of cells just outside the body).
        ``force_shell`` selects the shell location in units of dx;
        only relevant when ``force_method="surface_integral"``.

        ``lifting``: optional :class:`LiftingFunction` providing the
        Dirichlet inlet/outlet velocity decomposition. When provided,
        the PISO loop evolves ``u_hom`` and ``state["u"]`` stores
        ``u_hom``. ``state["u_pre_ibm"]`` and ``state["u_after_explicit"]``
        are reconstructed in the *physical* frame for downstream force
        extractors. Inlet ``VelocityBC`` should be passed with
        ``u_wall = 0``."""
        super().__init__(name, timestep, **kwargs)
        self._mesh = mesh
        self._bcs = bcs
        self._cfg = cfg
        self._static_bodies = list(static_bodies or ())
        self._dynamic_factories = list(dynamic_body_factories or ())
        # Single-body shared-contract alias: when exactly one dynamic body is
        # present the node also exposes the contract names drag_force /
        # drag_torque and body_* inputs (see environment/FLUID_NODE_CONTRACT.md).
        self._single_body = (
            self._dynamic_factories[0][0]
            if len(self._dynamic_factories) == 1 else None
        )
        self._body_force_fn = body_force_fn
        self._lifting = lifting
        if force_method not in ("brinkman", "surface_integral", "momentum_deficit"):
            raise ValueError(f"force_method={force_method!r} not supported")
        self._force_method = force_method
        self._force_shell = force_shell
        # Optional far-field velocity-sampling port (two-scale Schwarz
        # coupling): when set, the node accepts a ``sample_points``
        # [n, dim] input and emits ``velocity_at_points`` [n, dim] sampled
        # from the solved (physical) velocity field. See A1 / make_two_scale.
        self._n_sample_points = n_sample_points
        # Optional near→far reaction-forcing port: when set, the node accepts
        # ``forcing_points`` [n, dim] (m) and ``forcing_values`` [n, dim] (N)
        # and spreads them as a regularized body force (A2). ``forcing_sigma``
        # is the Gaussian spread width (m); defaults to ~1.5 mean cell sizes.
        self._n_forcing_points = n_forcing_points
        if forcing_sigma is None:
            mean_cell_len = float(jnp.mean(self._mesh.V)) ** (1.0 / self._mesh.dim)
            forcing_sigma = 1.5 * mean_cell_len
        self._forcing_sigma = float(forcing_sigma)
        # v0.2 #3: declare the mesh on the static_data channel so the
        # GraphManager tracks it for JIT-cache invalidation (a geometry
        # change now triggers a recompile) and it is never folded into
        # the state pytree.  Built once here; see _build_static_data.
        self._static_data = self._build_static_data()

    # ---- MimeNode contract ------------------------------------------

    def halo_width(self) -> dict[int, int]:
        """FVM cell-centred + face stencil needs one-cell connectivity.

        FVM stores state as ``(N_cells, ...)`` over an unstructured mesh,
        not as a structured ``(nx, ny, nz, ...)`` array, so the per-axis
        halo concept maps only loosely.  Returning a non-empty
        ``{0: 1}`` marks the node as non-pointwise so MADDENING's
        pointwise sharder refuses to shard it; a real pencil-mesh
        sharded FVM needs graph-partitioning halos which v0.2 does not
        yet provide.
        """
        return {0: 1}

    def _build_static_data(self) -> dict:
        """Unfold the :class:`FVMMesh` pytree into flat static_data keys.

        ``StaticArray`` rejects nested structures, so the mesh — a
        frozen pytree of arrays plus a ``patches`` tuple of
        ``BoundaryPatch`` — cannot be wrapped whole.  Each leaf array
        becomes its own top-level key (``mesh_*`` for the interior face
        graph and cell geometry, ``patch_<name>_*`` for each boundary
        patch); scalar / tuple metadata stays bare and is hashed by
        value.

        Every array uses ``replication="replicate"``: the FVM face
        graph mixes face-indexed (``owner``, ``Sf``, …) and cell-indexed
        (``V``, ``x``) arrays on different axes, so no single
        ``shard_axis`` is valid, and :meth:`halo_width` already blocks
        MADDENING's pointwise sharder from sharding the node.
        """
        from maddening.core.static_data import StaticArray
        m = self._mesh
        sd: dict = {
            "mesh_owner": StaticArray(m.owner),
            "mesh_neighbour": StaticArray(m.neighbour),
            "mesh_Sf": StaticArray(m.Sf),
            "mesh_n": StaticArray(m.n),
            "mesh_area": StaticArray(m.area),
            "mesh_d": StaticArray(m.d),
            "mesh_d_mag": StaticArray(m.d_mag),
            "mesh_w": StaticArray(m.w),
            "mesh_V": StaticArray(m.V),
            "mesh_x": StaticArray(m.x),
            # Bare scalar / tuple metadata — hashed by repr().
            "N_cells": m.N_cells,
            "N_faces": m.N_faces,
            "dim": m.dim,
            "cartesian_shape": m.cartesian_shape,
            "cartesian_spacing": m.cartesian_spacing,
            "cartesian_origin": m.cartesian_origin,
            "patch_names": tuple(p.name for p in m.patches),
        }
        # V_owner / V_neighbour are Optional on FVMMesh (the Cartesian
        # builders always populate them, but guard anyway).
        if m.V_owner is not None:
            sd["mesh_V_owner"] = StaticArray(m.V_owner)
        if m.V_neighbour is not None:
            sd["mesh_V_neighbour"] = StaticArray(m.V_neighbour)
        for p in m.patches:
            sd[f"patch_{p.name}_owner"] = StaticArray(p.owner)
            sd[f"patch_{p.name}_Sf"] = StaticArray(p.Sf)
            sd[f"patch_{p.name}_n"] = StaticArray(p.n)
            sd[f"patch_{p.name}_area"] = StaticArray(p.area)
            sd[f"patch_{p.name}_d"] = StaticArray(p.d)
            sd[f"patch_{p.name}_face_x"] = StaticArray(p.face_x)
        return sd

    @property
    def static_data(self) -> dict:
        """The FVM mesh, exposed via the v0.2 ``static_data`` channel.

        Built once in ``__init__`` (see :meth:`_build_static_data`).
        ``static_data`` is not checkpointed — a checkpoint/restore
        round-trip reconstructs the node with the same ``mesh``
        argument, which rebuilds this dict.
        """
        return self._static_data

    def initial_state(self) -> dict:
        s = piso_initial_state(self._mesh)
        # Drag outputs (per dynamic body that extracts force).
        for name, _ in self._dynamic_factories:
            s[f"force_{name}"] = jnp.zeros(self._mesh.dim, dtype=self._mesh.V.dtype)
            if self._mesh.dim == 3:
                s[f"torque_{name}"] = jnp.zeros(3, dtype=self._mesh.V.dtype)
            else:
                s[f"torque_{name}"] = jnp.zeros((), dtype=self._mesh.V.dtype)
        if self._single_body is not None:
            # Shared-contract aliases for the single immersed body.
            s["drag_force"] = s[f"force_{self._single_body}"]
            s["drag_torque"] = s[f"torque_{self._single_body}"]
        if self._n_sample_points is not None:
            s["velocity_at_points"] = jnp.zeros(
                (self._n_sample_points, self._mesh.dim), dtype=self._mesh.V.dtype)
        return s

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        spec = {}
        for name, _ in self._dynamic_factories:
            spec[f"{name}_position"] = BoundaryInputSpec(
                shape=(self._mesh.dim,),
                default=jnp.zeros(self._mesh.dim),
                description=f"Position of body {name!r} (m)",
                expected_units="m",
            )
            spec[f"{name}_linear_velocity"] = BoundaryInputSpec(
                shape=(self._mesh.dim,),
                default=jnp.zeros(self._mesh.dim),
                description=f"Linear velocity of body {name!r} (m/s)",
                expected_units="m/s",
            )
            if self._mesh.dim == 3:
                spec[f"{name}_angular_velocity"] = BoundaryInputSpec(
                    shape=(3,), default=jnp.zeros(3),
                    description=f"Angular velocity of body {name!r} (rad/s)",
                    expected_units="rad/s",
                )
        if self._single_body is not None:
            # Shared-contract input aliases for the single immersed body.
            spec["body_position"] = BoundaryInputSpec(
                shape=(self._mesh.dim,),
                default=jnp.zeros(self._mesh.dim),
                description="Body centroid position (m)",
                expected_units="m",
            )
            spec["body_velocity"] = BoundaryInputSpec(
                shape=(self._mesh.dim,),
                default=jnp.zeros(self._mesh.dim),
                description="Body linear velocity (m/s)",
                expected_units="m/s",
            )
            if self._mesh.dim == 3:
                spec["body_angular_velocity"] = BoundaryInputSpec(
                    shape=(3,), default=jnp.zeros(3),
                    description="Body angular velocity (rad/s)",
                    expected_units="rad/s",
                )
        if self._n_sample_points is not None:
            spec["sample_points"] = BoundaryInputSpec(
                shape=(self._n_sample_points, self._mesh.dim),
                default=jnp.zeros((self._n_sample_points, self._mesh.dim)),
                description="World points to sample the fluid velocity at (m)",
                expected_units="m",
            )
        if self._n_forcing_points is not None:
            spec["forcing_points"] = BoundaryInputSpec(
                shape=(self._n_forcing_points, self._mesh.dim),
                default=jnp.zeros((self._n_forcing_points, self._mesh.dim)),
                description="World points where reaction forces are applied (m)",
                expected_units="m",
            )
            spec["forcing_values"] = BoundaryInputSpec(
                shape=(self._n_forcing_points, self._mesh.dim),
                default=jnp.zeros((self._n_forcing_points, self._mesh.dim)),
                description="Reaction forces on the fluid at forcing_points (N)",
                expected_units="N",
            )
        return spec

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        spec = {}
        for name, _ in self._dynamic_factories:
            spec[f"force_{name}"] = BoundaryFluxSpec(
                shape=(self._mesh.dim,),
                description=f"Hydrodynamic force on {name!r} (N)",
                output_units="N",
            )
            if self._mesh.dim == 3:
                spec[f"torque_{name}"] = BoundaryFluxSpec(
                    shape=(3,),
                    description=f"Hydrodynamic torque on {name!r} (N·m)",
                    output_units="N*m",
                )
        if self._single_body is not None:
            # Shared-contract output aliases for the single immersed body.
            spec["drag_force"] = BoundaryFluxSpec(
                shape=(self._mesh.dim,),
                description="Hydrodynamic force on the body (N)",
                output_units="N",
            )
            if self._mesh.dim == 3:
                spec["drag_torque"] = BoundaryFluxSpec(
                    shape=(3,),
                    description="Hydrodynamic torque on the body (N·m)",
                    output_units="N*m",
                )
        if self._n_sample_points is not None:
            spec["velocity_at_points"] = BoundaryFluxSpec(
                shape=(self._n_sample_points, self._mesh.dim),
                description="Fluid velocity sampled at sample_points (m/s)",
                output_units="m/s",
            )
        return spec

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        # Build dynamic bodies from current boundary inputs. For the single
        # immersed body the shared-contract names (body_position / body_velocity
        # / body_angular_velocity) are accepted as aliases of the per-body
        # <name>_* inputs — see environment/FLUID_NODE_CONTRACT.md.
        def _body_in(name, suffix, contract_key, default):
            v = boundary_inputs.get(f"{name}_{suffix}")
            if v is None and name == self._single_body:
                v = boundary_inputs.get(contract_key)
            return default if v is None else v

        dynamic_bodies: list[IBMBody] = []
        for name, factory in self._dynamic_factories:
            body_inputs = {
                "position": _body_in(
                    name, "position", "body_position",
                    jnp.zeros(self._mesh.dim, dtype=self._mesh.V.dtype)),
                "linear_velocity": _body_in(
                    name, "linear_velocity", "body_velocity",
                    jnp.zeros(self._mesh.dim, dtype=self._mesh.V.dtype)),
            }
            if self._mesh.dim == 3:
                body_inputs["angular_velocity"] = _body_in(
                    name, "angular_velocity", "body_angular_velocity",
                    jnp.zeros(3))
            dynamic_bodies.append(factory(body_inputs))

        all_bodies = self._static_bodies + dynamic_bodies

        # Near→far reaction forcing (A2): spread the input point forces into a
        # body-acceleration field and add it to any static body_force_fn.
        body_force_fn = self._body_force_fn
        if self._n_forcing_points is not None:
            zeros_pts = jnp.zeros((self._n_forcing_points, self._mesh.dim),
                                  dtype=self._mesh.V.dtype)
            fp = boundary_inputs.get("forcing_points", zeros_pts)
            fv = boundary_inputs.get("forcing_values", zeros_pts)
            reaction_accel = spread_point_forces(
                fp, fv, self._mesh,
                sigma=self._forcing_sigma, rho=self._cfg.rho,
            ).astype(self._mesh.V.dtype)
            base_fn = self._body_force_fn

            def body_force_fn(t, _r=reaction_accel, _b=base_fn):
                if _b is None:
                    return _r
                return _r + jnp.asarray(_b(t))

        step = make_piso_step(
            self._mesh, self._bcs, self._cfg,
            body_force_fn=body_force_fn,
            ibm_bodies=all_bodies,
            lifting=self._lifting,
        )
        passable_keys = ("u", "p", "F", "t", "u_pre_ibm", "u_after_explicit", "i_step")
        new_state = step(
            {k: v for k, v in state.items() if k in passable_keys}, dt,
        )

        if self._force_method == "brinkman":
            # Per-cell Brinkman momentum-sink (biased low at moderate
            # IBM resolution; kept for backwards compatibility).
            forces = compute_ibm_forces(
                new_state["u_after_explicit"], self._mesh.x, self._mesh.V,
                dynamic_bodies,
                alpha=self._cfg.ibm_alpha, eps=self._cfg.ibm_eps,
                rho=self._cfg.rho, dt=dt,
            )
        elif self._force_method == "surface_integral":
            mu = self._cfg.rho * self._cfg.nu
            dx = self._mesh.cartesian_spacing[0]
            forces = {}
            for b in dynamic_bodies:
                F, T = surface_integral_force(
                    new_state["u"], new_state["p"], self._mesh, b.sdf,
                    mu=mu, dx=dx,
                    shell_inner=self._force_shell[0],
                    shell_outer=self._force_shell[1],
                    ref_point=b.ref_point,
                )
                forces[b.name] = {"force": F}
                if T is not None:
                    forces[b.name]["torque"] = T
        elif self._force_method == "momentum_deficit":
            # Control-volume momentum balance — recommended for
            # confined cases (λ ≳ 0.15) where surface integral suffers
            # IBM-band gradient contamination. Requires the static
            # bodies to include exactly one cylindrical pipe wall —
            # we look for the patch named "pipe_wall" in static bodies
            # and read the radius from its sdf attribute (assumes the
            # standard pipe-wall SDF set up via R_pipe).
            forces = {}
            mu = self._cfg.rho * self._cfg.nu
            for b in dynamic_bodies:
                F_z = momentum_deficit_drag(
                    new_state["u"], new_state["p"], self._mesh,
                    sphere_centre=b.ref_point,
                    sphere_radius=getattr(b, "_radius", 0.0),
                    pipe_radius=getattr(self, "_pipe_radius", 0.5),
                    pipe_axis=2, rho=self._cfg.rho,
                    margin_planes=4.0, body_force=0.0, mu=mu,
                )
                F_vec = jnp.zeros(self._mesh.dim, dtype=new_state["u"].dtype)
                F_vec = F_vec.at[2].set(F_z)
                forces[b.name] = {"force": F_vec}

        out = dict(new_state)
        dtype = self._mesh.V.dtype
        for name, _ in self._dynamic_factories:
            f = forces.get(name, {})
            out[f"force_{name}"] = f.get(
                "force",
                jnp.zeros(self._mesh.dim, dtype=dtype),
            ).astype(dtype)
            if self._mesh.dim == 3:
                out[f"torque_{name}"] = f.get(
                    "torque", jnp.zeros(3, dtype=dtype),
                ).astype(dtype)
            else:
                out[f"torque_{name}"] = f.get(
                    "torque",
                    jnp.zeros((), dtype=dtype),
                ).reshape(()).astype(dtype)
        if self._single_body is not None:
            # Shared-contract aliases for the single immersed body.
            out["drag_force"] = out[f"force_{self._single_body}"]
            out["drag_torque"] = out[f"torque_{self._single_body}"]
        if self._n_sample_points is not None:
            pts = boundary_inputs.get("sample_points")
            if pts is None:
                pts = jnp.zeros((self._n_sample_points, self._mesh.dim),
                                dtype=new_state["u"].dtype)
            # Reconstruct the physical velocity: PISO evolves u_hom when a
            # lifting field is present, so add the lift back before sampling.
            u_phys = new_state["u"]
            if self._lifting is not None:
                u_lift = self._lifting.at(new_state.get("i_step", 0))[0]
                u_phys = u_phys + u_lift
            out["velocity_at_points"] = sample_velocity_at_points(
                u_phys, pts, self._mesh).astype(new_state["u"].dtype)
        return out

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        out = {}
        for name, _ in self._dynamic_factories:
            out[f"force_{name}"] = state[f"force_{name}"]
            if self._mesh.dim == 3:
                out[f"torque_{name}"] = state[f"torque_{name}"]
        if self._single_body is not None:
            # Shared-contract aliases for the single immersed body.
            out["drag_force"] = state["drag_force"]
            out["drag_torque"] = state["drag_torque"]
        if self._n_sample_points is not None:
            out["velocity_at_points"] = state["velocity_at_points"]
        return out
