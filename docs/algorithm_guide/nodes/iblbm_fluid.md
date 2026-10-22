---
bibliography: ../../bibliography.bib
---

# IB-LBM Fluid Node

**Module**: `mime.nodes.environment.lbm.fluid_node`
**Class**: `IBLBMFluidNode`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-010`
**Version**: 1.1.0
**Verification Mode**: Mode 2 (Independent)

## Summary

3D immersed-boundary lattice Boltzmann ({term}`IB-LBM`) fluid solver for
confined microrobot flows. Wraps the existing D3Q19 LBM utilities
(`d3q19.py`, `bounce_back.py`, `helix_geometry.py`) as a MADDENING
{term}`SimulationNode` so the volumetric flow can be coupled into a
node-graph with rigid-body and magnetic-actuation nodes via edges.

The pipe wall uses simple bounce-back; the immersed body uses simple
bounce-back or {term}`Bouzidi interpolated bounce-back <Bouzidi scheme>` depending on
`use_bouzidi`.

The immersed body is **geometry-agnostic**: it is supplied by a
`LatticeBodyFactory` and may translate and rotate. See *Body geometry hook*.

## Body geometry hook

```{versionadded} 1.1.0
The body was previously hard-wired to `create_umr_mask` at a frozen grid
centre. It is now built per step by a factory from the body's pose.
```

`mime.nodes.environment.lbm.body` defines the LBM half of the shared
fluid-node body abstraction:

```python
LatticeBodyFactory = Callable[[pose, LatticeGrid], LatticeBody]
```

`update()` calls the factory each step with the traced pose assembled from the
contract inputs (`body_position` / `body_velocity` / `body_angular_velocity` /
`body_orientation`, all in lattice units) plus the node's integrated z-angle.
A `LatticeBody` carries the occupancy `mask`, an optional `sdf` (needed only
for Bouzidi), the `ref_point` (torque lever-arm origin), and the body's
`velocity` / `angular_velocity`.

Two geometries ship:

| Factory | Mask | SDF | Angular velocity |
|---|---|---|---|
| `make_umr_body_factory` | `create_umr_mask` (body \| fins) | `umr_sdf` | projected onto **z** |
| `make_sphere_body_factory` | `create_sphere_mask` | `fvm.sdf.sphere_sdf` | full vector |

The UMR projects `omega` onto z because `create_umr_mask` is parameterised by a
single z-rotation angle — imposing the full vector would apply a wall velocity
the mask's motion does not reproduce. The sphere is rotationally symmetric, so
it takes the whole vector.

### Relationship to `FVMFluidNode`

This deliberately mirrors `FVMFluidNode`'s `BodyFactory` / `IBMBody` pair —
same pose input names, same "rebuild the body inside `update()`" structure — so
the two solvers are swappable for one experiment (see
`environment/FLUID_NODE_CONTRACT.md`). The dataclass is **parallel, not
shared**, because the two solvers need different primitives:

* FVM's diffuse-penalty IBM only evaluates `sdf(x)` at cell centres, so an SDF
  *is* the whole body.
* LBM bounce-back is defined by which **neighbouring lattice node** is solid, so
  it needs a boolean occupancy mask. An SDF can yield a mask, but the shipped
  UMR mask is an analytic region test that is **not** the zero-set of `umr_sdf`
  — thresholding the SDF would silently change the validated
  `umr_confinement` result.
* LBM works in lattice units; FVM in SI.

What *is* shared: the pose input names, the factory shape, and the sphere SDF
itself (`fvm.sdf.sphere_sdf` is imported, not re-derived).

### Backward compatibility

Passing the legacy `body_geometry_params` still selects the UMR and reproduces
the pre-hook behaviour exactly:

* it is equivalent to `body_factory=make_umr_body_factory(**geom)` (pinned by
  `test_legacy_params_equal_explicit_umr_factory`);
* `body_position` defaults to the **body reference position** (the grid centre),
  not the origin, so an unwired graph leaves the body where it was frozen;
* `make_iblbm_rigid_body_edges` does **not** wire translation unless asked
  (`wire_translation=True`) — turning it on would change `umr_confinement`'s
  validated results.

## Governing Equations

D3Q19 BGK lattice Boltzmann equation:

$$
f_i(\mathbf{x}+\mathbf{c}_i\Delta t,\, t+\Delta t)
= f_i(\mathbf{x},t) - \tfrac{1}{\tau}\big(f_i - f_i^{\text{eq}}\big),
$$

with macroscopic moments

$$
\rho = \sum_i f_i, \qquad \rho\,\mathbf{u} = \sum_i \mathbf{c}_i f_i,
$$

and the equilibrium expansion to $O(u^2)$ in the lattice speed.

Body coupling uses {term}`momentum-exchange <Momentum exchange>` [@Ladd1994] on the post-streaming
populations along boundary links, giving force and torque

$$
\mathbf{F}_b = \sum_{\text{links}} (f_i + f_{\bar i})\,\mathbf{c}_i,
\qquad
\mathbf{T}_b = \sum_{\text{links}} (\mathbf{r}-\mathbf{r}_{cm}) \times
        (f_i + f_{\bar i})\,\mathbf{c}_i.
$$

## Discretization

* D3Q19 stencil; BGK collision with relaxation time $\tau$.
* Two-pass bounce-back per step: static pipe wall first, then the
  moving body. The body mask is rebuilt each step by the geometry
  hook at the current pose, so the wall geometry stays
  differentiable in (and traceable through) that pose.
* Body wall velocity is the rigid-body field
  $\mathbf{u}(\mathbf{x}) = \mathbf{v} + \boldsymbol\omega \times
  (\mathbf{x}-\mathbf{x}_{\text{ref}})$
  (`body.rigid_body_velocity_field`), evaluated at *every* node — the Ladd
  correction reads it at fluid boundary nodes, so it must not be masked to
  the solid.
* Bouzidi interpolated bounce-back [@Bouzidi2001] uses sparse SDF
  $q$-values along each missing link; `max_boundary_links_per_dir`
  is pre-sized at construction to avoid silent truncation inside
  `jnp.nonzero`.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---|---|---|
| BGK collision + streaming | `mime.nodes.environment.lbm.d3q19.lbm_step_split` | |
| Equilibrium $f_i^{\text{eq}}$ | `mime.nodes.environment.lbm.d3q19.init_equilibrium` | |
| Pipe wall bounce-back | `mime.nodes.environment.lbm.bounce_back.apply_bounce_back` | First pass |
| Bouzidi IBB on UMR | `mime.nodes.environment.lbm.bounce_back.apply_bouzidi_bounce_back` | Second pass |
| Sparse $q$-values from SDF | `mime.nodes.environment.lbm.bounce_back.compute_q_values_sdf_sparse` | |
| Momentum-exchange force | `mime.nodes.environment.lbm.bounce_back.compute_momentum_exchange_force` | |
| Momentum-exchange torque | `mime.nodes.environment.lbm.bounce_back.compute_momentum_exchange_torque` | |
| LBM→SI conversion | `maddening.core.transforms.lbm_to_si_force/torque` | Edge transform |

## Assumptions and Simplifications

1. Incompressible flow: $\mathrm{Ma} \ll 1$ at every lattice node
   (validated for $\mathrm{Ma}_{\text{tip}} \le 0.1$).
2. Newtonian fluid; viscosity derived from $\tau$.
3. Rigid body. The **geometry** decides how much of the twist it can
   represent: the UMR mask is parameterised by a z-angle so it consumes only
   `body_angular_velocity[2]`; the sphere consumes the full vector. Both
   consume `body_position` / `body_velocity`.
4. Stencil node: `halo_width()` returns `{0: 1, 1: 1, 2: 1}` (D3Q19
   streaming reads one neighbour per spatial axis). This marks the node
   non-pointwise — it runs on a single device by default, and shards across
   devices when wrapped in MADDENING v0.2.1's `ShardedStencilNode` (see
   *Multi-GPU sharding* below). See
   [Node API migration](../../architecture/node_api_migration.md).
5. The static pipe-wall mask is constructed once at init; only the
   dynamic body mask is rebuilt per step.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|---|---|---|
| $\mathrm{Ma}$ | 0 – 0.1 | Mach bound (max over all nodes, incl. fin tips) |
| $\tau$ | 0.55 – 1.5 | BGK relaxation stability window |
| $\mathrm{Re}_{\text{rot}}$ | 0 – ~30 | Tested across {term}`helical-UMR <Helical UMR>` replication cases |
| $\mathrm{Re}_{\text{translation}}$ | 0 – ~1 | Translating-sphere drag vs Stokes + Haberman–Sayre (MIME-VER-012). **Stokes regime only** |

```{warning}
`Re_translation` above ~1 is **admissible but unverified**. The finite-Re
sphere wake — the Schiller–Naumann regime, and vortex shedding above
$\mathrm{Re}\approx270$ — has *not* been verified in this node. Do not read the
Re 0–360 anatomical range below as a validation claim; it is an
*affordability* statement (see the next section).
```

## The Mach–Reynolds–resolution constraint

Weakly-compressible LBM does **not** let you choose $\mathrm{Re}$ freely: it is
tied to the Mach number and the grid. With

$$
\nu_{lu} = \tfrac{\tau - 1/2}{3},\qquad
\mathrm{Ma} = \sqrt{3}\, U_{lu},\qquad
\mathrm{Re} = \frac{U_{lu} D_{lu}}{\nu_{lu}},
$$

eliminating $U_{lu}$ gives the resolution a target Reynolds number demands:

$$
\boxed{\;D_{lu} \;\ge\; \frac{\sqrt{3}\,\mathrm{Re}\,(\tau - 1/2)}{3\,\mathrm{Ma}_{\max}}\;}
\qquad\Longrightarrow\qquad
D_{lu} \ge 0.289\,\mathrm{Re}
\quad(\tau = 0.55,\ \mathrm{Ma}_{\max} = 0.1).
$$

$\tau=0.55$ is the stability floor and $\mathrm{Ma}=0.1$ the accuracy ceiling
(compressibility error is $O(\mathrm{Ma}^2)\approx1\%$), so **0.289 Re is the
cheapest admissible resolution**, not a typical one. Raising Re on a fixed grid
does not "just work" — it raises Ma (compressibility error) or pushes $\tau$
toward the $0.5$ singularity. *Resolve* Re; don't push Ma.

### Cost envelope — a 1.7 mm sphere in a 6.3 mm ID tube

The motivating case ($D_{\text{sph}}=1.7$ mm, tube ID $6.3$ mm, so the tube is
$3.7\,D_{\text{sph}}$ across; tube length taken as $3\times$ ID). At
$\tau=0.55$, $\mathrm{Ma}=0.1$:

| Re | $D_{lu}$ | $dx$ | tube grid ($D\times D\times L$) | cells | `f` alone (f32) | working set (~4×) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 29 | 59 µm | $107^2\times320$ | 3.7 M | 0.28 GB | ~1.1 GB |
| 200 | 58 | 29 µm | $215^2\times645$ | 30 M | 2.3 GB | ~9 GB |
| 300 | 87 | 20 µm | $322^2\times967$ | 100 M | 7.6 GB | ~30 GB |
| 360 | 104 | 16 µm | $385^2\times1156$ | 171 M | 13 GB | ~52 GB |

So: **Re ≈ 100 is cheap** (fits a laptop GPU); **Re ≈ 300–360 needs an 80 GB
accelerator or multi-GPU sharding**, and is an O(10⁸)-cell run. This is a real
cost, not a soft one — budget for it rather than discovering it. Time-to-steady
adds a second factor: the viscous timescale is $R_{\text{tube},lu}^2/\nu_{lu}$
steps ($\sim1.7\times10^5$ steps at Re 360), and above $\mathrm{Re}\approx270$
the wake sheds, so drag must be time-averaged rather than read at a final step.

Two levers relax the table, both with a cost:

* **Raise $\tau$** — $D_{lu}$ scales with $(\tau-1/2)$, so $\tau=0.55$ is
  already the cheapest. Raising $\tau$ makes it *worse*.
* **Accept $\mathrm{Ma}>0.1$** — $D_{lu}\propto1/\mathrm{Ma}$, but
  compressibility error grows as $\mathrm{Ma}^2$ and the validated envelope
  ends at 0.1. Outside the validated range.

## Known Limitations and Failure Modes

1. **No ambient-flow / inflow boundary condition.** Relative flow must be
   produced by *translating the body through the lattice*; there is no
   co-moving-frame or uniform-inflow option. (`bounce_back.py` has
   `apply_velocity_inlet` / `apply_pressure_outlet` / `poiseuille_profile_3d`,
   but the node does not wire them.) Consequences: a body driven far along a
   vessel needs a correspondingly long lattice, and the anatomical flow regime
   is honestly *stagnant ambient*, not a driven vessel flow.
2. **No fresh-node refill for a moving boundary.** When the body advances,
   lattice nodes it vacates become fluid with no valid populations — the
   classic *fresh/uncovered node* problem. This node has no refill scheme
   (Lallemand–Luo / Caiazzo); solid nodes are never zeroed and keep collide-ing
   as if fluid, which acts as a crude implicit refill.

   Measured consequences (MIME-VER-015 and
   `test_translating_body_mass_offset_is_bounded_not_a_leak`), which are milder
   than the textbook worry:

   * **Drag ripple, no mean bias.** The drag develops a ripple locked to the
     one-cell-crossing period (3.1% peak-to-peak with simple BB, 1.25% with
     Bouzidi). The period-*averaged* drag matches the fixed-mask equivalent to
     0.4%, so the sweep costs a ripple, not a systematic error. Average over a
     crossing period.
   * **Mass error is a bounded offset, not a leak.** It saturates (~8e-4 at
     $U_{lu}=0.004$) within ~500 steps and does not grow over 24 swept cells;
     it scales with $U_{lu}$ and appears identically with a *fixed* mask, so it
     is the startup density adjustment to switching on a moving boundary, not
     the sweep. Long runs are therefore safe from compounding mass error.

   Keep $U_{lu}\ll1$ cell/step (it already must be, for Ma). This remains the
   least-validated part of the translation path — the above is characterised
   only at $U_{lu}\le0.004$ and $\mathrm{Re}<1$.
3. **Streaming is periodic; the geometry is not.** `lbm_step_split` wraps `f`
   with `jnp.roll` on every axis, but the masks use unwrapped coordinates. A
   body translated past the lattice edge therefore does **not** re-enter — it
   silently leaves the domain and the drag decays to noise. There is no guard.
4. `body_orientation` is plumbed to the geometry hook but **no shipped
   geometry consumes it**: the UMR mask is parameterised by the node's
   integrated z-angle (`body_angle`), and the sphere is symmetric. A body mask
   driven by a full quaternion is not implemented.
5. Bouzidi interpolated bounce-back is **not yet supported on the
   multi-GPU sharded path** (it needs per-slab SDF q-value recomputation);
   the node guards `use_bouzidi=True` under sharding with a clear
   `NotImplementedError`. Simple bounce-back shards. Single-device runs
   support both.
6. Per-step $q$-value recomputation costs ~0.1 s at $192^3$ on H100.
7. First step triggers JAX compilation (30–60 s at $192^3$).
8. If `max_boundary_links_per_dir` is undersized, `jnp.nonzero`
   **silently truncates** boundary links and accuracy degrades without
   error. The node pre-sizes it as `count_boundary_links()` (at the body's
   reference pose) × `boundary_links_margin` (default 1.5). The 1.5× margin was
   chosen for a *rotating, fixed-centre* UMR. **A translating body's link count
   also varies with its sub-cell position, so the reference-pose count is not
   an upper bound** — sweep the intended trajectory with
   `node.count_boundary_links(position_lu=...)` and raise the margin.
9. Diffuse IB kernels (Peskin delta) would create
   direction-dependent velocity transfer errors at clinical
   frequencies; this node uses bounce-back instead.
10. `compute_momentum_exchange_torque` is ill-conditioned for *moving curved
    walls* (a near-cancellation whose error grows with resolution) — see its
    docstring. Forces and static boundaries are fine.
11. **A periodic tube carries a spurious co-flow.** Streaming is periodic on
    every axis, so a body translating along the tube injects momentum whose
    only sink is wall shear — and wall shear only exists if the tube carries a
    mean flow. The tube therefore develops a steady co-flow *in the direction
    of travel*, so the body's true relative velocity is **below** its
    prescribed `body_velocity`. Measured at 14% of $U$ on the MIME-VER-015
    configuration ($L_z = 96$, $R_{\text{tube}} = 22$); it scales as
    $\bar u \sim F/(8\pi\mu L_z)$, i.e. $\propto 1/L_z$. **This is not a
    solver error** — it is what a periodic domain means — but it biases any
    comparison against a sealed-tube reference (Haberman–Sayre) and any
    force-driven trajectory. Mitigate with a long tube, or re-reference to the
    measured far-field velocity.

## Stability Conditions

* $\tau \ge 0.55$ to keep BGK stable for the discrete equilibrium.
* $\mathrm{Ma} \le 0.1$ for the incompressible-limit expansion to
  hold pointwise.

## Multi-GPU sharding

```{versionadded} v0.2
Pencil decomposition across devices via MADDENING v0.2.1's
`ShardedStencilNode`.
```

Construct the node with `multigpu_shard_axis=<spatial axis>` and wrap it:

```python
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

node = IBLBMFluidNode(..., multigpu_shard_axis=2)
mesh = create_device_mesh(shape=(4,))
sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 2},
                             boundary="periodic")
```

The sharded `update_padded` collides on the halo-padded slab, streams with
the slice-based `d3q19.stream_padded`, recomputes the pipe + UMR missing-link
masks per slab (`bounce_back.compute_missing_mask_sharded` — periodic roll on
the full axes, halo-slice on the shard axis), rebuilds the UMR body and
rotation-velocity field on the slab's *global* coordinates (an `origin`
offset on the geometry helpers and the momentum-exchange torque), and returns
`drag_force` / `drag_torque` as per-slab partial sums that the wrapper
`lax.psum`s (declared via `domain_integral_fields()`). Only the pipe wall is a
sharded `StaticArray` (it is axis-aligned `(nx, ny, nz)`); the missing-link
masks are recomputed per slab.

This is **bit-identical** to the single-device step under jit, validated on a
4-device CPU virtual-device mesh in
`tests/verification/test_lbm_sharded_contract.py`. Bouzidi IBB on the sharded
path is deferred (see *Known Limitations*).

## State Variables

| Field | Shape | Units | Description |
|---|---|---|---|
| f | (nx, ny, nz, 19) | lattice | D3Q19 populations (the only spatial state) |
| body_angle | () | rad | Accumulated body rotation |
| drag_force | (3,) | lattice | Momentum-exchange force — a domain-integral *output*, not evolving state (`domain_integral_fields`) |
| drag_torque | (3,) | lattice | Momentum-exchange torque — domain-integral output |

The pipe wall and its missing-link mask are **not** state: the pipe wall is a
non-evolving `static_data` array and the UMR mask is recomputed from
`body_angle` each step. `state_fields()` is therefore `["f", "body_angle"]`.

## Parameters

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| nx, ny, nz | int | — | cells | Lattice dimensions |
| tau | float | — | — | BGK relaxation time |
| vessel_radius_lu | float | — | lattice | Pipe radius in lattice units |
| body_factory | `LatticeBodyFactory` \| None | None | — | Geometry hook: `(pose, grid) -> LatticeBody`. Exactly one of this / `body_geometry_params` |
| body_geometry_params | dict \| None | None | lattice | Legacy UMR shorthand — `create_umr_mask` kwargs; equivalent to `make_umr_body_factory(**geom)` |
| use_bouzidi | bool | False | — | Interpolated bounce-back on body (needs `LatticeBody.sdf`; not on the sharded path) |
| dx_physical | float | 1.0 | m | Physical lattice spacing (for SI conversion) |
| body_reference_position_lu | (3,) \| None | grid centre | lattice | Pose held when `body_position` is unwired; also the pose used to pre-size the Bouzidi link buffer |
| boundary_links_margin | float | 1.5 | — | Safety factor on the at-reference boundary-link count (see *Known Limitations* #8) |
| multigpu_shard_axis | int \| None | None | — | Spatial axis (0/1/2) to decompose across devices; set when wrapping in `ShardedStencilNode` (see *Multi-GPU sharding*) |

## Boundary Inputs

The shared fluid-node contract's `body_*` inputs
(`environment/FLUID_NODE_CONTRACT.md`), in **lattice units** — the LBM node is
the contract's documented lattice-units exception, so the edges carry the
conversions (`make_iblbm_rigid_body_edges`).

| Field | Shape | Default | Coupling Type | Description |
|---|---|---|---|---|
| body_position | (3,) | **reference position** (grid centre) | replacive | Body centroid [cells]. Default is the reference pose, *not* the origin, so an unwired graph pins the body where the pre-1.1.0 node froze it |
| body_velocity | (3,) | zeros | replacive | Linear velocity [cells/step] |
| body_angular_velocity | (3,) | zeros | replacive | Angular velocity [rad/step] |
| body_orientation | (4,) | (1,0,0,0) | replacive | Orientation quaternion. Passed to the hook; no shipped geometry reads it (Limitation #4) |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|---|---|---|---|
| drag_force | (3,) | lattice | Momentum-exchange force (convert via edge) |
| drag_torque | (3,) | lattice | Momentum-exchange torque (convert via edge) |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | Viscosity Range |
|---|---|---|---|
| Iliac artery (confined rotating UMR) | quasi-stagnant | 0 – 0.1 | 3 – 4 mPa·s |
| Large vessel / in-vitro tube (translating millibot) | stagnant ambient | 0 – 360 | 0.7 – 4 mPa·s |

```{important}
The second row is an **affordability** bound, not a validation claim. It says
the solver *admits* a particle-Reynolds regime of 0–360 — at the resolution the
[Mach–Reynolds–resolution constraint](#the-machreynoldsresolution-constraint)
demands, which at Re 360 is an O(10⁸)-cell run. Verified drag is **Stokes
regime only** (MIME-VER-012). The flow regime is *stagnant ambient* because the
node has no inflow BC (Limitation #1) — the relative velocity comes from the
body moving through otherwise-quiescent fluid.
```

### Clinical Relevance

LBM gives the full volumetric flow that drives pulsatile drag and
near-wall lubrication on the UMR. In the hybrid solver, LBM
provides the background velocity that the
[Stokeslet BEM](stokeslet_fluid.md) uses as its {term}`Schwarz <Schwarz coupling>` boundary
condition — the LBM does *not* discretise the body itself, which
sidesteps Mach-number and IB-kernel pathologies at high rotation
rates.

## References

- [@Ladd1994] Ladd (1994). *Numerical simulations of particulate suspensions via a discretized Boltzmann equation.* J. Fluid Mech. 271, 285–309.
- [@Bouzidi2001] Bouzidi, Firdaouss & Lallemand (2001). *Momentum transfer of a Boltzmann-lattice fluid with boundaries.* Phys. Fluids 13(11).

## Verification Evidence

- MIME-VER-lbm-001: Poiseuille flow in cylinder (analytical)
- MIME-VER-lbm-002: Womersley unsteady pipe flow
- MIME-VER-lbm-003: Rotating-sphere torque vs. Stokes formula
- **MIME-VER-012**: translating-sphere drag vs Stokes + Haberman–Sayre tube
  correction — `tests/verification/test_lbm_sphere_drag.py`
- Geometry hook + translation unit tests: `tests/nodes/lbm/test_lbm_body.py`
- UMR no-regression guards: `tests/nodes/lbm/test_iblbm_fluid_node.py`
  (`test_update_matches_sweep_logic`, `test_bouzidi_path_matches_reference`)
  and `test_lbm_body.py::test_legacy_params_equal_explicit_umr_factory`
- Unit tests: `tests/nodes/lbm/`
- Multi-GPU bit-compat: `tests/verification/test_lbm_sharded_contract.py`

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-02-18 | Initial implementation — D3Q19 BGK + Bouzidi IBB |
| 1.1.0 | 2026-05-31 | v0.2: `static_data` masks, `halo_width()`, and multi-GPU sharding — implemented `update_padded`'s multi-device path (bit-identical to single-device on a 4-device mesh); simple bounce-back shards, Bouzidi deferred |
| 1.1.0 | 2026-07-15 | **Geometry hook + translation.** Body is no longer hard-wired to the UMR: `LatticeBodyFactory` (`body.py`) mirrors FVM's `BodyFactory`/`IBMBody`; `make_sphere_body_factory` added. `body_position` / `body_velocity` consumed (opt-in edges via `wire_translation=True`); `body_orientation` plumbed to the hook. Honest regime + cost envelope for the translating-body case. UMR path unchanged (defaults pin the body at the grid centre). |
