# EffectModel contract

```{versionadded} v0.2
The `mime.effects` package ships the **pilot** of the EffectModel contract —
the Protocol surface, the registry, the `Experiment` composition + validation,
and the `HydrodynamicModel` family. The `MagneticModel` family and
cross-effect coupling implementation land in v0.3.
```

The **EffectModel** abstraction is a common builder pattern for *environment
effects that deliver force/torque to a body*. The hydrodynamic family (fluid
drag) is the first concrete instance; the magnetic family follows in v0.3,
and acoustic / electric / thermal slots are designed-in for post-1.0. The
load-bearing insight (from `ADR-2026-EFFECT-MODEL`) is that the fluid-node
contract was never really a *fluid* abstraction — it is a builder for
"a subgraph that emits force/torque to a body", and fluids are just the first
case.

This page documents the v0.2 pilot. The full design rationale, the decision
log, and the post-1.0 extension path (Nelson-group eMNS / OctoMag coil arrays)
live in `ADR-2026-EFFECT-MODEL.md`.

## Why

A microrobotics experiment composes several physical effects on one body —
hydrodynamic drag, a magnetic moment in an applied field, possibly acoustic
streaming. Today each is wired by hand into a `GraphManager` with bespoke
edge helpers. The EffectModel contract makes effects **composable** and
**swappable**: attach a `HydrodynamicModel.LBM` and a (future)
`MagneticModel.PointDipole` to an `Experiment`, call `build()`, and get a
validated graph — with the option to swap the LBM backend for FVM, Stokeslet
or DefectCorrection across the same edges without touching the rest of the
graph.

Composability is a first-class v1.0 goal: the four-way hydrodynamic swap, the
magnetic-model swap, and multi-effect composition on one body are part of the
1.0.0 definition-of-done.

## Protocol surface

`mime.effects.protocol` defines the contract (after six rounds of design
iteration in the ADR):

```python
class EffectModel(Protocol):
    @property
    def coupling_ports(self) -> dict[str, EdgeSpec]: ...   # cross-effect output ports
    def applicable_regime(self) -> Regime: ...             # advisory regime metadata
    def required_body_properties(self) -> set[str]: ...    # per-physics body sections
    def required_medium_properties(self) -> set[str]: ...  # medium properties
    def build(self, gm, *, body, medium) -> EffectHandle: ...  # materialise the subgraph
```

* **`coupling_ports`** is an instance property, **computable from constructor
  arguments alone** — `build()` validation (pass 2) needs the EdgeSpecs before
  any subgraph exists. Empty for effects with no cross-coupling.
* **`build(gm, *, body, medium)`** materialises the effect's nodes and edges
  onto the `GraphManager`. Called once per `Experiment.build()`; the graph is
  **static for the lifetime of the run** (no mid-run mutation).
* **`EffectHandle`** is a pure-introspection return value — the names of the
  nodes the effect added.

`SourcedEffectModel[S]` is the sub-protocol for effects with external sources
(magnetic dipoles, coils, acoustic transducers). `S` is a **covariant**
`TypeVar` bound to `Source` because it appears only in return position
(`sources`), so `SourcedEffectModel[MagneticSource]` is usable where
`SourcedEffectModel[Source]` is expected. The hydrodynamic pilot has no
sources; the type exists so the magnetic family slots in without a refactor.

`BaseEffectModel` is a convenience base supplying the no-op declarations
(empty `coupling_ports`, empty required-property sets) so concrete models
override only what they use.

### Regimes

`Regime` is a polymorphic ABC — each physics family ships its own subclass
with its own `check()`, and the family-agnostic caller just invokes
`model.applicable_regime().check(...)` (no central `isinstance` dispatch).
`HydrodynamicRegime(re_range)` emits a `RegimeWarning` when the medium's
Reynolds number falls outside the validated creeping-flow range. **Regimes
are advisory only** — they surface warnings, never gate execution, because
cross-domain regime classifications are research-active and baking a gate in
would lock in a wrong classification.

## Registry

```python
from mime.effects import register_effect, list_registered_effects, get_effect

@register_effect("HydrodynamicModel.LBM")
class LBM(_HydrodynamicEffect): ...

get_effect("HydrodynamicModel.LBM")      # -> the class
list_registered_effects()                # -> {name: class}
```

The registry is the single source of truth for YAML `type:` string
resolution, MICROROBOTICA's effect-picker dropdowns, and serialisation
round-trips. A decorator (not importlib auto-discovery) keeps the surface
auditable and avoids arbitrary-code config files.

## Body and Medium

`Body` and `Medium` are **fat dataclasses** (not Protocols) — at v1.0 scale
(≤ 5 families, ≤ 25 fields total) the simpler approach wins, with a written
refactor tripwire in the ADR.

* **`Body(name, node, properties)`** — `name` is the body's node name in the
  graph (effects wire `drag_force` / `drag_torque` / `magnetic_*` edges into
  it); `node` is the body node itself (added to the graph once by
  `Experiment.build`); `properties` holds per-physics property *sections*,
  e.g. `{"hydrodynamic": {...}}`.
* **`Medium(properties)`** — the surrounding-fluid properties, e.g.
  `{"density": 1060.0, "viscosity": 1.2e-3, "reynolds_number": 1e-3}`.

## Experiment composition and the six-pass `build()`

`Experiment` formalises the existing directory convention and adds the effect
composition surface:

```python
exp = Experiment(name="ar4_helical_drive",
                 mime_version_min="0.2.0")
exp.set_body(Body(name="rigid_body", node=rigid, properties={"hydrodynamic": {}}))
exp.set_medium(Medium(properties={"density": 1060.0, "viscosity": 1.2e-3}))
exp.attach(HydrodynamicModel.LBM(lbm_node, dx_physical=dx, dt_physical=dt))
gm, handles = exp.build()
```

`build()` runs **six ordered passes**, cheap checks first; each pass assumes
its predecessors succeeded, and each failure raises a specific typed error so
the validation promise is testable rather than aspirational:

| # | Pass | On failure |
|---|---|---|
| 1 | **Effect-reference resolution** — every `CouplingSpec` source/target resolves to an attached effect. | `CouplingError` |
| 2 | **Port existence + type compatibility** — source port ∈ `coupling_ports`, target port ∈ the target's input ports, EdgeSpecs structurally compatible. | `CouplingError` / `PortTypeMismatchError` |
| 3 | **Body/Medium property presence** — every required body section and medium property is present. | `BodyPropertyMissing` / `MediumPropertyMissing` |
| 4 | **Regime check** — advisories collected onto `Experiment.warnings` and emitted via `warnings.warn`. **Execution continues.** | *(warnings only)* |
| 5 | **Per-effect `build()`** — each attached model materialises its subgraph in attachment order. | `EffectBuildError` (wraps the underlying exception) |
| 6 | **`gm.compile()`** — MADDENING's edge validation runs. | MADDENING errors pass through |

### Load-time version validation

The `Experiment` carries version-and-provenance metadata as concrete fields:
`mime_version_min`, `mime_version_max`, `asset_paths`, `asset_hashes`,
`benchmark_refs`, `citation`. **Version compatibility is validated at
construction (load) time, before `build()`** — a clean-install reproducer
gets a clear `IncompatibleMimeVersionError` ("your MIME is too old / too
new") immediately, not after graph construction.

## HydrodynamicModel family

`HydrodynamicModel` is a namespace of four swappable backends, each adapting
an existing fluid node over the shared `FLUID_NODE_CONTRACT.md` drag
interface:

| Backend | Wraps | Edges |
|---|---|---|
| `HydrodynamicModel.LBM` | `IBLBMFluidNode` | lattice→SI drag transforms (`make_iblbm_rigid_body_edges`) |
| `HydrodynamicModel.Stokeslet` | `StokesletFluidNode` | SI (`make_stokeslet_rigid_body_edges`) |
| `HydrodynamicModel.FVM` | `FVMFluidNode` | SI (generic `drag_force` / `drag_torque`) |
| `HydrodynamicModel.DefectCorrection` | `DefectCorrectionFluidNode` | SI (generic) |

Each is an **adapter** — it wraps a pre-constructed fluid node (backend
parameters belong on the node) and wires its `drag_force` / `drag_torque`
into the body. Because the fluid nodes emit the contract names, swapping one
backend for another across the same body edges compiles clean — this is the
interchangeability guarantee, exercised in
`tests/effects/test_effect_model.py`.

## What's deferred to v0.3

Per the ADR scope boundaries, the v0.2 pilot deliberately excludes:

* **`MagneticModel` family** — `PointDipole` / `UniformField` refactor of the
  existing magnetic chain (per `MAGNETIC_NODE_AUDIT.md`).
* **`SourceInputProvider`** — `ConstantInput | NodeFieldRef | ExternalInputRef`
  carried on each source for its required external inputs (pose, coil current,
  …).
* **Cross-effect coupling implementation** — the ports are designed-in, but the
  additive operator-splitting / strong-coupling implementation behind them is
  v0.3 (the realistic cross-coupling cases — acoustic streaming, MHD — are
  post-1.0).
* **Full `make_experiment(params) -> Experiment` migration** of the de Boer
  and de Jongh experiments — those experiments' magnetic actuation chains are
  not yet EffectModels, so they cannot be expressed purely through the pilot.
* **Acoustic / electric / thermal-radiation** model slots — designed-in,
  unimplemented (acoustic contingent on MADDENING multi-rate verification).

## References

* `ADR-2026-EFFECT-MODEL.md` — the accepted design, decision log, and post-1.0
  extension path.
* `MAGNETIC_NODE_AUDIT.md` — the magnetic-family audit feeding v0.3.
* `src/mime/nodes/environment/FLUID_NODE_CONTRACT.md` — the shared fluid-node
  interface the `HydrodynamicModel` backends build on.
