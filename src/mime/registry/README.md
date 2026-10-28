# `mime.registry`

**How every kind of entry is registered, found and versioned.**

Interactions, observations, dynamics, field providers and controllers are all registered entries. So are mapping
kinds, geometry kinds and drive kinds. A lab adds its own the same way MIME adds MIME's, from its own package, without
editing MIME.

## What goes here

- **The registration functions**: `register_interaction`, `register_observation`, `register_dynamics`,
  `register_field_provider`, `register_controller`, and the same for the other kinds.
- **Keys of (name, version).** The saved document stores the registered name and version, so a rerun on a newer MIME
  either reproduces the result or says which model changed.
- **The small form of an entry**: its name and version, what it applies to, what it publishes and accepts, its
  capability flags, its own validity, and a build function that uses a restricted builder.
- **Support tables**, as predicates over capability records. There is no default model: a model choice (a "rung") is
  written in the script. An unsupported one is refused with the list of valid ones and their ranges.

## Boundaries

- The build only resolves names from installed, version-pinned packages. It never installs anything.
- Loading someone else's experiment never executes their code.
- One small fixture package outside `mime`, built in CI, proves that registering from outside works.

## Status

Landed in dev1's phase 2, the small form:
- the five `register_*` functions, keyed by (name, version), which refuse a malformed entry at registration;
- the entry bases (`Interaction`, `Observation`, `Dynamics`, `FieldProvider`, `Controller`): frozen dataclasses whose
  fields are their settings, each declaring what it applies to (roles, or thing kinds), what it reads, publishes,
  accepts and delivers, its flags, its validity (listed, not yet checked) and its differentiability;
- plans as data (`Plan`, `Read`, `Delivery`, `Registration`, `GroupRequest`) and capability records;
- support tables as predicates over capability records (`requires`, `SupportRow`), with no default rung.

Since phase 4a, **units in annotations**: an entry's dimensioned setting declares its dimension once, `margin:
Annotated[Quantity, Length]`, with the markers of `mime.units`. Registration reads the annotations
(`dimensioned_settings`, recorded on `Registered.settings`) and refuses a `Quantity` setting with no marker (`rate`
excepted, a period or a frequency with its own rule), a marker on a setting that holds no `Quantity`, and a default of
another dimension. Phase 1 refuses a setting of another dimension, naming the entry, the setting and both dimensions,
and hands the entry its settings in SI through its context: `ctx.scalar("margin")`, `ctx.vector("x0")`.

Since phase 4b: **drives** (`Drive`, `register_drive`: a command generator, one node and one exclusive edge into its
command) and **geometry kinds** (`Geometry`, `register_geometry`: regions, surfaces and directions, with each surface's
shape and parameters) are entries like the others, saved and loaded by name and version. A plan reads **values** of
the things it binds (`Plan.values`, `Value(role, name)`; a dynamics' `values` reads its own thing's), may hold
**confinements** (`Confinement`), and an interaction may name **reserved rungs** with their reasons. Settings may be
marked `Direction` (a thing's direction by reference, or three numbers; the entry reads the unit vector) or
`Commanded` (a drive's quantity, in the dimension of its command). Every entry may judge its settings in SI
(`setting_problems`), and a dynamics its initial state (`initial_problems`). A field provider declares its
`commands`. `RoleKind.SURFACE` binds a surface of a thing's geometry.

Since the first documentation phase, **the fields the docs are generated from**, on every entry: `builds`, each
rung's node classes, which the node walk checks against what a case builds; `gradient`, a `GradientRecord` of where
the derivative is not exact, why, and the route to `exact`, for every status but `exact`; `references`, bibliography
keys; and an interaction's `family`. A value the code does not establish is `NOT_ESTABLISHED`, listed with its reason
by `mime docs check`. Every MIME entry declares them, and every one's are established: its `references` are keys of
`docs/bibliography.bib`, each checked there by `mime docs check`, or `()` where it rests on no work. An entry whose
rungs differ in how a derivative passes through them declares `rung_gradients`, each built rung's own status and
record (`RungGradient`): the lock records the attached rung's, the pages print each rung's, and registration refuses a
declaration whose rungs are not the entry's built ones, whose entry status is not the least of its rungs', or whose
switches are not the entry's.

Since phase 7d, **a geometry kind's build-side capabilities**, each an optional method on `Geometry` a kind
implements or leaves (`implements(kind, "samples")`; `CAPABILITIES`): `samples(parameters, patch, counts)`, its surface
by named patch (`patches`, each mapped to the region whose boundary it lies on) as points, normals and weights, a JAX
function of its parameters so that a shape's derivatives flow through it, with `sample_counts` for a spacing; `area`;
`family_parameters`; `radial_reach`, about its own axis; and `tangent`, where it declares one. A **body's hull** is its
one part outside every other whose kind gives a radial reach (`Hull`): a law reads it through its context,
`ctx.geometry(of="body")`, as it reads the region a medium fills through `ctx.region`.

A name outside a registered namespace is MIME's, and is refused from a module outside `mime`. Mapping kinds register
with their first use. Namespace ownership through entry points, code fingerprints and claim
records come later.

Since phase 7a2: **`ChainSegment`**, the dynamics base of a chain segment (its frames, commands and description in
a `Segment`, its placement on its mount, why it takes no load), which a lab subclasses for a mechanism of its own; a
dynamics' `rungs`, `reserved_rungs` and `limits`; a field provider's `parts` and `values`; and a solid's
`enclosing_radius`.

Phases and releases are explained in the [package manifest](../README.md#status).
