# `mime.things`

**The things an experiment contains, and how scripts refer to them.**

## What goes here

- **`Structure`**: a passive solid such as a tube wall, a phantom or a channel wall. It has a material, a geometry and
  dynamics. The default dynamics is `Fixed`, which declares the rigid-wall assumption as a line the build can check.
- **`Medium`**: water, blood, a gel. It `fills=` a structure's region, so a lumen's geometry is written once. It has
  one dynamics per field it carries. Its flow is quiescent unless declared otherwise.
- **`Body`**: a robot. It has parts with roles (a magnet, a shell), a dynamics and an initial `State`.
- **`Apparatus`**: what drives the experiment: a coil array, or a robot arm carrying a magnet. It has parts, a chain,
  field providers, a driver, a calibration and commands.
- **The part resolver.** It derives mass, centre of mass, inertia, buoyant volume and magnet moments from parts and
  materials, and records each derived value.
- **Handles and typed paths.** `add_body(...)` returns a handle, so a typo is a Python error on its line and a rename
  is an IDE rename. Typed paths reach deeper: `tube.axis`, `umr.parts.magnets.moment`. Strings are still accepted
  everywhere, and are validated with "did you mean" suggestions.

## Boundaries

- Declarations only: a thing builds nothing itself. Its dynamics entry (from `dynamics/`) builds its nodes.
- A thing is named once. The document saves names, never Python object identities.

## Status

Landed in dev1's phase 2: the four thing kinds as declarations with names, dynamics and a body's initial `State`;
handles; typed paths for an apparatus's fields (`source.magnetic`); `did_you_mean`.

Since phase 4b: a structure's `geometry` (a registered geometry kind, `Tube`) and its typed paths from the handle,
`tube.lumen` (a `RegionPath`), `tube.inner_surface` (a `SurfacePath`) and `tube.axis` (a `DirectionPath`); a
medium's `fills=` region; an apparatus's `commands`, each a drive (a quantity being short for the constant drive).
Materials and parts with roles come with phase 5, and so does the part resolver; until then a body's mass, inertia,
buoyant volume, extent, centre of buoyancy and magnetic moment are declared in the experiment
(`exp.declare("umr.mass", ...)`).

Since phase 6: every thing has a prim path, an experiment name like its own. A thing written Python-first is at
`/World/<name>`; a body may bind a prim a referenced asset brings in (`Body(prim=scene["/World/UMR"], ...)`), and is
then named after the prim's last element unless it is given a name (two names per thing; a name or a prim held twice
is refused). `PrimPath` is such a path. A body bound to a prim gives parts below it their role: `magnets={"Magnets":
PermanentMagnet()}`, whose magnetisation direction is design data read from the part's prim (`mime.scene`), or
declared with `PermanentMagnet(direction=...)` when the prim carries none. One magnet per body until the part
resolver sums several.

Phases and releases are explained in the [package manifest](../README.md#status).
