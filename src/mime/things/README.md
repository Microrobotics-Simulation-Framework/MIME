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
A body's mass, inertia, buoyant volume, extent, centre of buoyancy and magnetic moment may be declared in the
experiment (`exp.declare("umr.mass", ...)`); since phase 5 all but the extent can come from its parts instead.

Since phase 6: every thing has a prim path, an experiment name like its own. A thing written Python-first is at
`/World/<name>`; a body may bind a prim a referenced asset brings in (`Body(prim=scene["/World/UMR"], ...)`), and is
then named after the prim's last element unless it is given a name (two names per thing; a name or a prim held twice
is refused). `PrimPath` is such a path. A body bound to a prim gives parts below it their role: `magnets={"Magnets":
PermanentMagnet()}`, whose magnetisation direction is design data read from the part's prim (`mime.scene`), or
declared with `PermanentMagnet(direction=...)` when the prim carries none. A body with several magnets has its moment
derived from its parts (phase 5).

Since phase 5: a medium's `material` (a `mime.materials.Record`), and a body's `parts`, each a `Part`: a solid (a
registered `mime.registry.Solid`, from `mime.scene`), a material, a placement, and the part it lies `inside`, whose
volume and mass are counted in that part's (a magnet embedded in a shell declared at the body's mean density).
A part of a body written Python-first is placed with `at=` (its centroid in the body's frame, whose origin is the
centre of mass) and turned with the body; a part of a bound body is a prim below it, placed and turned by the
asset's transforms. A Python-first body's magnets are among its parts, with their directions declared. **The part
resolver** (`_parts.py`, `derive`) takes each part's volume, centroid and inertia per unit mass in the body's frame,
its density and its magnetisation (`B_r / mu_0` along its direction, a rigid magnetisation), and gives the body's
mass, inertia (by the parallel-axis theorem), buoyant volume, centre of buoyancy and magnetic moment, each with one
line per input. It refuses parts that put the centre of mass off the body's origin, and an inertia whose principal
axes are not the body frame's; it never shifts or turns them. Parts are not written to the stage yet.

Since phase 7a2, an apparatus's **`chain`**, segments by name in chain order (`chain={"arm": ..., "motor": ...}`),
its **`parts`**, each fixed to a frame of a segment (`Part(..., mount="motor.rotor", at=...)`) or to the world, and
its **`magnets`**, a part's role with its direction declared. An apparatus's moment is its one magnet's,
`<apparatus>.magnetic_moment`, declared as a magnitude. A body's `magnet_centre`, where its magnet sits, is derived
from its one magnet part, or declared.

Phases and releases are explained in the [package manifest](../README.md#status).
