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

Empty. It lands in dev1's phase 2.

Phases and releases are explained in the [package manifest](../README.md#status).
