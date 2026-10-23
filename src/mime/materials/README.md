# `mime.materials`

**The seam to OpenMaterica**, and the only importer of it.

## What goes here

- **Material reads.** Every dynamics, provider and interaction declares the material values it reads: the field, of
  which thing, with what requirement. Phase 1 sends those reads here.
  - OpenMaterica returns the record at the stated conditions, with its provenance.
  - Or it returns its verdict under the experiment's profile, which is passed through unchanged.
- **The inline floor.** A value declared in the experiment itself, with its reason. It serves exact historical
  reproductions, and anything with no record yet. The saved document holds references and declarations, never copies
  of record values.
- **Values become node parameters**, set at construction and never written after.
- **Later:**
  - per-region values, once OpenMaterica's scenes land;
  - materials read from bindings on the USD stage;
  - writing OpenMaterica's plain scene manifest through its region writer;
  - mixing rules.

## Co-development

MIME and OpenMaterica are developed together. CI runs MIME against OpenMaterica's `main` branch as well as its
release, so either side sees the other's breakage within a day.

## Status

Since phase 4b, the inline floor: `VALUE_KINDS` names every value of a thing laws read (a body's `mass`, `inertia`,
`buoyant_volume`, `centre_of_buoyancy`, `magnetic_moment` and `extent`; a medium's `density`), with its dimension,
shape and meaning; `resolve` reads one, and today only a declaration in the experiment provides one, labelled
"declared in the experiment" with its reason. Phase 1 reads every value through here, and nowhere else, so OpenMaterica's
records (phase 5) and the part resolver replace most declarations here without changing the entries that read them.

Phases and releases are explained in the [package manifest](../README.md#status).
