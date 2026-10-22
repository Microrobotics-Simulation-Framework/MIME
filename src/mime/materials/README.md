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

Empty. Record reads land in dev1's phase 5.

Phases and releases are explained in the [package manifest](../README.md#status).
