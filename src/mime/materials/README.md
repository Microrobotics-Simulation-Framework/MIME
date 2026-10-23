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
`buoyant_volume`, `centre_of_buoyancy`, `magnetic_moment` and `extent`; a medium's `density` and `viscosity`),
with its dimension, shape and meaning. Phase 1 reads every value through here, and nowhere else.

Since phase 5, records from OpenMaterica (dev3), and the part resolver's inputs:

- **Where a value comes from**, in this order: a declaration in the experiment, which takes the place of anything
  else and the report says of what; a medium's record (`Medium(material=Record(...))`), read at the experiment's
  `conditions=`; a body's parts, through the part resolver (`mime.things`), whose own inputs, a part's `density` and
  a magnet's `remanence` (`PART_VALUE_KINDS`, at `<body>.parts.<part>.<value>`), come from the part's material or a
  declaration the same way. `Resolver` holds that order; each value says its source, its reason and its inputs.
- **A read** (`_read.py`) sends OpenMaterica's plain read: the record by its id, version and content hash, the
  field (`RecordField`: `bulk.density`, `fluid.dynamic_viscosity`, `magnetic.remanence`), the experiment's
  conditions with their tolerances, the tokens the value's meaning fixes (a density is `apparent`) and those the
  material requires. MIME passes the verdict and its reasons through unchanged; what the build does with them,
  until profiles land in a1 (a1's will refine it): `met` builds; `not_met` is refused; `unknown` is refused unless
  the material's `Acceptance` for that field covers every reason given, by code or `code:condition`; `no_pick` is
  refused even then, since no one value exists until OpenMaterica reads picks; an interval is read as its bounds
  when the material declares a `Nominal`, which must lie inside them; a table is OpenMaterica's value at the
  condition, and outside its range the material's `Extrapolation` applies, `constant` (the end value) or `linear`
  (the end segment continued, computed here from the table the read returns).
- **Records as pinned files** (`_records.py`): `record_file(path, sha256=...)` or `packaged_record(name)` (copies
  MIME ships in `mime.data`). The bytes are checked against the pin before anything reads them, and handed to
  OpenMaterica's own loader as read, so nothing changes between the pin and the read. The saved form writes each file
  under `records/`; the document holds the reference (file, SHA-256, id, version, content hash) and never a value.

Labelled choices, for a1 to revisit: an acceptance names the reasons it covers rather than accepting a whole read;
an interval's value is checked against the bounds here, not through OpenMaterica's `nominal` acceptance (which
reads `want: value` and so returns no bounds to record); the lock records OpenMaterica's version in its environment
record, and every read as sent and answered in its resolved record, so a load under a reader that answers
differently is refused.

Later: per-region values once OpenMaterica's scenes land; materials read from bindings on the USD stage; writing
OpenMaterica's plain scene manifest through its region writer; mixing rules; `MaterialAt`.

Phases and releases are explained in the [package manifest](../README.md#status).
