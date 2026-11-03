# Materials: how a value reaches an entry, and what declared means

A law or a dynamics never reads a record. It names the **values** it needs, and the build brings each one to it in SI,
from one of three places, saying which. This note holds the definitions that the entries' "what it reads" tables and the
build report use.

## A value, and its three sources

A **value** is a number a law reads of a thing: a body's `mass`, `inertia`, `buoyant_volume`, `centre_of_buoyancy` and
`magnetic_moment`, a medium's `density` and `viscosity`, and the values of a body's or an apparatus's **parts** (a part's
`density`, a magnet part's `remanence`, an apparatus part's `mass`). `mime.materials.VALUE_KINDS` lists them, each with its
dimension, its shape and what it means. A value is read from one place, in this order:

1. **A declaration** in the experiment, `exp.declare("umr.mass", kg(...), reason="...")`
   (`umr.parts.shell.density` for a part's). It takes the place of anything else, and the build report says of what.
2. **A record**: a medium's (`Medium(material=Record(...))`) or a part's material, read at the experiment's
   `conditions=`. A record is a pinned file (`record_file(path, sha256=...)`, or `packaged_record(name)` for a copy MIME
   ships): its bytes are checked against the pin before anything reads it, and the saved form carries the file and each
   read as it was sent and answered.
3. **A body's parts**: its mass, inertia, buoyant volume, centre of buoyancy and magnetic moment are derived from the
   parts' solids, placements and densities (and a magnet's remanence), with one line per input. A value that cannot be
   derived says which input is missing.

The parts of a chain's segment are the apparatus's (how they mount: [the multibody note](multibody.md)), and the world they sit in is [the scene](scene.md).

A setting (a geometry's size, a command) is not a value: its source goes on its quantity, `mm(..., reason="...")`, or on
`exp.source(path, reason="...")`.

## From the record to the node

An entry declares the values it reads, `values = {"fluid_density": Value("medium", "density"), ...}`, by role. The build
resolves each once, in SI, with its source and its reason, and hands it to the entry in `ctx.values`
(`ctx.scalar_value("fluid_density")`); the entry passes it to its node's constructor, where it becomes a parameter leaf of
the graph, so a study can move it. The node reads a number and nothing of where it came from; the source stays in the build
report and the saved form.

A record is read through OpenMaterica, which answers with a verdict and its reasons, and MIME passes them through
unchanged. `met` builds. `not_met` is refused in every profile (`record.not-met`), unless the material declares how to
extrapolate a read that falls outside its range. `unknown` builds only where each of its reasons is covered, by the material's
`Acceptance` or by a waiver the experiment's profile generates for that class of reason, and is refused otherwise
(`record.unknown`, or `record.no-pick` where the record holds several values that the conditions do not choose between). Which profile does what is [the profiles note](profiles.md).

## Declared, derived, read, and fitted

Each resolved value carries a label the report and the saved form print:

- **Declared in the experiment**: a number the experiment states, with a reason that is a text, a `chosen(...)` choice or
  `pending()`. MIME checks its units, shape and finiteness, never its truth. Exploratory builds with a pending reason and
  stamps every output; research refuses it by name (`value.reason-pending`).
- **Derived from its parts**: arithmetic on the parts' values, each input on a line of its own.
- **Record**: the record's value at the stated conditions, labelled with the record it was read from.

**Fitted** is a different thing from all three: a number that came out of fitting a model to data. None of the reasons a
declaration can carry says so, and nothing in the value path fits or re-fits a number. A number fitted elsewhere is
therefore declared here, and its reason is where the fit is named; a record's field that was itself measured or fitted is
the record's provenance, passed through and not judged. A number in a declaration is a declaration until something fits it,
and the report prints it as "declared in the experiment".
