# `mime.saved`

**The saved form: what makes an experiment reproducible by someone else.**

## What is here

- **The document**, `experiment.mime.yaml`. It records what the experiment declares: the schema version, the MIME
  version and the experiment's name; the run settings (`dt`, the seed); the world frame (until the scene holds it);
  every thing with its kind, its dynamics by registered name and version, its settings and initial state; every
  interaction and observation by registered name and version, with its settings and its bindings by name. An alias
  such as `ToyDrag.Quadratic()` saves as its family with the rung among its settings. Settings left at their defaults
  are not written; the resolved record holds them. Slots for material references, the scene and the controller stay
  empty until those land. It never holds code, and never holds copies of a record's values.
- **The lock**, `mime.lock` (JSON). The environment (MIME, MADDENING, OpenMaterica, JAX, jaxlib and Python versions,
  the platform, the device, the precision, the PRNG); every entry used, with the distribution it came from and that
  distribution's version; the derived seeds; the hashes; the resolved record; the derived graph; and the record of
  the reference outputs.
- **The resolved record**: every setting with its default, each body's initial state in SI, the nodes each entry
  built in the canonical order, the registrations, reader slots, coupling groups, and each entry's declared gradient
  status.
- **The reference outputs**, `reference.npz`: every field of every node over the first steps, and every static
  array's contents.

- **The scene**, `scene/scene.usda`, and the asset layers it references, under `scene/layers/` with their bytes as the
  build read them.

The file names are provisional until the experiment folder.

Both files are described by closed TypedDicts (`ExperimentDocument`, `Lock`). A TypedDict does nothing at run time,
so the loader checks what it reads against the same annotations, refusing an unknown key, a missing key or a value of
another type by its path; a test keeps the two in step.

## Loading

`Experiment.load(folder)`:

1. reads the document with PyYAML's safe loader (a Python tag, or a key given twice, is refused) and the lock with
   `json`, so reading runs no code, and holds the stage and every asset layer to the hash the lock records (a changed
   asset is refused as `load.asset-changed`, a stage edited by hand as `load.stage-changed`);
2. finds each entry among installed packages only, through the `mime.entries` entry point of its namespace, and never
   imports a module the document names or installs anything. A missing entry, or another version, is refused with
   the line to install, taken from the lock;
3. rebuilds, and makes three checks:
   - **the input hash**, exactly: the document, the resolved record and the entries' names and versions. A difference
     is refused, naming the item that differs;
   - **the environment**, by report: every difference from the one the lock was written in, never a refusal;
   - **the derived graph**: within the environment the lock was written in it must match hash for hash (a factory
     that builds something else under the same version is refused); in another, its values and static arrays are
     compared within the lock's tolerance and reported. The rebuilt stage is held to the lock's hash the same way.

`BuiltExperiment.verify()` and `mime verify <folder>` rerun the reference steps from the initial state: bit for bit
within the environment the lock was written in, and within the lock's tolerance in another. **"The same result" means
bit for bit within one environment; across environments the checks report differences within tolerances and claim no
bits.**

A package that registers entries declares, for each namespace, `[project.entry-points."mime.entries"]
<namespace> = "<module whose import registers them>"`. `save` refuses an entry a load elsewhere could not find this
way, and a setting the document cannot hold.

## How it stays complete

A CI test walks the registry in a fresh process and requires, for every registered entry and every kind of thing, a
round-trip case: build, save, load in a fresh process, rebuild, then the same derived hash and bitwise-identical
first steps. An entry added without one fails CI (`tests/saved/`).

## Status

Landed in dev1's phase 3, on the toy entries.

Since phase 4b the document holds a structure's geometry kind (by name and version, with its settings), the region a
medium fills, each command's drive, typed paths as their strings (`rotation_axis: tube.axis`), and `declared`: every
value declared in the experiment, in SI with its unit and its reason. The lock's resolved record lists each value read
with its source ("declared in the experiment"), reason and readers, the geometry kinds, the fills and the gathered
node order; its entries pin the geometry kinds and drives with their distributions. Material references (phase 5),
the scene's root, layers and variants (phase 6) and the controller (phase 7) join the document as they land; the
experiment folder and `pack` come at 0.3.0.

Since phase 6 the document holds the scene (`up`, `units`, and each reference's prim, its file below the folder and
its SHA-256) in place of the stand-in frame, gravity's magnitude, every thing's prim path beside its name, and a
body's magnet roles. The lock holds the stage (its file and hash, and each layer's hash), the resolved scene (up,
units, every thing's prim, the bound things, each magnet's direction in its part's frame and its body's, and where
it came from), and usd-core's version in the environment.

Phases and releases are explained in the [package manifest](../README.md#status).
