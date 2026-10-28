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

## The bundle

A **bundle** is a saved folder as one file, to hand to someone else (`mime pack <folder>`, `mime.saved.pack`). It is a
zip with nothing compressed, named `<folder>.mimebundle`:

- `manifest.json` first: the format (with its version in the name), the experiment's name, every file's sha256 and size,
  the asset references of the lock, and the **bundle hash**, the sha256 of the rest of the manifest. It names the bundle's
  content whatever the zip around it.
- The folder's files under `experiment/`: the document, the lock, the reference outputs, the scene and its layers, the
  records and the descriptions.

**Deterministic.** The same folder makes the same bytes: entries sorted, a fixed timestamp and mode, no compression
(a compressor's output differs between zlib builds). A test packs twice, and from a copy with other modification times
and modes, and compares bytes.

**What it will not carry.** `pack` first loads the folder (`Experiment.load`, so a folder that does not load is refused
with the load's reasons) and then reads every file's bytes, refusing, with every reason at once:
- a **path of this machine**: an absolute path under `/home`, `/Users`, `/root`, `/tmp`, `/var`, `/mnt`, `/opt`, `/srv`,
  `/private` or `/runner`, a Windows user folder, or the current user's home directory, wherever it is (the rule of the
  documentation generator's pages, which a test holds the pattern to) as `bundle.machine-path`;
- a **credential**: the exact value of any token MIME holds, in the environment or the file, as `bundle.credential`
  (the message names the file, never the token);
- a **link or a special file** in the folder (`bundle.special-file`);
- a **file over the size limit** (`bundle.file-too-large`).

**The size limit** is fifty megabytes a file (`mime.saved.INLINE_LIMIT_BYTES`), a choice and not a measurement: a bundle
is read and hashed member by member, and one a person downloads, checks and keeps beside their experiment should stay
small. A larger file is an **asset**: the lock's `assets` records it with its sha256, size, licence and locations, and
it travels as that reference, never copied, and is fetched on request ([`mime.assets`](../assets/README.md)). A large
file in the folder that the lock references is left out of the bundle; one it does not is refused.

**Reading one back.** `mime pull <doi | address | file>` (`mime.saved.pull`) gets the bundle through the assets
resolver: a bundle is a content-addressed file like an asset, stored in the same cache under its hash, subject to the
same offline switch and refusal codes, and a DOI needs no account (pooch resolves the record and holds the download to
the repository's checksum). A bundle given as a file on this machine is read where it is. Then, before anything is
trusted:
- every member's name is checked (relative, plain parts, no `..`, no drive letter, no backslash), and every member's
  kind: a link, a directory, a device, a pipe, a doubled or an encrypted member, or one compressed, is refused as
  `bundle.unsafe`;
- the manifest is checked for its shape and its own hash, and the members against it: one unlisted, one missing, or one
  whose size or sha256 differs is refused as `bundle.changed`;
- files are written by this code's own opens and never by an extraction, into a hidden folder beside the new one,
  which is renamed to it only when every hash matched and the folder reads as a saved experiment (the lock, the
  reference outputs, the stage, the layers and the records, each held to its hash). The new folder must not exist (or
  be empty); a refused pull leaves nothing.

It then lists the packages and versions the lock names that are not installed, with the command to install them. **It
never installs and never runs anything.**

**What a hash in a bundle can and cannot vouch for.** The manifest's hashes make a bundle self-consistent: a damaged or
truncated download, a member swapped, a file added, are found. They do not make it authentic: someone who rewrites the
files and recomputes the manifest makes a bundle that passes. Authenticity is the DOI (a repository's record cannot be
changed after publishing), the repository's checksum that pooch holds a DOI download to, or a hash you were given
out of band. After a pull, `mime verify` reruns the experiment against its reference outputs.

## Sharing an experiment

```
mime pack ./fl9_water/                    # fl9_water.mimebundle, and its sha256
# publish the file: a Zenodo record, a release, a shared drive (mime push, once built, uploads it)

mime pull 10.5281/zenodo.1234567 --into ./fl9_water/   # or an address, or the file
mime verify ./fl9_water/
```

`pull` ends with a summary in names and sizes: whether the manifest, scene and records are right, which assets were
cached or downloaded (and the command to fetch those left), and the packages to install first. Install them, then
`mime verify`. "The same result" is as everywhere in this form: bit for bit in the environment the lock was written in,
and within its tolerances in another.

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

Since phase 5 the document holds the experiment's `conditions`, each value with its tolerance; `materials`, each a
reference to its record (the file below the folder, its SHA-256, the record's id, version and content hash) with the
experiment's own terms for reading it (required tokens, acceptances, nominal values, extrapolations, each with its
reason), never a value from the record; a medium's material by name; and a body's parts (each solid by name and
version, its placement, material and host). The lock's resolved record holds `materials`: the conditions, each
record's pins, every read as sent and answered (OpenMaterica's status and reasons, and what the build did), and each
part as resolved; each value records its inputs (`details`). `save` writes the record files under `records/`, the
lock pins them (`records`), and a load refuses a file changed by so much as a byte (`load.record-changed`).

Since phase 7a2 the document holds an apparatus's chain (a list of segments, in order, each by name and version
with its settings), its parts with their mounts and its magnets; a file an entry reads (a robot's description) is
written by its path below the folder, `descriptions/<sha256 prefix>-<file>` (so that two files of one name keep their
own copies), with its sha256, and `save` writes it there. The lock's resolved record holds each segment's description
as read (`descriptions`: every element by group, what was read past and why, the links read as massless).

Since the sharing piece (after the assets) `mime pack` and `mime pull` make and read the bundle, ahead of `mime push`.

Phases and releases are explained in the [package manifest](../README.md#status).
