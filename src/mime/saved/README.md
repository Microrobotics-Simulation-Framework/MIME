# `mime.saved`

**The saved form: what makes an experiment reproducible by someone else.**

## What is here

- **The document**, `experiment.mime.yaml`. It records what the experiment declares: the schema version, the MIME
  version and the experiment's name; the run settings (`dt`, the seed); the world frame (until the scene holds it);
  every thing with its kind, its dynamics by registered name and version, its settings and initial state; every
  interaction and observation by registered name and version, with its settings and its bindings by name. An alias
  such as `ToyDrag.Quadratic()` saves as its family with the rung among its settings. Settings left at their defaults
  are not written; the resolved record holds them. Slots for material references, the scene and the controller stay
  empty until those land. It never holds code, and never holds copies of a record's values. It holds what may be
  shared (see [Terms and tiers](#terms-and-tiers)): the experiment's own licence, each custom licence it names, each
  third-party file's terms and each declared permission, each only where something is declared.
- **The lock**, `mime.lock` (JSON). The environment (MIME, MADDENING, OpenMaterica, JAX, jaxlib and Python versions,
  the platform, the device, the precision, the PRNG, and what no pin holds: the GPU's device kind with its driver, CUDA
  and cuDNN as JAX reports them, the CPU's model and the hash of its flags, the matmul precision, `XLA_FLAGS` and
  whether they turn XLA's deterministic GPU mode on, the runner's own record, from `MIME_RUNNER`, which MIME stores
  and compares and never reads, and the runner's information, from `MIME_RUNNER_INFO`, which it stores and never
  compares); every entry used, with the distribution it came from and that
  distribution's version; the derived seeds; the hashes; the resolved record; the derived graph; the record of
  the reference outputs; and, where there are any, the remote assets (`assets`) and the files that travel as
  references, each with the asset that holds its bytes (`referenced_files`), both outside the input hash.
- **The resolved record**: every setting with its default, each body's initial state in SI, the nodes each entry
  built in the canonical order, the registrations, reader slots, coupling groups, and each entry's declared gradient
  status: the attached rung's, where the entry's rungs declare their own.
- **The reference outputs**, `reference.npz`: every field of every node over the first steps, and every static
  array's contents.

- **The scene**, `scene/scene.usda`, and the asset layers it references, under `scene/layers/` with their bytes as the
  build read them.

The file names are provisional until the experiment folder.

Both files are described by closed TypedDicts (`ExperimentDocument`, `Lock`). A TypedDict does nothing at run time,
so the loader checks what it reads against the same annotations, refusing an unknown key, a missing key or a value of
another type by its path; a test keeps the two in step.

## What a saved folder never holds

A saved folder holds what reproduces the experiment and nothing else, never a result and never presentation:
`layout_problem` refuses any other path in it, and the schema any other key. A run's results, a recording, a comparison
with another rung, a measured track and a look-dev or preview layer are overlays, and an overlay lives in a run or study
folder, or in the asset cache (`mime.viz`), never in the saved folder it is over.

**Every layer MIME composes resolves inside its own folder**: a sublayer or a reference is a path below the folder that
holds the layer, with no path out of it, so that a folder resolves again after it is moved, copied or published. A preview
baked from an asset (a GLB, a decimated mesh) is a derived file: it carries that asset's licence tier (design #198) and
is never pinned by its bytes in an experiment's lock (design #183).

## Loading

`Experiment.load(folder)`:

1. reads the document with PyYAML's safe loader (a Python tag, or a key given twice, is refused) and the lock with
   `json`, so reading runs no code, and holds the stage and every asset layer to the hash the lock records (a changed
   asset is refused as `load.asset-changed`, a stage edited by hand as `load.stage-changed`);
2. refuses a form written by a newer MIME before judging it: a document or a lock whose schema is `mime.document/N` or
   `mime.lock/N` above the newest this MIME reads, a bundle of a later `mime.bundle/N` format and a `sharing.yaml` of a
   later `mime.sharing/N` are each one refusal, `load.newer-form`, naming the file, the version found and the versions
   this MIME reads, with the advice to upgrade `mime-engine`. An id of another family, and a version this MIME reads,
   are not that refusal; the schema check judges them as it did. (A run folder's `outputs.json` of a later version is
   refused as `result.format`, with the same advice.)
3. finds each entry among installed packages only, through the `mime.entries` entry point of its namespace, and never
   imports a module the document names or installs anything. A missing entry, or another version, is refused with
   the line to install, taken from the lock;
4. rebuilds, and makes three checks:
   - **the input hash**, exactly: the document, the resolved record and the entries' names and versions. A difference
     is refused, naming the item that differs;
   - **the environment**, by report: every difference from the one the lock was written in, never a refusal;
   - **the derived graph**: within the environment the lock was written in it must match hash for hash (a factory
     that builds something else under the same version is refused); in another, its values and static arrays are
     compared within the lock's tolerance and reported. The rebuilt stage is held to the lock's hash the same way.

`BuiltExperiment.verify()` and `mime verify <folder>` rerun the reference steps from the initial state: bit for bit
within the environment the lock was written in, and within the lock's tolerance in another. **"The same result" means
bit for bit within one environment; across environments the checks report differences within tolerances and claim no
bits.** **"The same environment" means equal on every field of the lock's environment but the runner's information**: the
versions, the platform, the device and the precision, and the GPU and its driver and libraries, the CPU, the matmul
precision, XLA's flags and deterministic mode, and the runner's record. A difference in any one of them, in the lock or
in the process that loads it, makes the load another environment. A field of the GPU or the CPU that either side could
not read is never equal to anything, `None` included. The runner's information (a workspace's image ID, a client's
version) is for the reader and is never compared.

A package that registers entries declares, for each namespace, `[project.entry-points."mime.entries"]
<namespace> = "<module whose import registers them>"`. `save` refuses an entry a load elsewhere could not find this
way, and a setting the document cannot hold.

## Terms and tiers

**Who may hold which bytes** is the document's to say (designs #198 and #199), and `pack` obeys it. The words are in
[`mime.terms`](../terms.py):

- **A licence is an SPDX licence expression** in its canonical spelling, checked by the packaging standards' own parser;
  a licence of the author's own is a `CustomLicence`, a `LicenseRef-` id with its title and an https link. Free text is
  refused, and so is another spelling of a real licence (the canonical one is named). `NOASSERTION` says nobody stated
  one, and reads as unknown. An asset reference's licence is held to the same rule (`asset.reference`). An id the
  installed `packaging` does not list (one SPDX added after its release) is refused where a script declares it, naming
  the update (`pip install -U packaging`); read from a saved folder it is read as written and judged unknown, the load
  and `mime verify` go on and their report says so, and `pack` and the card refuse it (`terms.unlisted-licence`),
  since they would state a verdict this packaging cannot give. A document whose terms were edited by hand into a form
  `save` never writes is refused at load by the code `save` gives it.
- **The experiment's own licence**, `exp.licence(...)`, covers its own files: those MIME writes and those its author
  brings with no terms declared. An experiment may declare none: saving and `pack` never refuse that, and the NOTICE,
  the card and `mime pack` say so; `mime push` will not publish one until it declares a licence. Its own files are
  always carried (its author packs them), and their verdict is the licence's: permitted only for a licence on MIME's
  lists, otherwise unknown (or not permitted, for a custom licence that says so), in the tier **carried**; with no
  licence declared they are **unknown**, "the author's own, no licence declared: a reuser is granted no licence", never
  permitted. A folder of a form before licences declares none, and `mime pack` adds a line where it holds layers or
  descriptions: a third party's file in it is carried as the author's own until its terms are declared.
- **A third-party file is declared with its terms** where it is declared, `asset(path, terms=Terms(...))`: its licence,
  the credit its licence asks for, its role (results depend on it, or it only draws the scene) and its origin, the
  locations of its exact bytes. The document records them by the file's path below the folder, with the sha256 they
  were declared for. Terms bind by path, so two copies of one file's bytes (a mesh copied for a second prim) are
  declared with the same terms or refused (`terms.file`, naming both), and `pack` never carries a copy of bytes that
  travel as a reference, whatever terms it has (`bundle.referenced-copy`, naming both).
- **A permission** from the rights holder, `exp.permission(Permission(...))`: the grantor, the grantee, the files by
  sha256, the scope (redistribute, modify, commercial use), the date and the evidence (a public statement's link, or a
  written permission held privately, which is never published). MIME checks its form and binds it to the bytes, so a
  changed file inherits nothing; it never vets its truth.

Each file's **verdict** is in OpenMaterica's words for a source's redistribution status: permitted, not permitted or
unknown. A licence on MIME's open list (public-domain dedication, attribution, the permissive software licences) or
conditional list (the Creative Commons non-commercial, share-alike and no-derivatives licences), or a permission that
grants redistribution, makes a file permitted; every other licence is unknown until a person declares a permission, and
a custom licence may say not permitted. A record file is judged by its record's own licence of its values, as given.
The **tier** says where the bytes live: open and conditional files travel inside a bundle, a conditional one's
conditions said in its NOTICE and on its card; **a file that may not be redistributed travels as a reference to its
origin, whatever its size, and is never packed**, nor its bytes at another path. So does a file over the size limit
whose terms give an origin. The lock names each (`referenced_files`) and the asset that holds its bytes (one asset for
one sha256), which `mime assets fetch <folder>` fetches and places; since the lock is outside the hashed record,
`pack` holds each such asset to the file's terms (their licence, and their origin as its locations, primary first),
and an asset MIME ships to the licence it ships it under (`bundle.lock-terms`). An asset the lock names that is neither
has no source MIME knows, and its licence is the lock's word only.

`save` refuses what is incomplete, every reason at once, by its code: a licence that is not an expression
(`terms.licence`), a custom licence without its title or link (`terms.custom-licence`), terms for a file the experiment
does not hold, for other bytes, or without credit or role (`terms.file`), an origin that is not a location
(`terms.origin`), and a permission lacking a part, or bound to bytes no file has (`terms.permission`).

**The NOTICE and the card.** `mime card <folder> --notice` (`mime.saved.notice`) writes the NOTICE: the experiment's
licence, every file the folder names with its sha256, size, owner, licence, verdict and where its bytes live, a third
party's credit, role, origin and permissions in #198's words, each custom licence with its link, the text of each
licence that asks to travel with every copy (MIT and the BSD licences) under its file's credit, and the availability
statement: "reproducible exactly from this bundle alone" only where the bundle carries every file, and otherwise each
file it does not carry, with why, its licence, its role and where it is available from. `mime card <folder>`
(`mime.saved.card_text`) writes the card, as `mime push` will write a record's description: every condition each
file's licence or permission carries (non-commercial, share-alike, no-derivatives) is flagged, and "what reproduces"
says what `mime verify <folder>` would check (the card runs no check). Both are generated and the same bytes each
time. **Credit** (the creators with their ORCIDs, the
contributors and their roles) lives in `sharing.yaml` beside the saved form, which `mime push <folder> --to <service>
--init` writes (see [`mime.push`](../push/README.md)): the card names its creators and contributors, and where the folder
holds none says so where the creators go; each third-party file's credit is under its licences either way. `pack`
neither carries nor refuses `sharing.yaml`, the one file of a folder that is neither the saved form's nor a stray.

## The bundle

A **bundle** is a saved folder as one file, to hand to someone else (`mime pack <folder>`, `mime.saved.pack`). It is a
zip with nothing compressed, named `<folder>.mimebundle`:

- `manifest.json` first: the format (with its version in the name), the experiment's name, every carried file's sha256
  and size, each file that travels as a reference with the asset that holds it, the NOTICE's sha256 and size, the asset
  references of the lock, and the **bundle hash**, the sha256 of the rest of the manifest. It names the bundle's
  content whatever the zip around it.
- The `NOTICE` (see [Terms and tiers](#terms-and-tiers)), which `unpack` checks and keeps and writes into no folder.
- The folder's files under `experiment/`: the document, the lock, the reference outputs, the scene and its layers, the
  records and the descriptions, less those that travel as references.

The format before references (its manifest's format ends in one) still unpacks: every file carried, no NOTICE. A bundle
of a format this MIME does not read (a later one) is refused before any other check, by `unpack` and `pull` alike,
with the advice to install the MIME that packed it, so that a member a later format adds is never judged as a hostile
bundle's. A MIME from before this format (a development checkout; no release holds format one alone) instead refuses
one as not a bundle `mime pack` wrote, except `mime pull` without `--into`, which reads the manifest first.

**Deterministic.** The same folder makes the same bytes: entries sorted, a fixed timestamp and mode, no compression
(a compressor's output differs between zlib builds). A test packs twice, and from a copy with other modification times
and modes, and compares bytes.

**What it carries, and what it will not.** `pack` first loads the folder (`Experiment.load`, so a folder that does not
load is refused with the load's reasons) and then carries **the files the saved form names and nothing else**:
- the document (`experiment.mime.yaml`), the lock (`mime.lock`), the reference outputs (`reference.npz`), the stage
  (`scene/scene.usda`) and the layers it references (`scene/layers/<name>`), the records the lock pins
  (`records/<name>`) and the description files the document names (`descriptions/<name>`): the layout above, one level
  down where it has one, each file one the lock or the document lists;
- no part of any path starts with a dot, and no name is one a filesystem treats specially (a Windows device name such as
  `CON` or `NUL`, a name with `: < > " | ? *`, one that ends in a dot or a space, one longer than the name limit most filesystems keep, a control or
  bidirectional character); and no two names are one on a case-insensitive filesystem (`A.json` and `a.json`, or the
  same accented letter written two ways), so that a bundle unpacks to the same folder on Linux, macOS and Windows.

A file that is not one of these is refused as `bundle.stray-file` (an `.env`, a `.git` directory, an editor's file, a
note), and two names that collide as `bundle.name-collision`: the bundle carries the folder a save wrote, not whatever
else was lying in it. This is the control that matters most, because what is searched for below can only be searched for
in the forms it is known in. `pack` then reads every file's bytes, refusing, with every reason at once:
- a **path of this machine** (`bundle.machine-path`): an absolute path under `/home`, `/Users`, `/root`, `/tmp`,
  `/var`, `/mnt`, `/opt`, `/srv`, `/private` or `/runner`, a Windows user folder, or the current user's home directory,
  wherever it is (the rule of the documentation generator's pages, which a test holds the pattern to); and beyond that
  rule, because a path in a record hides from it: the same roots written as `file:///home/...` (the third slash hides the
  root from the rule), `/Volumes/`, `/media/` and `/run/user/`, any Windows drive (`D:\data\x`, as JSON or YAML
  doubles the backslash too) and a UNC share (`\\server\share`), these two in files that are text only (valid Unicode text,
  no NUL), since random bytes in a binary array spell them often enough to refuse a clean one. **Not flagged, on purpose:** `/data`, `/scratch` and
  `/work`, which are as likely a rooted name inside a record as a cluster's disk, and `~/` and `$HOME/`, which are not
  absolute. A drive pattern can also match a short label written `a:/b`; the remedy is the same as for a path;
- a **credential** (`bundle.credential`): any token MIME holds, in the environment or the file, as written, percent-
  encoded, JSON-escaped, in base64 or hex or as two-byte Unicode text, and inside a gzip stream or a compressed zip member (the message
  names the file, never the token). **It cannot catch every encoding**: a token run through a cipher, reversed, split
  across lines or upper-cased passes, which is why the whitelist above comes first;
- a **file it cannot search** (`bundle.unscannable`): a compressed file that expands past
  `mime.saved.EXPANSION_LIMIT_BYTES`, is nested more than four deep, or cannot be read inside;
- a **link or a special file** in the folder (`bundle.special-file`);
- a **file over the size limit** that travels as no reference (`bundle.file-too-large`);
- a **file that may not be redistributed** whose terms give no origin (`bundle.no-location`), and a lock whose
  references are not the ones the document's terms make, or whose assets state another licence or other locations
  than their source (`bundle.lock-terms`): `pack` judges every file's terms itself, and holds the lock to its
  judgement;
- a **copy of a referenced file's bytes** that would be carried, at another path and under whatever terms
  (`bundle.referenced-copy`);
- a **licence naming an id the installed `packaging` does not list** (`terms.unlisted-licence`).

**The size limit** is fifty megabytes a file (`mime.saved.INLINE_LIMIT_BYTES`), a choice and not a measurement: a bundle
is read and hashed member by member, and one a person downloads, checks and keeps beside their experiment should stay
small. A larger file is an **asset**: the lock's `assets` records it with its sha256, size, licence and locations, and
it travels as that reference, never copied, and is fetched on request ([`mime.assets`](../assets/README.md)). A large
file of the folder travels as a reference only where its terms give an origin: a save then writes it into the lock
(`referenced_files`), and a lock that lists one whose terms give none is refused (`bundle.lock-terms`), as one that
does not list it is (`bundle.file-too-large`).

**Reading one back.** `mime pull <doi | address | file>` (`mime.saved.pull`) gets the bundle through the assets
resolver: a bundle is a content-addressed file like an asset, stored in the same cache under its hash, subject to the
same offline switch and refusal codes, and a DOI needs no account (pooch resolves the record and holds the download to
the repository's checksum). A bundle given as a file on this machine is read where it is. **The destination is checked
first**, before anything is downloaded. **The download is held to a size**: the size the record states for the file, or,
where nothing states one (a bundle at an address), `mime.assets.RETRIEVE_LIMIT_BYTES` (half a gigabyte, a choice and not a
measurement), raised by `--max-bundle-bytes`; it stops where it passes the size and refuses as `asset.size-exceeded`.
Then, before anything is trusted:
- every member's name is checked against **the saved form's own layout** (above): a member that is not one of its files,
  any path part starting with a dot, a name a filesystem treats specially, two names that collide in case, or a name
  with a backslash, a drive letter, `..` or an absolute path is refused as `bundle.unsafe`, and so is every member's kind:
  a link, a directory, a device, a pipe, a doubled or an encrypted member, or one compressed;
- the manifest is checked for its shape (a manifest nested too deeply is a refusal, not a traceback) and its own hash, and
  the members against it: one unlisted, one missing, or one whose size or sha256 differs is refused as `bundle.changed`;
- files are written by this code's own opens and never by an extraction, into a hidden folder beside the new one,
  which is renamed to it only when every hash matched and the folder reads as a saved experiment (the lock, the
  reference outputs, the stage, the layers and the records, each held to its hash), and **its files are exactly the ones
  the lock and the document name**. The lock is read for what it may print: every distribution name must be a valid
  Python distribution name and every version a valid Python version (the packaging standards' own rules), and every file it names must be in the layout, or the bundle is refused
  (`load.lock-shape`). The new folder must not exist (or be empty); a refused pull leaves nothing. (A process killed
  outright is not a refusal: it can leave a hidden `.<name>.partial-<hex>` folder beside the target, which is safe to delete.)
- each file that travels as a reference must be one the layout holds, carried by no member, and named in the lock with
  an asset of the same sha256 and size, its bytes carried by no member at another path and its name no member's in
  another case (`bundle.manifest`, `bundle.unsafe`); it is placed from the asset cache where it is cached, copied and
  hashed, never over another file, the folder judged by its real path (a folder reached through a link, as macOS's
  `/tmp` and `/var` are, is placed in like any other). `pull` fetches the assets first, unless `--no-assets`, and says
  in a line of its summary what it placed and what it did not; a file still missing is left for `mime assets fetch
  <folder>`, which exits two while any is unplaced (saying why: not cached, another file in its place), and a load of
  the folder until then is refused as `asset.missing` with that command.

It then lists the packages and versions the lock names that are not installed, with the command to install them, each
word quoted for a shell. The packages are the bundle author's choice, which the summary says. **It never installs and
never runs anything.** Everything a bundle's text puts on the screen goes out with its control, escape and separator
characters written as text (`\x1b`), so that a name cannot clear the screen, set the window title or forge a line of the
summary.

**Pulling from a stranger.** A bundle comes from whoever wrote it, and the checks above are what lets you read it without
trusting it: the layout and name checks, the hashes, the lock's names and the size caps. Three things are still theirs.
**The assets:** a pulled lock names the hosts its assets come from (any `https://` address, and `http://` to this
machine); each is held to the declared size and sha256, but the request itself goes to a host the author chose.
`--no-assets` unpacks without contacting any, and `mime assets fetch <folder>` fetches them later. **The packages:** the
command lists what the author used, which you decide whether to install. **A bundle at an address** has no checksum at
all before the manifest's own, which proves self-consistency and not origin (next paragraph). Read the folder before you
`mime verify` it: verifying runs the experiment, with the code you have installed.

**What a hash in a bundle can and cannot vouch for.** The manifest's hashes make a bundle self-consistent: a damaged or
truncated download, a member swapped, a file added, are found. They do not make it authentic: someone who rewrites the
files and recomputes the manifest makes a bundle that passes. Authenticity is the DOI (a repository's record cannot be
changed after publishing), the repository's checksum that pooch holds a DOI download to, or a hash you were given
out of band. After a pull, `mime verify` reruns the experiment against its reference outputs.

## Sharing an experiment

```
mime pack ./fl9_water/                    # fl9_water.mimebundle, and its sha256
# or a draft record on Zenodo: mime push ./fl9_water/ --to zenodo-sandbox (publishing comes later)

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
Since phase 7e a stage edited after its lock was written is judged against the stage the rebuild writes from the
same document: an edit confined to its derived prims (a kind's drawn mesh, a region's primitive), output only and
written again by every build, is reported in the load report (`derived_prims_edited`), and any other edit is refused
(`load.stage-changed`), as before. Profiles do not judge stage edits in dev1 (exploratory and research report a
derived prims' edit alike); a1's `strict` may refuse the first too, at that seam (`stage_change_refusals`).

Since phase 7l-a the saved form has been in its second version (the third is below), as the document's and the lock's
schemas say. The document holds the `profile`, `sources` (each `exp.source`), a quantity's reason beside its value (`{value, unit, reason}`), and a reason
keyed by its kind (a text as it is, `{pending: true}`, `{chosen: ...}`); gravity is absent where the experiment states
none, standard by convention; a condition's and a nominal's reason are held beside them. The lock's resolved record
holds `profile`, every part of it in a canonical order (by target, then code), so the order a script declares things
in never reaches the hash: each item left pending and each item listed, by its target and codes; every waiver with a
pending basis (its target, the codes it covers, its basis, whether the profile generated it); each default taken by
path with its kind, value and entry, never its rule's or basis's words, so that a reworded basis refuses no folder;
each setting's source; and each declared change of role. A difference in that record under the MIME version that
wrote the lock is MIME's own defect, refused as internal, never as the installed entries'. A folder of the first
version loads (`_versions.py`): its document read with its profile as exploratory, the profile under which every
build it held builds, and its input hash compared in its own form; the load report says the saved form's version and
the profile it is read under.

The saved form is in its third version since the drag's rung was renamed `mrs`: a rung's name is in the document and in
the resolved record, the drag node's type is in the derived graph, and all three are hashed. The same version writes
the possessive in the sources of declared values plainly (`body umr's mass`, where a quoted name had left `'umr''s`). A
folder of the first or second version loads: its document is read as the current schema (a document that names the
rung by its old name reads as naming `mrs`, and the load report says so), and the rebuilt experiment is written back
into the version its lock was written in (`document_in_form`, `resolved_in_form`, `physics_in_form`) and hashed there,
so an old lock is compared with the form it was hashed in. Saving it again writes the third.

Phases and releases are explained in the [package manifest](../README.md#status).
