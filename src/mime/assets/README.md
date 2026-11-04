# `mime.assets`

**Files referenced by hash, wherever they are kept, and fetched on request. The library knows how to fetch and check a
file, never what or from where.** The wall tables the method of regularised Stokeslets reads are far too large for the
repository or the wheel; a robot's meshes may not be redistributed at all; a lab's scan lives on the lab's own
server. An experiment names each such file by the sha256 of its bytes, never by where it lives, and the file is stored
on the machine only when someone asks.

## The record

One record type serves every asset. An `AssetRef` states what the file is, `name`, `sha256` and `size` (bytes); under
what terms, its `licence`; and where it may be fetched from, `locations`, primary first. Each of these is stated where
it is known, and written only then:

- `credit` (whom the licence asks to be credited), `role` (`results`, or `presentation`: no result depends on it) and
  `notice` (terms to show with the file);
- `kind`, where its truth comes from: `authored` (true by declaration: a CAD mesh, a description), `measured` (a fact
  with an uncertainty: a calibration) or `derived` (a computation: a wall table), with `derivation`, the sha256 of its
  derivation record (below);
- `licences`, each custom licence its licence names (a `LicenseRef-` id with its title, link and what a person says it
  allows, `mime.terms.CustomLicence`);
- `version` and `citation`, its own identity, as a shared part's will need;
- `adapters`, each scheme of a location MIME does not read itself, with the package that provides its adapter.

A record of the five keys every record has writes exactly those five, as every lock of an earlier saved form holds them.
**The tier is a verdict, never written by hand**: `ref.verdict` is `mime.terms.licence_verdict` of the licence and the
record's custom licences (`open`, `conditional`, or `referenced`: the bytes never travel, only the record), and a record
that writes a tier of its own is refused. The AR4's meshes are records whose custom licence says `not_permitted`, so
their tier is `referenced`.

The licence is an SPDX licence expression in its canonical spelling, or `NOASSERTION` where nobody has stated one (a
bundle fetched by address); free text is refused (see [`mime.terms`](../terms.py)). A malformed record is refused whole
as `asset.reference`. In a lock, a manifest or a package it is the record `AssetRecord`, through `to_record()` and
`AssetRef.from_record()`. Because the hash names the file, a new home for it is one more location: no saved
experiment changes, and none breaks.

**A manifest** (`Manifest`, its format `mime.asset_manifest` at its first version) is a named group of ordinary records, with the credit and
notice the group is shown with, and how a description names each of its files (a mesh's path, with its git blob). A
robot's meshes are a manifest beside its description: the AR4's is `examples/descriptions/ar4/visuals.json`. A
manifest's `credit` and `licences` fill each record that states none. The earlier visual-set form
(`mime.visual_set`, its only version) is still read, its hand-written tier dropped and the verdict computed; any other format is
refused as `asset.manifest`, never misread.

## Where records come from, and the pin

In this order: **the experiment's own declaration, or its lock when it is loaded** (the pins); **the manifest beside
the description in use** (`<description stem>.visuals.json`, else `visuals.json`, next to a URDF: `mime.scene`);
**installed packages**, each through a `mime.assets` entry point (its name the source's, its value a package whose
data holds JSON files of records and manifests, as `mime.entries` registers entries from outside MIME); and **MIME's
own package data**, `mime.data/assets/`, for MIME's own assets only (the wall tables). `known()` is every record the
shared sources (the last two) agree on, by name; `reference(name)` is one, under the pins.

- **The pin wins.** A loaded experiment's lock holds each record it read, whole, and its build reads that record
  whatever a shared source holds under the name now (`pinned`); the build report lists each name a shared source now
  holds as another file, as `stale_inputs` says of a derived asset's inputs. A saved stage likewise pins each mesh by
  its manifest's name and its sha256. A record the pin names that is not stored is refused as `asset.missing`, with the
  command that fetches exactly what the lock names.
- **Two shared sources that hold one name as different files, or as the same bytes under different terms** (its
  licence and custom licences, credit, notice, role, kind or derivation), with no pin to settle it, are refused as
  `asset.conflict`, naming both, whichever source is read first; `known()` leaves the name out. A pin settles it, and
  the build report says a source holds the file under other terms. A new location is never a conflict: the same bytes
  under the same terms in two sources are one record with both sources' locations.
- **A file declared by its record**: an experiment states a one-off file's whole record with `mime.scene.asset(
  AssetRef(...))` (a scan used in one study): its sha256 pins it, its licence, credit, role and locations are its
  terms and origin, and the build reads it from the cache, never downloading it. Saved, the lock holds the record whole.

## Locations, adapters, and where this machine gets a file

**The hash is the trust; a location is only a hint.** An adapter (`Adapter`) opens a byte stream for a location, with
the file's size where the store states it, under the credential the resolver hands it, and returns a `Redirect` where
the store answers with another location. Everything else is the resolver's, so an adapter written by hand behaves as
one that wraps a multi-protocol library (a storage plugin wrapping `fsspec` or Apache OpenDAL, say): the
hash check, the size cap, the cache, placed files, the tiers, host binding, timeouts, and retries with backoff.

| Scheme | Adapter | Reads |
|---|---|---|
| `https://` (and `http://` to this machine) | MIME's, on the standard library | a server's file; a forge's raw file at a commit |
| `doi:<doi>/<file>` | MIME's | the DOI's record, resolved to the file's address (pooch's resolution of Zenodo, figshare, Dataverse), each request of the step held to the resolver's timeout, a timeout asked again as any location's |
| `package:<source>/<path>` | MIME's | bytes inside an installed package that declares the `mime.assets` source `<source>` (`mime` for MIME's own), through `importlib.resources`: a record never makes MIME import a module it names |
| `file:///<directory>/` | MIME's | a mirror: a local or mounted directory (a lab's share), its files named by sha256 or by a name the asset may be placed under; regular files only, a link never followed, nothing run |
| any other | a package's, through a `mime.asset_adapters` entry point (its name the scheme, its value `module:object`), or a script's, `register_adapter(adapter)` | what it says |

- **Redirects** are followed by the resolver (as Hugging Face's to its CDN): each target is checked as a location is,
  ten at most, and **a redirect never changes the scheme to a local one**: it leads to `https`; to plain `http` only
  from a location on this machine to this machine; or from an adapter's own scheme to that same scheme (its store's
  business, the credential still bound to its host). A redirect to a `file:` mirror, a `package:` file, a `doi:` or
  another adapter's scheme is refused: a remote server never chooses what MIME reads on this machine, or which store it
  asks with a credential. `mime assets sources` follows the same rule.
- **Credentials.** An adapter that `needs_credential` is handed the token the machine's store holds for the location's
  host (`mime credentials set host:<host>`; the hook is `stored_credential`), and for no other: **a credential is bound
  to its host**, never read from a record, a location string or the locations file, never sent to another host, never
  carried across a redirect to another host, never logged or written into a refusal. **The resolver enforces the last
  whatever an adapter writes**: while a credential it handed is in use, every text it says of the attempt (an
  adapter's error of any type, a redirect's target, a line of `mime assets sources`) and every log record the process
  makes, a logged traceback among them, has the token hidden in every text form a token takes (as written,
  percent-encoded, JSON-escaped, in hex or base64). MIME's own adapters need none: an authenticated store (a token, a
  signed URL, object storage) is a plugin's adapter.
- **TLS.** `https` trusts the system's certificates and certifi's; an institution's server signed by its own CA needs
  its bundle, `MIME_CA_BUNDLE=<.pem>` (or `ca_bundle` in the locations file). The standard proxy variables
  (`HTTPS_PROXY`, `NO_PROXY`) are honoured; this machine is reached directly.
- **Retries and resuming.** A timeout, a cut stream, or a server's answer that it is busy or failing (too many
  requests, a server error) is asked again, three times, with a doubling wait (a wait the store states is waited
  instead, up to half a minute: a `Retry-After` of seconds or an HTTP date; one that is no wait a store means, such as
  `nan`, `inf`, a negative number, more than a day or a word, leaves the doubling wait). An interrupted
  download is resumed where the adapter reads a byte range (`ranged`): for `https`, `Range` with `If-Range` on the
  validator the first answer gave; a server that ignores the range sends the whole file, which starts again. The full
  hash at the end is the check.
- **A user's own adapter** is a first-class path, for storage nobody has written one for: the interface is public,
  registered by a package's entry point or one call; a saved lock names the adapter's package for each such scheme
  (`adapters`), so `mime pull` lists it among the packages to install, never installing it; and a file placed in the
  cache by its sha256 still works without it. `mime.testing.verify_adapter` is the conformance harness an adapter's
  author runs in their own tests, on loopback stores: a file fetched whole, a stream cut at the declared size, a
  missing file and a refused login told apart, and a credential never in an error's text and never sent to another host.

**Institutional storage** fits without a change of format: a mounted share is a mirror path; an internal `https`
server works with its CA bundle; an authenticated one is a plugin's adapter on the credentials hook.

**The locations file.** A lab keeps its own copies and says so once: a TOML file at `MIME_ASSET_LOCATIONS`, else
`locations.toml` in the user's config directory, maps an asset by its sha256 or its name to extra locations, and names
`mirrors` tried for every asset. They are tried after the cache and the files placed in it, and **before** the
record's own locations, since a lab's mirror is usually why the file exists. The file never changes a pin, a record or
a saved folder: **a saved lock holds only the record's own locations**, so an internal server's address never travels
with a shared experiment. A file that cannot be read, or that holds a location MIME does not read, is refused as
`asset.locations-file`, never read as no mirror.

```toml
mirrors = ["file:///mnt/lab-share/mime-assets/"]
ca_bundle = "/etc/ssl/certs/lab-ca.pem"

[assets]
"unit_tube_R0.997" = ["https://files.lab.example.org/mime/unit_tube_R0.997.npz"]
```

**Availability made visible**: `mime assets sources <name-or-folder>` prints, for each asset, its terms, its tier and
each location in the order a fetch tries them, looked at now without downloading (a ranged request for the first byte,
or the adapter's open and close): reachable or not, and why. With `MIME_OFFLINE` set it says it did not look.

## The one resolver

`resolve(ref)` is the single lookup of a content-addressed file. `fetch` is its name for the command, and `mime pull`
uses it for a bundle (a packed experiment is a content-addressed file too) and for the bundle's assets alike, so
they share the cache, the offline switch, the hash check, the refusal codes and the size report. A bundle named by its
DOI or address has no hash to look up before it is downloaded, so `retrieve(location, name=..., known_hash=...)` stores
such a file: the download is held to the repository's own checksum where there is one (Zenodo's md5), hashed in full,
and stored in the cache under its sha256, where `locate` finds it. (`retrieve` still downloads through pooch's
downloader, which `mime pull --signed-in` replaces with its own; an asset's download goes through the adapters.) In
order:

1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged. No network. (`fetch`
   skips this step and hashes the file in full.)
2. **A file already in the cache directory** with the right hash, such as one a cluster's administrator copied in,
   under its sha256 or under a name a person would give it (below). It is hashed in full, once, and kept. No network. A file with other bytes, or of another size (a copy that has not
   finished), is **never deleted or overwritten** if a person placed it: it is refused as `asset.placed-file`, naming
   the file, offline or not, and left where it is. A file MIME downloaded earlier, whose bytes have since changed, is
   replaced by the third step.
3. **The locations**: the locations file's for the asset, then the record's own, primary then backups, each through
   its adapter into a folder of its own, the stream hashed as it arrives. The file is hashed in full again, moved
   into place, and recorded. A wrong hash
   is refused as `asset.hash-mismatch`, all locations failing as `asset.unreachable`. **The size the reference declares
   is enforced while the file downloads**: a download stops once it has passed that many bytes, what it had received is
   discarded, and the next location is tried (all failing this way is `asset.size-exceeded`). A lock that
   declares a megabyte therefore cannot make a download of a gigabyte, and the confirmation `mime pull` asks for above a
   gigabyte is about the size the download can be. A bundle's download is held to the size its record states, or to
   `RETRIEVE_LIMIT_BYTES` (half a gigabyte) where nothing states one: `retrieve(..., max_bytes=...)`.

The second and third steps hold a lock on the asset (a file `<sha256>.lock` beside it: `flock` on POSIX, `msvcrt.locking` on Windows,
which this PR's tests do not exercise). Several processes fetching at once (a job array on a shared filesystem) wait for
the first, then find the file stored: one download per asset, a temporary name of each process's own, and an atomic
rename into place. A process killed mid-download leaves a `<sha256>.incoming-*` folder, removed by the next fetch of
that asset.

`locate(ref)` is what a build calls. It is the first step only: it **never downloads**, and a file that is not there is
refused as `asset.missing`. The refusal is one line in the experiment's terms, with the exact command to paste:

```
the experiment needs 'wall_R2.500' (505 MB), which isn't downloaded
Run: mime assets fetch ./fl9_water/
```

It names the asset and its size, never its hash or the cache's path. Where the experiment's folder is known
(`locate(ref, folder=...)`) the command is scoped to it; otherwise it is `mime assets fetch <name>`.

## The cache

- It is **one directory**: `pooch.os_cache("mime")/assets` by default, and the directory in **`MIME_ASSETS_DIR`**
  (clusters, air-gapped labs) *instead*: when the variable is set, the platform directory is not looked at, and
  downloads go to the variable's directory. A relative value is read from the working directory.
- A file is stored as `<sha256>`, with `<sha256>.verified` beside it: the size and modification time the file had when
  it was hashed in full, the result of that hash, and who wrote the file (MIME's download, or a person). A file a person
  placed in the directory is accepted by its hash, and is never deleted or overwritten; only a file MIME downloaded may
  be replaced.
- **A file placed under a name.** A lab copying tables by hand (an air-gapped cluster) copies them under the names it
  has, and the cache accepts three, in this order after the sha256: the asset's own name, the asset's name with the
  extension of its locations' files, and the file name a location serves (`wall_R2.500.npz`). It is hashed in full and
  accepted when the hash is the reference's: the cache then holds a hard link under the sha256 (a copy on a filesystem
  without them), and the person's file stays where it is, as it is. The first two names are this asset's alone, so a
  file there with other bytes, or of another size (a copy still in progress), is refused as `asset.placed-file`. The
  third may belong to another asset (the `r30/` and `r40/` tables share their file names), so a file there is accepted
  only when its hash matches, refused when it has the right size and other bytes, and otherwise left alone as another
  asset's. No file is deleted, overwritten or renamed. The hard link makes the cache's file and the person's one inode:
  a person who rewrites their file in place, with the same size and the old modification time (`rsync -a --inplace`,
  `cp -p`), changes the cache's bytes too, and `locate` (which checks size and time) will not see it; `mime assets
  check` does.
- **`MIME_OFFLINE`** set to `true`, `yes` or `on` (any case), or to the numeral for true, downloads nothing: the third
  step is refused as `asset.offline`. Unset, empty, `false`, `no`, `off` and the numeral for false leave the network
  open. Any other value is refused
  (`asset.offline-setting`), never read as online. The cache and a file copied in by hand still work, so one placed
  there is accepted.

**What `locate` checks, and what it does not.** Hashing a file of this size on every build is too slow, so `locate`
checks cheaply: the file's size is the reference's, and its size and modification time are those recorded at the full
verification. That finds a truncated, replaced or edited file. It does not find a flipped byte that leaves both
unchanged; `mime assets check` re-hashes in full, and a test writes that limit down. When `check` finds a file corrupt it
marks the record, and `locate` refuses the file (`asset.missing`, with the command) until `mime assets fetch` has
repaired it: `fetch` hashes in full and downloads again a file MIME wrote whose bytes are wrong.

## The commands

The words are in one place, `_commands.py`.

- `mime assets list`: every known asset with its size, cached or missing.
- `mime assets fetch <name-or-folder>... | --all`: resolve, hashing the file in full (a cached file too), and download
  again what is wrong. An argument that is a directory is an
  experiment folder, and stands for exactly the assets its lock references and nothing else (write `./name` to mean a
  folder), and each file of it that travels as a reference is then placed at its path from the cache
  (`mime.saved.place_referenced`); any other is an asset's name; `--all` is every known asset. Exits zero, or two when
  refused or while a file that travels as a reference is left unplaced (`asset.missing`, with why).
- `mime assets check`: re-hash every cached asset in full. Exits one when a file is corrupt, marks it (so `locate`
  refuses it) and leaves it where it is.
- `mime assets sources <name-or-folder>...`: each asset's terms, tier and locations, each looked at now (above).
- `mime assets audit <wheel|sdist|archive|directory>... [--manifest FILE]`: the guard (below). Exits one when it finds
  anything.

"Check" and not "verify": in MIME, verify reruns an experiment against its reference outputs.

## What a folder's lock references

`folder_assets(folder)` returns the references the folder's `mime.lock` names, from the lock's `assets` key (a list of
`AssetRecord`), by reading the JSON alone: no entry is loaded, so a folder's assets can be stored before it can be
rebuilt. The key is part of the saved form: `mime.saved`'s `Lock` has `assets` as an optional list of
`RemoteAssetRecord` (the keys and types of this module's `AssetRecord`, which a test holds together), so `read_folder`
accepts a lock that carries it and refuses a malformed record.

**The build writes it.** An entry's plan names the shipped references it reads (a wall kind's correction names its
table: the tube's, `unit_tube_R0.997`), and the build writes them under the key, each once; an experiment that reads
none writes no key. So a tube experiment's lock names the tube table, `mime assets fetch <folder>` stores it, and the
script `mime modal` writes fetches it onto the assets Volume before a run. A load compares the key with what the
rebuilt experiment reads and refuses (`load.assets`) a lock that does not name an asset it reads, by name, and one
that names it under another hash; a lock may name more than the entries read: a bundle's own files, such as a file
whose terms let no bundle carry it, which the lock's `referenced_files` names with the asset that holds its bytes.
Since the saved form that added them, each record is whole (its credit, role, kind, custom licences and adapter packages), and the lock's
`visuals` holds the manifests of the meshes its stage draws, whose records `mime visuals fetch <folder>` fetches.
`mime assets fetch <name>` still fetches MIME's own assets by name (the Modal image and its recipe fetch
`unit_tube_R0.997` so).

## Per asset, wherever bytes leave the machine

Every exit judges each asset by its own tier, never a bundle as a whole: **open and conditional bytes travel inside,
with their conditions in the NOTICE; referenced bytes never do, only their record.**

| Exit | How it holds |
|---|---|
| save | a file whose tier is referenced is listed in the lock's `referenced_files`, its record in `assets` (`mime.saved`); a derived file over a referenced input is referenced whatever licence its record declares |
| `pack` | such a file never enters the bundle, whatever its size, nor its bytes at another path (`bundle.referenced-copy`); and every file it would carry is held to every record in reach, whatever the folder's own records say: bytes a shared source, the lock, a derived file's inputs or a mesh baked into the cache say may not travel are refused (`bundle.referenced-copy`). The manifest's `files` are what is inside, its `referenced` what is fetched, and its `assets` each record with its locations: from where |
| `push` | a referenced asset is never uploaded (`asset_choices`, which takes the derived terms and every record in reach too), its record travels in the bundle's lock; one that may travel is uploaded where it has no Zenodo location yet; and whatever chose the uploads, each file the record would hold is held to every record in reach (`push.referenced-copy`) |
| recordings and scenes | a run folder's recording composes over its own folder's scene, written by save (above), every composed layer resolves inside its folder, and a mesh baked from a referenced source is composed at view time from the asset cache, never written into a saved or run folder |
| the repository, a wheel, an sdist, a plugin's wheel | the guard, below, in MIME's CI and at each release |

## Derived files' terms

A derived file takes the strictest terms of its inputs: the `AND` of their verdicts (`mime.terms.derived_verdict`), so
one referenced input makes it referenced. Where inputs carry different conditional licences MIME does not judge
whether their terms can hold together, and the file is referenced unless a declared permission bound to its own hash
says otherwise (`exp.permission`). A derived asset's inputs are read from its derivation record, by hash
(`derived_terms(record, known)`); an input no record names has terms nobody stated, and makes the file referenced. A
mesh MIME bakes is a derived file of its source: its record in the cache states its tier.

**Every exit applies it** (save, `pack`, `push`, a run's folder, the guard), through one function, `record_terms(ref,
known)`: a record's licence, and for a record that names its derivation record (`derivation`, its sha256) the
strictest of that and its inputs', each input judged so in turn along a chain. A record that declares an open licence
over a referenced input is referenced. The derivation record is read by hash (`derivation_by_hash`): among those MIME
ships, a file of that hash in the asset cache, and those inside a target the guard reads; **one that cannot be read
here, or that describes another file, makes the file referenced** (it fails closed). A record whose kind is derived
and that names no derivation record states no inputs, and is judged by its licence (MIME's earlier wall tables are
such records).

## The guard

`audit(target)` and `mime assets audit` read a wheel, an sdist, a zip or tar archive, or a directory, hash every file
in it, and find **any file whose sha256 is a referenced record's** (every record known here: the shared sources',
each manifest, lock and record file inside the target, and the manifests given), **any derived file of one** (a
record's derivation whose inputs make it referenced, as every exit judges it, and a mesh baked from one in the local
cache, by its record's output hash), and **any mesh or scene file no record names**. A mesh whose record is open or
conditional passes, and so do a saved folder's own stage and layers, which its lock names.

- **Formats by their bytes**, the extension a second hint only: binary and ASCII STL, USD's crate and text (the crate
  is what MIME's baking writes), OBJ, PLY, glTF and its binary, OFF, VTK, Gmsh, VRML, X3D, COLLADA, FBX's binary,
  STEP, Blender and Alembic.
- **Containers by their bytes, never by name**: a zip (a wheel, a bundle, an `.npz`, a zip renamed `.dat`), a tar, and
  a gzip, bzip2 or xz stream, one inside another to six layers (`DEPTH_LIMIT`: an sdist's gzip and its tar are two); a
  file inside another is named after a `!` (`pkg/a.tar.gz!inner/m.stl`).
- **Base64 blocks in a text file** that decode to half a kibibyte or more (`BASE64_MIN_BYTES`): a file of base64, a
  data URI, lines broken as mail breaks them, either alphabet; each is read as a file of its own, named by where it
  starts (`page.html#base64@<byte>`).
- **Nothing is skipped by name**, in an archive or in a tree, but a checkout's own `.git` at the top of a tree, which
  the report says it left out. A link in a tree is not followed (an archive of the tree holds the link, not the bytes),
  and the report counts them.
- **Its limits refuse, never skip** (`asset.audit`): a file it must look inside over a gibibyte (`MEMBER_LIMIT_BYTES`:
  an archive, a stream, a text it searches), more than sixteen gibibytes expanded from one target
  (`EXPANDED_LIMIT_BYTES`), more than a million files (`COUNT_LIMIT`), a layer deeper than six, and an archive or a
  stream it cannot read to its end (corrupt, encrypted, a compression it does not know). A larger file of any other
  kind is hashed as it is read.

**What a hash guard cannot do**: a converted file (an STL written out as USD by any tool, a mesh decimated or
re-encoded) has other bytes, and only the mesh check sees it, as a mesh no record names; a mesh held as plain numbers
(an array in an `.npz`) is no mesh format at all. And the guard trusts the target's own records for a mesh no other
record knows: a plugin that ships its own record calling a mesh open is believed, unless a record in reach (a shared
source, a manifest given with `--manifest`) says otherwise, which then wins.

MIME's CI runs it on the built wheel and sdist and on the committed tree (`git archive`: what a GitHub release's
source archive and Zenodo's snapshot of a tag hold), with a planted mesh as its control; a plugin's CI runs the same
command, since a plugin's wheel archived by sha256 is a redistribution too.

## Derived assets

An asset's truth is authored (true by declaration), measured (a fact with an uncertainty) or derived (a
cached computation: a wall table, a surrogate's weights, its training data). The reference is the same for all three;
a derived asset adds a **derivation record** (`DerivationRecord`): its builder and the commit it ran at, the sha256 of
each source file it ran and whether the tree was dirty (`working_state`; `None` where a build ran outside the
repository and recorded neither), its inputs by the role each plays and its hash, its parameters, its checks with their
results, and its validity domain.

- **`derive(asset, builder, parameters, inputs, ...)`** runs a `Builder` (its name, version and `produce`) and returns
  the bytes with their record; a check that failed, or no check at all, refuses the build (`asset.derivation-check`),
  so a failed or an unchecked build produces no asset.
- **`stale_inputs(record, current, unjudged=...)`**: each input whose role is now played by an asset with another hash. A
  surrogate built through a table that has since been replaced is stale, mechanically (`mime.surrogates.staleness`).
  It fails closed: an input whose role nothing plays and no caller declared unjudged (a misspelled `wall:Tube`) is
  refused by name (`asset.unknown-role`), not passed as fresh.
- **The shipped records** are `mime.data`'s `derivations/<asset>.json` (`shipped_derivations`, `derivation_of`): the
  corrected tube table's, read from its file's provenance arrays (`table_provenance` in the tube's wall correction)
  with its gate's checks; and `dejongh_mlp_v2`'s, whose inputs are the tables that make it stale.

Registered builders, `mime assets build` (locally or through the Modal template) and rebuilds are the general version,
after `Study.sweep`; they register builders behind this interface.

## What ships

`mime.data`'s `assets/unit_tube_tables.json` holds the reference of the corrected table of the unit tube's image system,
`unit_tube_R0.997` (readable from the axis to the radius its name says; MIME-ANO-117), pinned to its revision of the
public dataset `NickERoy/mime-wall-tables` (public domain, CC0) by a resolve URL; a test holds its hash and size to the
uploaded file's, recorded apart from the package. The tube's wall kind (`tube`) reads it, through `locate`. The table at the earlier, lower truncation is kept beside it as `unit_tube_R0.997_1667_modes`, superseded (its reciprocity missed the gate's tolerance at the wall's edge) and read by nothing.

Each states its role (`results`) and kind (`derived`), the corrected table with its derivation record's sha256. MIME's
own data holds no other record: a robot's meshes are a manifest beside its description, and a lab's records are its
package's (`mime.assets` entry point).

`assets/wall_tables.json` holds the references of the v0.2 line's eleven wall tables, each pinned to one revision of the
same dataset beside the `SHA256SUMS.txt` it was uploaded with; a test holds them to each other. They are kept as the
record: wrong by MIME-ANO-117, and read by nothing.

Hugging Face is the only location today; a Zenodo location will be put first, and Hugging Face kept as a backup and
then dropped.

## Status

The wall tables are wired to it. `mime pack`, `mime pull` and `mime push` are built on this module (`retrieve`, the file placed under a name). Records travel with what needs them (sources, pins, manifests), open locations go through adapters with the resolver keeping the hash, the size cap, host binding, retries and resuming, and the guard runs in CI. Left for later: a Software Heritage adapter and a git-at-commit adapter, real authenticated adapters on the credentials hook (a storage plugin), a private tier, and `retrieve` (a bundle by DOI or address) moved onto the adapters. The corrected unit-tube table is shipped as a reference and the tube's wall kind reads it through `locate`; CI stores it once per job (`.github/actions/unit-tube`) and the tests never download it. A saved experiment's lock records the assets its entries read, and
derived assets carry derivation records (above).

Releases are explained in the [package manifest](../README.md#status).
