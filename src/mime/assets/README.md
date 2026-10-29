# `mime.assets`

**Large files, referenced by hash and fetched on request.** The wall tables the boundary-element method reads are far
too large for the repository or the wheel. An experiment names such a file by the sha256 of its bytes, never by where
it lives, and the file is stored on the machine only when someone asks.

## The reference

An `AssetRef` is `name`, `sha256`, `size` (bytes), `licence`, and `locations`: an ordered tuple of `https://` URLs or
`doi:` URLs that pooch resolves (a DOI, then the file's name in the record). Plain `http://` is accepted only for a
literal loopback host (`localhost` or the IPv4 or IPv6 loopback address). The host is read once, by the parser the download uses
(`urllib3`'s), and a URL with a backslash, whitespace, a control character, userinfo (`user@`) or a percent-escape in its
host is refused, since parsers read those differently: a location that is refused is never contacted, even by a
reference built around the check. A malformed reference is refused whole as `asset.reference`. (A redirect is followed
wherever it leads, as Hugging Face's is to its CDN; the hash is checked after it, so the bytes are the reference's
whatever the route.)

In a lock it is the record `AssetRecord` (`name`, `sha256`, `size`, `licence`, `locations`), through `to_record()` and
`AssetRef.from_record()`. Because the hash names the file, a new home for it is one more location: no saved
experiment changes, and none breaks.

**Primary, then backups.** The first location is the primary, the permanent archive (Zenodo, once it exists); the
rest are backups, tried only when the one before fails. A location fails when it is unreachable, errors, or serves
other bytes than the hash.

## The one resolver

`resolve(ref)` is the single lookup of a content-addressed file. `fetch` is its name for the command, and `mime pull`
uses it for a bundle (a packed experiment is a content-addressed file too) and for the bundle's assets alike, so
they share the cache, the offline switch, the hash check, the refusal codes and the size report. A bundle named by its
DOI or address has no hash to look up before it is downloaded, so `retrieve(location, name=..., known_hash=...)` stores
such a file: the download is held to the repository's own checksum where there is one (Zenodo's md5), hashed in full,
and stored in the cache under its sha256, where `locate` finds it. In order:

1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged. No network. (`fetch`
   skips this step and hashes the file in full.)
2. **A file already in the cache directory** with the right hash, such as one a cluster's administrator copied in,
   under its sha256 or under a name a person would give it (below). It is hashed in full, once, and kept. No network. A file with other bytes, or of another size (a copy that has not
   finished), is **never deleted or overwritten** if a person placed it: it is refused as `asset.placed-file`, naming
   the file, offline or not, and left where it is. A file MIME downloaded earlier, whose bytes have since changed, is
   replaced by the third step.
3. **The locations**, primary then backups, each through pooch with `known_hash` into a folder of its own, which stores
   the file only if the hash is right. The file is hashed in full again, moved into place, and recorded. A wrong hash
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
  folder); any other is an asset's name; `--all` is every known asset. Exits zero, or two when refused.
- `mime assets check`: re-hash every cached asset in full. Exits one when a file is corrupt, marks it (so `locate`
  refuses it) and leaves it where it is.

"Check" and not "verify": in MIME, verify reruns an experiment against its reference outputs.

## What a folder's lock references

`folder_assets(folder)` returns the references the folder's `mime.lock` names, from the lock's `assets` key (a list of
`AssetRecord`), by reading the JSON alone: no entry is loaded, so a folder's assets can be stored before it can be
rebuilt. Nothing in the current saved form records an asset, so the key is absent and the folder references none. The key
is part of the saved form: `mime.saved`'s `Lock` has `assets` as an optional list of `RemoteAssetRecord` (the keys and
types of this module's `AssetRecord`, which a test holds together), so `read_folder` accepts a lock that carries it
and refuses a malformed record. The build that records the first asset writes that key, and the reading needs no change.

## What ships

`mime.data`'s `assets/unit_tube_tables.json` holds the reference of the corrected table of the unit tube's image system,
`unit_tube_R0.997` (readable from the axis to the radius its name says; MIME-ANO-117), pinned to its revision of the
public dataset `NickERoy/mime-wall-tables` (public domain, CC0) by a resolve URL; a test holds its hash and size to the
uploaded file's, recorded apart from the package. Nothing reads it yet: the tube's wall kind will.

`assets/wall_tables.json` holds the references of the v0.2 line's eleven wall tables, each pinned to one revision of the
same dataset beside the `SHA256SUMS.txt` it was uploaded with; a test holds them to each other. They are kept as the
record: wrong by MIME-ANO-117, and read by nothing.

Hugging Face is the only location today; a Zenodo location will be put first, and Hugging Face kept as a backup and
then dropped.

## Status

Landed before phase 7h, which wires the wall tables. `mime pack` and `mime pull` are built on this module (`retrieve`, the file placed under a name); `mime push` will be. The corrected unit-tube table is shipped as a reference; the tube's wall kind, which will read it through `locate`, is still reserved.

Phases and releases are explained in the [package manifest](../README.md#status).
