# `mime.assets`

**Large files, referenced by hash and fetched on request.** The wall tables the boundary-element method reads are far
too large for the repository or the wheel. An experiment names such a file by the sha256 of its bytes, never by where
it lives, and the file is stored on the machine only when someone asks.

## The reference

An `AssetRef` is `name`, `sha256`, `size` (bytes), `licence`, and `locations`: an ordered tuple of `https://` URLs or
`doi:` URLs that pooch resolves (a DOI, then the file's name in the record). Plain `http://` is accepted for the local machine
only. A malformed reference is refused whole as `asset.reference`.

In a lock it is the record `AssetRecord` (`name`, `sha256`, `size`, `licence`, `locations`), through `to_record()` and
`AssetRef.from_record()`. Because the hash names the file, a new home for it is one more location: no saved
experiment changes, and none breaks.

**Primary, then backups.** The first location is the primary, the permanent archive (Zenodo, once it exists); the
rest are backups, tried only when the one before fails. A location fails when it is unreachable, errors, or serves
other bytes than the hash.

## The one resolver

`resolve(ref)` is the single lookup of a content-addressed file. `fetch` is its name for the command, and `mime pull`
will use it for a bundle (a packed experiment is a content-addressed file too) and for the bundle's assets alike, so
they share the cache, the offline switch, the hash check, the refusal codes and the size report. In order:

1. **The cache**, keyed by the hash: a file whose full verification is recorded and unchanged. No network.
2. **A file already in the cache directory** with the right hash, such as one a cluster's administrator copied in.
   It is hashed in full, once, and kept. No network.
3. **The locations**, primary then backups, each through pooch with `known_hash`, which stores the file only if the
   hash is right. The stored file is hashed in full again before it is recorded. A wrong hash is refused as
   `asset.hash-mismatch`, all locations failing as `asset.unreachable`.

`locate(ref)` is what a build calls. It is the first step only: it **never downloads**, and a file that is not there is
refused as `asset.missing`. The refusal is one line in the experiment's terms, with the exact command to paste:

```
the experiment needs 'wall_R2.500' (505 MB), which isn't downloaded
Run: mime assets fetch ./fl9_water/
```

It names the asset and its size, never its hash or the cache's path. Where the experiment's folder is known
(`locate(ref, folder=...)`) the command is scoped to it; otherwise it is `mime assets fetch <name>`.

## The cache

- It is `pooch.os_cache("mime")/assets`, or the directory in **`MIME_ASSETS_DIR`** (clusters, air-gapped labs).
- A file is stored as `<sha256>`, with `<sha256>.verified` beside it: the size and modification time the file had
  when it was hashed in full.
- **`MIME_OFFLINE`** set to `true` downloads nothing: the third step is refused as `asset.offline`. The cache and a file copied in by hand
  still work, so one placed there is accepted.

**What `locate` checks, and what it does not.** Hashing a file of this size on every build is too slow, so `locate`
checks cheaply: the file's size is the reference's, and its size and modification time are those recorded at the full
verification. That finds a truncated, replaced or edited file. It does not find a flipped byte that leaves both
unchanged; `mime assets check` re-hashes in full, and a test writes that limit down.

## The commands

The words are in one place, `_commands.py`.

- `mime assets list`: every known asset with its size, cached or missing.
- `mime assets fetch <name-or-folder>... | --all`: resolve, then hash in full. An argument that is a directory is an
  experiment folder, and stands for exactly the assets its lock references and nothing else (write `./name` to mean a
  folder); any other is an asset's name; `--all` is every known asset. Exits zero, or two when refused.
- `mime assets check`: re-hash every cached asset in full. Exits one when a file is corrupt, and leaves it where it is.

"Check" and not "verify": in MIME, verify reruns an experiment against its reference outputs.

## What a folder's lock references

`folder_assets(folder)` returns the references the folder's `mime.lock` names, from the lock's `assets` key (a list of
`AssetRecord`), by reading the JSON alone: no entry is loaded, so a folder's assets can be stored before it can be
rebuilt. Nothing in the current saved form records an asset, so the key is absent and the folder references none. The
build that records the first asset writes that key, and the reading needs no change; a test holds it to a real lock
with the key added.

## What ships

`mime.data`'s `assets/wall_tables.json` holds the references of the eleven wall tables, each pinned to one revision of
the public dataset `NickERoy/mime-wall-tables` (public domain, CC0) by a resolve URL, beside the `SHA256SUMS.txt` the
dataset was uploaded with; a test holds them to each other. Hugging Face is the only location today; a Zenodo location
will be put first, and Hugging Face kept as a backup and then dropped. Nothing reads them yet: the tube's wall kind
will.

## Status

Landed before phase 7h, which wires the wall tables. `mime pack`, `mime push` and `mime pull` build on this module.

Phases and releases are explained in the [package manifest](../README.md#status).
