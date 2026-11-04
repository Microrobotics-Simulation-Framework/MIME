# Assets in depth

[Install and environment](install.md#assets-a-run-fetches) says how to fetch what a run needs. This page says what an asset
is, where an experiment's records come from, where its bytes may go, and how a lab keeps its own copies. The full contract is
in `src/mime/assets/README.md`.

## The record

An asset is a file named by the `sha256` of its bytes and its size, never by where it lives. Its **record** says what it is
and under what terms: its licence (an SPDX expression, or a lab's own `LicenseRef-`), its credit, its role (a result depends
on it, or it is only for drawing the stage), and its kind by where its truth comes from: **authored** (a CAD mesh, a
description: true by declaration), **measured** (a calibration, digitised data: a fact with an uncertainty) or **derived**
(a wall table, a surrogate's weights: a computation, with a derivation record of its inputs by hash and the build that made
it). The record also lists places to fetch the file from; they are only hints, since the hash decides whether the bytes are
the right ones.

## Where an experiment's records come from

In this order:

1. **The experiment itself**: a record it declares, or the one its lock holds once it is saved. A saved experiment carries
   every record whole, so it never depends on a registry being installed.
2. **The manifest beside a description**: choosing a robot's description brings the records of its meshes.
3. **Installed packages**, through a `mime.assets` entry point ([extending MIME](lab_author.md#shipping-assets-from-a-lab)).
4. **MIME's own**, for MIME's own files, such as the wall tables.

**The pin wins.** An experiment that pinned a hash keeps it when a shared source later holds another version of the same
name; the build report says so (its `assets` field, [reading a build report](build_report.md#assets)). Two shared sources
that disagree on a name, with no pin to settle it, are refused (`asset.conflict`).

## Where the bytes may go

The licence decides where the bytes may live, asset by asset, and never whether MIME accepts the file:

- **Open and conditional** licences: the bytes travel inside a pack or a push, with their conditions in its notice.
- **Referenced** (a licence that forbids hosting copies, or a third party's file whose licence is unknown): only the
  record travels. The file is fetched
  from its owner, a mirror the user keeps, or a copy placed by hand, and the record's hash checks it. The AR4 arm's meshes
  are referenced: Annin Robotics' licence lets no one host them, so `examples/descriptions/ar4/visuals.json` names each at a
  pinned commit of Annin Robotics' repository, and `mime visuals fetch` converts them on your machine, into the cache only.
- **A derived file takes the strictest terms of its inputs**: a table made from a referenced file is referenced, whatever
  licence its own record states. A recording or a scene references a restricted layer and never embeds it, so a recording
  stays shareable while the restricted part is fetched and baked locally.

Saving, `mime pack`, `mime push` and a run's recording each check this for every file they would write; a file whose terms
let no copy travel is refused there, or travels as its record alone.

## Where this machine gets a file

The cache first, then a file placed in the cache directory by hand, then the locations file's entries for the asset, then
the record's own locations in order. Each download is cut off at the record's size and kept only if its hash is right. A
remote location may redirect only to https (a location on this machine itself, to http too), never to a file on this
machine.

- **The locations file** (`MIME_ASSET_LOCATIONS`, else `locations.toml` in your config directory) maps a hash or a name to a
  lab's own copies: a mounted share, an internal server. It is never written into a saved experiment, so an internal address
  never travels.
- **Schemes:** `https`, `doi`, `package:` (a file inside an installed package) and a mirror directory. A plugin adds others
  with a `mime.asset_adapters` entry point, for storage that needs a login; `mime.testing.verify_adapter` checks one.
- **Credentials** are bound to their host: kept with `mime credentials set host:storage.example.org`, sent to that host and to
  no other, and never written into an error, a log or a saved file. A server signed by an institution's own certificate
  authority needs `MIME_CA_BUNDLE`.

## For a package's CI

`mime assets audit dist/*` reads a wheel, an sdist or a directory, looks inside archives, compressed streams and base64
blocks, and fails on any file whose terms let no copy be hosted, and on a mesh no record names. MIME's own CI runs it on its
packages and its repository tree. What a hash cannot catch: a mesh converted to another format has other bytes, and a
package's own records are believed for a mesh no other record knows.
