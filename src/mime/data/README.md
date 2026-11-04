# `mime.data`

**Small package data shipped in the wheel.**

## What goes here

- **Surrogate weights**, pinned by sha256.
- **Calibrations** for electromagnetic navigation systems, from a2.
- **The environment's pins**, `env/`: each variant's hashed requirements, exported from `uv.lock`, the base image of
  each Python series and the uv that installs the pins (`uv.json`), each pinned by digest, which [`mime.env`](../env/README.md) reads.

## The rule

Only small files belong here. Anything large is an asset, referenced by hash from the experiment: `assets/` holds
the references (not the files) of the wall tables, read by [`mime.assets`](../assets/README.md).

Every file here is committed and ships in the wheel and the sdist: in the 0.2 line a root `.gitignore` rule kept the
surrogate's weights out of git, so they reached neither GitHub nor PyPI. `tests/test_packaged_data.py` fails if git
stops tracking one or ignores it, and `tests/test_packaging.py` if a built distribution lacks it.

## Status

**`mlp_cholesky_weights_v2.npz`** (151,887 bytes, sha256 `ac1fca5c...0a345`), the drag
surrogate's weights, as the asset **`dejongh_mlp_v2`**. `asset(name)` names a packaged asset with the sha256 MIME pins
for it; `asset_bytes` reads its contents and refuses a file whose sha256 is not the one asked for.
The saved form writes an asset by name and sha256, the lock records its source and sha256, and a load refuses an
installation that ships other contents under the name: assets by name from MIME's own package. An external file is
declared by its record (`mime.scene.asset(AssetRef(...))`), wherever it is kept (`mime.assets`).

**`assets/`**: the records of MIME's own remote assets, the wall tables (each with its role and kind, the corrected
tube table with its derivation record's sha256). MIME's data holds records of its own assets only: a robot's meshes are
a manifest beside its description (the AR4's in `examples/descriptions/ar4/`), and a lab's records are its package's.

**`records/`**: copies of OpenMaterica's example records `example/water` and `example/ndfeb_n45`, as
the assets `openmaterica_example_water` and `openmaterica_example_ndfeb_n45`, which `mime.materials.packaged_record`
names. Their values are CC0-1.0, their origin OpenMaterica's tag `openmaterica-v0.1.0.dev3`; they are examples, not
reviewed library data, and are never edited here (`records/README.md`).

Releases are explained in the [package manifest](../README.md#status).
