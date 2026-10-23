# `mime.data`

**Small package data shipped in the wheel.**

## What goes here

- **Surrogate weights**, pinned by sha256.
- **Calibrations** for electromagnetic navigation systems, from a2.

## The rule

Only small files belong here. Anything large is an asset, referenced by hash from the experiment.

Every file here is committed and ships in the wheel and the sdist: in the 0.2 line a root `.gitignore` rule kept the
surrogate's weights out of git, so they reached neither GitHub nor PyPI. `tests/test_packaged_data.py` fails if git
stops tracking one or ignores it, and `tests/test_packaging.py` if a built distribution lacks it.

## Status

Since phase 4c1, **`mlp_cholesky_weights_v2.npz`** (151,887 bytes, sha256 `ac1fca5c...0a345`), the drag
surrogate's weights, as the asset **`dejongh_mlp_v2`**. `asset(name)` names a packaged asset with the sha256 MIME pins
for it (also `mime.asset`); `asset_bytes` reads its contents and refuses a file whose sha256 is not the one asked for.
The saved form writes an asset by name and sha256, the lock records its source and sha256, and a load refuses an
installation that ships other contents under the name. This is the smallest form of the design's `asset(...)`:
assets by name from MIME's own package; assets by DOI or URL, an experiment folder's own, and assets that cannot be
redistributed come later.

Phases and releases are explained in the [package manifest](../README.md#status).
