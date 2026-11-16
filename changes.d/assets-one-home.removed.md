- `mime.data`'s and `mime.scene`'s asset functions and classes, folded into `mime.assets.asset` and `mime.assets.Asset`:
  `mime.data.asset`, `Asset`, `AssetError`, `PACKAGED`, `PackagedAsset`, `asset_bytes`, `asset_problem` and `asset_source`
  (read a named asset's bytes with `asset(name).read()`; a refusal is a `MimeError` with its code), and `mime.scene.asset`
  and `mime.scene.Asset` (`mime.asset` is the one function; a path is written `asset(Path(...))`).
