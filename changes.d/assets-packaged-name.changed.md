- **A packaged name is MIME's, and a resolver owns its namespace** (review of the one `asset()`). A lock's record, a pin or a
  script's `AssetRef` that takes the name of a record MIME ships (`navion_2`, `dejongh_mlp_v2`, ...) for other bytes or
  terms is refused (`asset.packaged-name`), where the lock is read for `mime assets fetch`, where a build keeps its pins and
  in `asset(AssetRef(...))`; a record that restates MIME's own changes nothing. A dotted name in a namespace an installed
  resolver owns goes to that resolver first and is never respelled onto MIME's own `<source>_<name>`
  (`openmaterica.example_water` asks the `openmaterica` resolver); the earlier `<source>_<name>` spelling of an installed
  package's record now says once, with a `FutureWarning`, which dotted name to write. `mime scaffold plugin` refuses the
  first word of a packaged name (`navion`, `openmaterica`, `dejongh`, `unit`, ...) as a namespace. A resolver that exits
  (`SystemExit`) is refused as `asset.resolver`; an interrupt propagates.
- `asset()` refuses every spelling of a remote form as `asset.source` (`DOI:...`, `HTTP:...`, `s3:...`, `//host/...`,
  `\\host\share\...`, and a `Path` that reads as a URL), with an action that never suggests a path; a named `Asset(name)`
  without its sha256 is refused as it was before the fold. The documentation of `mime.assets`' internal names in the
  package READMEs now says they are internal.
