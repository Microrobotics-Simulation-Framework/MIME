- **One way to declare a file: `asset(source)`** (`mime.asset`, `mime.assets.asset`), its origin decided by the type of
  `source` and never guessed from a string's shape: a string is a record's name (`asset("dejongh_mlp_v2")`,
  `asset("softtissue.fibrin_plasma")`), a path a local file (`asset(Path("arm.urdf"), sha256=..., terms=...)`), an
  `AssetRef` a whole record. It always gives one `Asset` (`mime.assets.Asset`), whose `origin` says which and whose `ref()` is
  the record behind it: a name's record, the record declared, or, for a local file, one made from its bytes, its licence
  the one its terms state or `NOASSERTION`, so the record alone grants nothing until a licence is declared. A string that is
  a location is refused (`asset.source`: a remote file needs a record with its sha256), a name no record holds is refused
  with `asset(Path(...))` as the action (`asset.unknown`), and any other type is `call.wrong-type`.
- **Resolvers.** A `mime.assets` entry point whose value names an object (`module:object`) is a `mime.assets.Resolver`:
  asked for a name in its namespace that nothing else holds, it answers with that name's record, held as any other by
  its sha256 and licence. One that answers with another name's record, or fails, is refused (`asset.resolver`).
  `mime.testing.RecordResolver` answers from records held in memory, for a test's own source.
