- **MIME's packaged files are records**, in `mime.data`'s `assets/packaged.json` (the surrogate's weights, the
  sphere-plane table, the two OpenMaterica example records and the three eMNS calibrations, each with its licence and
  credit), read in place from the installed package and held to the sha256 the experiment pins. Their names, pins, saved
  forms and the lock's `resolved.assets` are as they were, and every one is open. `mime assets list` shows them as
  `packaged`. A calibration and a surrogate's weights read a record's name, whether MIME packages it or an installed
  package ships it (`calibration=asset("labx.emns.yaml")`, no longer `Asset(name, sha256)` typed by hand); an arm's
  description and a scene's layer read a file or a whole record, and a name given to `scene.reference` is refused
  (`asset.source`).
- **Records are named with a dot, like entries**: a package's records are `<namespace>.<name>`, MIME's own have no
  namespace. A record still spelled `<namespace>_<name>` is found under either spelling for this release and keeps its own
  name, so no lock changes. `mime scaffold plugin` writes `<namespace>.example_record`, and the lab fixture's records
  moved onto dots but one, kept under its earlier spelling to hold the rule.
