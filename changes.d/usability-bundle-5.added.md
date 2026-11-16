- **Usability bundle 5: `mime.scene.rotation`, two filters, and a warning at `run()`.** `mime.scene.rotation(axis, angle)`
  is the unit quaternion for a turn of `angle` (`deg(30.0)`) about `axis` (`Z`, or three numbers), right-handed and in
  MIME's order `(w, x, y, z)`, which `State`'s docstring now states (`(x, y, z, w)` from ROS or SciPy is a different
  rotation, never refused); a zero axis, a bare number or a non-finite angle is refused as `call.rotation`.
  `mime registry --entry <name>` lists (or, with `--json`, exports, with the namespaces of those entries) the entries of one name, every version and kind, and
  `mime assets list --namespace <source>` the records of one asset source (`mime`, or an installed package's `mime.assets`
  entry point); a name that holds nothing is refused with the nearest names (`registry.unknown-entry`, `asset.unknown`),
  never an empty answer. `run()` warns, before it spends the run, of an observation whose window starts at or after
  the run's end (its outputs at the end would read as nothing observed, `run.window-past-end`, at the read).
