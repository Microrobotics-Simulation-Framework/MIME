- **CLI bundle 6: no silent overwrite, and fewer refusals that were someone else's fault.**
  `mime run --out <folder>` refuses MIME's own run folder too, as it refuses any folder that holds something
  (`run.out-not-empty`): a run never replaces another run unasked, and `--force` runs into it with the warning and the
  cancel-until-writing it already had (`BuiltExperiment.run_to` from Python still runs again into its own folder).
  `mime modal` refuses a folder that holds a script of an earlier run (`modal.out-exists`) unless `--force`.
  `mime env check` without a folder, `--base` or `--mime-version` no longer fails a plain install on versions newer than
  the pins CI tested: a distribution installed at another version, inside the ranges the installed distributions
  declare, and one nothing pins, are notes in the report; one outside its ranges, and every difference under `--strict`
  (implied by a folder, `--base` and `--mime-version`, the reproduction's own arguments), still refuse. `mime docs build`
  outside a checkout says, at its start and at its end, that it built only the generated pages and left the guides out.
  A switch owner outside MIME's namespace that does not start with its package's name is reported as the package's to fix,
  with the owner to write.
