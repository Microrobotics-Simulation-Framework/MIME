- **A live session never writes over another's run folder.** Two `mime serve` sessions started in the same second in
  one folder shared the default `<experiment>-live-<UTC second>/`, and the second deleted the first's `experiment/` and
  truncated its `commands.jsonl`; `mime serve --out <a folder that holds something>` wrote over it the same way. The
  default run folder is now made by the session that takes it (`-2`, `-3` and on for a second and a third in the same
  second; `mime.live.default_out` makes it). `--out` takes `mime run`'s rule, judged before anything binds or loads: a
  new folder or an empty one, any other refused (`run.out-not-empty`) unless `--force` (new; `serve(force=)`,
  `built.live(force=)`), which warns what it holds and replaces only the files a run writes. A session opens its command
  log exclusively, before it writes anything else into the folder, so a folder that holds one already is refused,
  with `--force` too (`live.log-exists`, new), and `experiment/` is written file by file, never deleted first.
  `replay` and `mime run --commands` take the same rule.
