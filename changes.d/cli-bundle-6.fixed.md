- **CLI bundle 6: the CLI's loose ends.** `mime registry --json | head` ends quietly (exit 141, as a process ended by
  SIGPIPE shows) instead of asking the user to report a bug. Every command that takes a folder refuses an absent one as
  `load.missing-folder`, with the folders beside it and the nearest (`check`, `modal`, `push` and `scaffold bundle|study`
  named it by the first file each looked for). `mime verify` and `mime pending` given a run folder say it is one
  (`load.run-folder`) and what reads it. `mime study run --out <folder that holds something>` refuses before it loads the
  base and plans, not after about half a minute of it. An unrecognised flag shows its command's usage, not the top one's.
  `mime serve --commands 127.0.0.1:5555` (no `tcp://`) is refused as `live.endpoint`, not as `live.remote-bind`.
