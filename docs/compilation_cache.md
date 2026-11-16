# XLA's compilation cache

**`mime run` and `mime serve` keep the programs JAX compiles, so that the second run of an experiment compiles less.**
The numbers a run gives are the same with it and without it; what changes is the time spent before the first step.

## What happens by default

- **`mime run` and `mime serve`** point JAX's persistent compilation cache at `xla` in the user's cache directory
  (`platformdirs.user_cache_dir("mime")`; `~/.cache/mime/xla` on Linux), in a folder per jaxlib release, platform and,
  on a CPU, the hash of its instruction-set flags (see [`mime.env`](../src/mime/env/README.md)). Every program is stored,
  however quick it was to compile (`jax_persistent_cache_min_compile_time_secs = 0` and
  `jax_persistent_cache_min_entry_size_bytes = -1`, the values MIME's own CI uses), because an experiment's build
  compiles a few hundred small programs and the saving is in those as much as in the one large step.
- **`mime verify` does not use it.** A verification compiles as a fresh process would, so that what it reproduces is what
  a fresh process computes; its help says so.
- **`--no-xla-cache`** on `mime run` or `mime serve` keeps no cache for that command.
- **A cache JAX was already given is left alone.** With `JAX_COMPILATION_CACHE_DIR` set (or the setting made before
  the command), MIME changes nothing: it is the user's choice.
- **The cache holds at most 2 GB in all, in two parts.** MIME gives JAX 64 MB as its own cache size limit
  (`jax_compilation_cache_max_size`) for the directory the command uses: JAX then evicts the least recently used entry
  (reading one renews it) when it adds another. 64 MB is about 3,300 entries at the 19 kB mean measured on one experiment
  (493 entries, median 3.3 kB, largest 0.83 MB, which fits 77 times). It is held low because JAX scans the directory at
  each write it makes: measured through MIME's setup with 3,300 entries resident, a write took a median 62 ms (maximum
  100 ms) at the full limit, against 1 ms in an empty directory and 12 ms with 500. At the start of a command, MIME holds
  the directories of other jaxlib releases, platforms and CPUs, which JAX's limit does not reach, to what is left of the
  2 GB (1.94 GB), oldest file first. An entry deleted costs one recompile; an entry over 64 MB is not stored (JAX warns
  and writes nothing, evicting nothing).
- **JAX's limit also puts a file lock around every read and write of an entry**, which is the reason it is set (below);
  it needs the `filelock` package, which MIME depends on. The lock costs about 0.3 to 1 ms on each read of an entry:
  about 20 ms of a 2.8 s warm `mime run` of the toy experiment (measured, 20 reads), 0.7 %. If another process holds the
  lock (a command compiling and writing at that moment), a read or write waits for it, up to 10 s, and then is skipped
  with JAX's own warning: a run slows down at worst and never hangs. A `.lockfile` the user cannot write (a read-only or
  another user's) is found when the command starts, and is the "cannot be written" note below.
- **`mime cache clear`** deletes the directory and says how much it held.
- **A cache that cannot be written** (a read-only home, a full disk) does not stop the run: the command says so once on
  standard error and compiles without it.
- **Two commands at once are safe.** Without a limit JAX writes an entry in place, with no lock, so two `mime run`s (or a
  `mime serve`, or the shards of a study run at once) that compile one program together can both write its entry, and when
  their frames differ in length the file ends as a complete frame plus the longer one's tail, which JAX never rewrites. With
  the limit set, JAX takes `<directory>/.lockfile` around each read and write, so the writes follow one another (measured,
  two processes compiling 150 programs together into a fresh directory, 13 trials on each of Python 3.12 and 3.14: no
  damaged entry with the limit, 16 and 121 damaged entries without it). A lock held longer than 10 s is JAX's timeout: that
  read or write is skipped with JAX's own warning, and the run goes on.
- **What it cannot do.** It holds compiled programs, not tracing: a second run still traces and lowers each program,
  and the first run of an experiment, or of a changed one, pays in full. A process killed mid-write (a second
  Ctrl+C), or an entry written before JAX's limit was set (by an earlier MIME, or by two commands at once then), can leave
  a truncated or damaged one. The next run that
  reads it gets a warning from JAX and compiles instead: it succeeds and its numbers are those of a run with no cache
  (tested with entries cut to half and to nothing). JAX does not rewrite an entry whose file exists, so the warning
  repeats on every run that reads it.

## If a run prints `Error reading persistent compilation cache entry`

That is JAX saying an entry of the cache is damaged (`Unknown frame descriptor`, `ended before the end-of-stream
marker`): the run went on and compiled the program instead, and its numbers are right. The warning comes back on every
run that reads the entry, so remove the cache once with `mime cache clear`. The next run builds it again. On a terminal,
`mime run`, `mime serve` and `mime study run` print that command beside the first such warning. To keep a command out of it meanwhile, `--no-xla-cache`.

## In a script of your own

A global JAX setting is not MIME's to change, so a script that calls the library keeps JAX's defaults. Two lines turn the
cache on, with the same directory, thresholds and limits `mime run` uses, for the length of the block:

<!-- mime-fragment: lines for a script of the reader's own, shown and not run; the functions they call are exercised by tests/env/test_user_xla_cache.py and tests/env/test_run.py -->
```python
from mime.env import user_xla_cache

with user_xla_cache():
    result = experiment.build().run(duration)
```

To keep it on for the whole process instead, point JAX at a directory yourself:

<!-- mime-fragment: lines for a script of the reader's own, shown and not run; the functions they call are exercised by tests/env/test_user_xla_cache.py and tests/env/test_run.py -->
```python
from mime.env import use_xla_cache, user_xla_cache_root

use_xla_cache(user_xla_cache_root())
```

This sets the directory only, with JAX's default of caching a program that took a second or more to compile. Add
`jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)` and
`jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)` for the thresholds above. It is not capped and takes no lock (JAX locks only when it is given a size limit, `jax_compilation_cache_max_size`, with the
`filelock` package): set that limit too if processes of your own share the directory, and call
`mime.env.prune_to_cap(user_xla_cache_root())` to cap the other directories.

## What it does not claim

A run reads the cache's entries instead of compiling, and an entry is chosen by JAX's own key (the program, the jaxlib
release, the platform and XLA's flags). MIME's tests compare the state of a run with the cache off, cold and warm, and
a changed program against its own uncached run; they bound the difference between those runs on the cases they run, and
do not show that every program on every platform is unaffected. The size an experiment adds and the time saved are in
the pull request that introduced it, measured on one machine.
