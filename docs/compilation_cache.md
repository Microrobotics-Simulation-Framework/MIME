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
- **The cache is capped at 2 GB.** At the start of a command, if the directory holds more, the oldest entries (by when
  they were written; reading one does not renew it) are deleted until it fits. A run can add to a cache at the cap, so
  the directory can pass it by one run's worth until the next command. An entry deleted costs one recompile.
- **`mime cache clear`** deletes the directory and says how much it held.
- **A cache that cannot be written** (a read-only home, a full disk) does not stop the run: the command says so once on
  standard error and compiles without it.
- **What it cannot do.** It holds compiled programs, not tracing: a second run still traces and lowers each program,
  and the first run of an experiment, or of a changed one, pays in full. JAX writes an entry in place, so a
  process killed mid-write can leave a truncated one, which JAX is expected to warn about and recompile (MADDENING's CI
  prunes such entries for that reason; MIME's tests do not exercise this); `mime cache clear` removes it.

## In a script of your own

A global JAX setting is not MIME's to change, so a script that calls the library keeps JAX's defaults. Two lines turn the
cache on, with the same directory, thresholds and cap `mime run` uses, for the length of the block:

```python
from mime.env import user_xla_cache

with user_xla_cache():
    result = experiment.build().run(duration)
```

To keep it on for the whole process instead, point JAX at a directory yourself:

```python
from mime.env import use_xla_cache, user_xla_cache_root

use_xla_cache(user_xla_cache_root())
```

This sets the directory only, with JAX's default of caching a program that took a second or more to compile. Add
`jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)` and
`jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)` for the thresholds above. It is not capped; call
`mime.env.prune_to_cap(user_xla_cache_root())` to cap it.

## What it does not claim

A run reads the cache's entries instead of compiling, and an entry is chosen by JAX's own key (the program, the jaxlib
release, the platform and XLA's flags). MIME's tests compare the state of a run with the cache off, cold and warm, and
a changed program against its own uncached run; they bound the difference between those runs on the cases they run, and
do not show that every program on every platform is unaffected. The size an experiment adds and the time saved are in
the pull request that introduced it, measured on one machine.
