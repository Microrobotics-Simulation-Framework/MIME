# `mime.env`

**The environment a run is made in: pinned where it can be, recorded where it cannot, and checked before the run.**

## What is here

- **The pins**, in `mime.data`'s `env/` folder, shipped in the wheel:
  - `requirements-cpu.txt` and `requirements-cuda12.txt`: `uv export`'s output from MIME's `uv.lock`, byte for byte,
    one file per variant. Every distribution a variant installs besides MIME is there with its version, its
    environment marker and its hashes; the `cuda12` variant adds JAX's CUDA plugin, its PJRT and every `nvidia-*`
    library, locked through the `cuda12` extra. `EXPORT_ARGUMENTS` holds the command, and a test regenerates both
    files from the lock and compares their bytes, so a lock changed without them fails.
  - `bases.json`: for each Python series MIME supports, the base image (`python:<series>-slim-<debian>`) pinned by the
    digest of its index, with its linux/amd64 manifest's digest, its Python version, its Debian release and point
    version, its glibc and the distributions it ships. `scripts/bump_bases.py` reads them from the image itself; a
    pull request bumps the file, never a build.
- **`check`**, which `mime env check` runs before a run, in the run's process:
  - it compares the distributions installed in this interpreter's own site-packages (not whatever a runner appends
    to `sys.path`) with the variant's pins whose markers hold here, MIME at its own version, the lab's pins and, with
    `--base`, the base image's own; each one missing, at another version or pinned by nothing is refused, naming it;
  - with `--base`, it compares Python's version, Debian's release and version, and glibc with the base image's;
  - it compares JAX's devices with the request: an exact GPU type, or the CPU. A GPU requested and a CPU seen is the
    fallback JAX makes without a word, and is refused as one;
  - then it applies the precision the experiment's lock records (float64 or not), the highest matmul precision on a
    GPU, and, with `--xla-cache`, XLA's compilation cache under its directory.
- **What the lock records of what no pin holds** (`gpu_record`, `cpu_record`, `runner_record`, `xla_flags`,
  `deterministic`): see [`mime.saved`](../saved/README.md).
- **XLA's cache directory**: `<root>/jaxlib-<release>/<platform>/<device>`, the device being the GPU's kind or, on a
  CPU, the first hex digits of the hash of its flags. JAX's cache key holds no CPU feature while XLA compiles for the
  host's, so an entry is never read on another CPU. Without `--xla-cache` there is no persistent cache.
- **The GPU types** (`EXACT_GPUS`), as Modal names them. A name Modal may serve with another model (`A100`, `H100`,
  `B200+`) is refused with the exact names to use instead.
- **`modal_script`**, which `mime modal` runs: the script that runs a saved experiment on Modal. See
  [Running on Modal](../../../docs/running_on_modal.md).

## Boundaries

- Nothing in MIME imports `modal`. The script `mime modal` writes is its only importer, and a test imports `mime`
  and every submodule in a fresh process and finds no `modal` module.
- `check` and the generator read a lock as JSON, never by loading the experiment, so a folder whose entries are not
  installed can still be checked against and written for.

## Refusals

`env.missing`, `env.pin` and `env.unpinned` (the distributions), `env.python` and `env.os` (the base image),
`env.device` (the devices), `env.gpu-type` (a loose or unknown GPU type), `env.variant`, `env.lock` (no precision to
apply), `env.runner` (a `MIME_RUNNER` that is not a JSON object of strings), and `modal.wheel` (no wheel of this MIME
on PyPI).

## Status

Landed in dev1 with the Modal template. Prebuilt images on GitHub's container registry and memory snapshots come
later.

Phases and releases are explained in the [package manifest](../README.md#status).
