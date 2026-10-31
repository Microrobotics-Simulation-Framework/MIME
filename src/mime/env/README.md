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
  - `uv.json`: the uv that installs the pins, its image on GitHub's registry pinned by the digest of its index
    (`ghcr.io/astral-sh/uv:<release>@sha256:...`). The script copies uv from the digest alone
    (`COPY --from=ghcr.io/astral-sh/uv@sha256:...`), since Modal's builder refuses a tag with a digest, and records
    the digest it copied.
- **`check`**, which `mime env check` runs before a run, in the run's process:
  - it compares the distributions installed in this interpreter's own site-packages (not whatever a runner appends
    to `sys.path`) with the variant's pins whose markers hold here, MIME at the version of the pinned wheel
    (`--mime-version`; without it MIME is present but held to no version, since its own metadata would be compared with
    itself), the lab's pins and, with `--base`, the base image's own; each one missing, at another version or pinned by
    nothing is refused, naming it;
  - it then reads where Python imports each of those distributions' top-level modules from and refuses any found
    outside this environment's site-packages (a `PYTHONPATH` entry, the user site), naming the folder and the modules
    (`env.shadow`); a distribution installed in place from a checkout is left out;
  - with `--base`, it compares Python's version, Debian's release and version, and glibc with the base image's;
  - it compares JAX's devices with the request: an exact GPU type, or the CPU. A GPU requested and a CPU seen is the
    fallback JAX makes without a word, and is refused as one. Either A100 request is met by any A100, and the report
    and the lock name the kind served;
  - then it applies the precision the experiment's lock records (float64 or not, and its matmul precision), refusing
    a GPU run whose lock records a matmul precision other than the highest (`env.precision`), and, with `--xla-cache`,
    XLA's compilation cache under its directory.
- **What the lock records of what no pin holds** (`gpu_record`, `cpu_record`, `runner_record`, `runner_info`,
  `xla_flags`, `deterministic`): see [`mime.saved`](../saved/README.md).
- **XLA's cache directory**: `<root>/jaxlib-<release>/<platform>/<device>`, the device being the GPU's compute
  capability (`sm_80` for both A100s) or, on a CPU, the first hex digits of the hash of its flags. JAX's cache key holds
  no CPU feature while XLA compiles for the host's, so an entry is never read on another CPU; on a GPU JAX's key still
  decides whether an entry is a hit. Without `--xla-cache` there is no persistent cache.
- **The GPU types** (`EXACT_GPUS`), as Modal names them. A name Modal may serve with another model (`A100`, `H100`,
  `B200+`) is refused with the exact names to use instead. The two A100 names are one family: Modal's guide names no
  way to stop an upgrade but `H100!`, and a measured run saw a request for the smaller A100 served by the larger card (see Running on
  Modal).
- **`modal_script`**, which `mime modal` runs: the script that runs a saved experiment on Modal. See
  [Running on Modal](../../../docs/running_on_modal.md).

## Boundaries

- Nothing in MIME imports `modal`. The script `mime modal` writes is its only importer, and a test imports `mime`
  and every submodule in a fresh process and finds no `modal` module.
- `check` and the generator read a lock as JSON, never by loading the experiment, so a folder whose entries are not
  installed can still be checked against and written for.

## Refusals

`env.missing`, `env.pin` and `env.unpinned` (the distributions), `env.python` and `env.os` (the base image),
`env.shadow` (a module imported from outside the environment), `env.device` (the devices), `env.gpu-type` (a loose or
unknown GPU type), `env.variant`, `env.lock` (no precision to apply), `env.precision` (a GPU run whose lock records a
matmul precision other than the highest), `env.lab` (a lab file that is not pins), `env.runner` (a `MIME_RUNNER` or
`MIME_RUNNER_INFO` that is not a JSON object of strings), and `modal.wheel` (no wheel of this MIME on PyPI).

## Status

Landed in dev1 with the Modal template. Prebuilt images on GitHub's container registry and memory snapshots come
later.

Phases and releases are explained in the [package manifest](../README.md#status).
