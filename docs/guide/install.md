# Install and environment

How to get MIME running, and the one rule about numerical precision that is not obvious. The platforms
page (`docs/platforms.md`) says what each operating system supports and what a test shows of it, and is not repeated here.

## What it needs

- **Python 3.12 to 3.14.** `pyproject.toml` says 3.12 or later; the classifiers and CI's lanes name 3.12, 3.13 and 3.14.
- **`usd-core`**, which MIME reads and writes the scene's USD stage with. It is a dependency, installed from its wheels with
  the rest.
- **JAX**, as a CPU build by default. For a GPU on Linux add the `cuda12` extra, `mime-engine[cuda12]`, which adds JAX's CUDA
  plugin. There is no GPU build for macOS or native Windows (the platforms page, `docs/platforms.md`).
- **MADDENING and OpenMaterica**, the coupling engine and the materials library, each within the range `pyproject.toml`
  names. MIME's lock (`uv.lock`) pins the exact releases, and the files shipped in the wheel pin every distribution of a run
  by hash.
- **The `verify` extra** for `mime docs build` and MIME's own tests: it adds Hypothesis and MADDENING's verification
  harnesses. A script that builds and runs experiments needs neither.

MIME 0.3 is a pre-release, so pip installs it only when asked to, extras included:

<!-- mime-fragment: an install line that downloads from PyPI, shown and not run by a check -->
```sh
pip install --pre "mime-engine[verify]"
```

A plain `pip install mime-engine` installs the earlier line, which these pages do not describe; `mime --version` says
which you have.

## Checking an environment

`mime env check` holds the environment you are in to MIME's pins before a run: each distribution at its pinned version, no
module imported from outside the environment, the devices JAX sees against the GPU you asked for. A GPU requested while JAX
sees only a CPU is refused, because JAX would otherwise fall back without a word. A checkout installed as CONTRIBUTING
says, `uv sync --locked` with or without `--extra verify`, passes it: the dev group's tools and the `verify` extra are
accepted at the versions MIME's lock pins, and refused at any other.

<!-- mime-fragment: a shell line for an experiment folder the reader has; the command's flags are parsed against the CLI by mime docs check -->
```sh
mime env check my_experiment --variant cpu
```

[The CLI reference](cli.md) lists its flags.

## Assets a run fetches

Large files are named by the hash of their bytes and fetched on request, into one cache directory. A build never
downloads: a file it needs and the cache lacks is refused by name, with the command to fetch it.

- **MIME's own.** A tube experiment whose drag is the MRS rung reads `unit_tube_R0.997`, the table of the unit tube's
  wall, about two hundred megabytes. Fetch it once before the first run:

<!-- mime-fragment: a command that downloads a large file from the network, shown and not run by a check; the command is parsed against the CLI -->
```sh
mime assets fetch unit_tube_R0.997
```

- **An experiment's.** A saved experiment's lock names every asset it uses, so `mime assets fetch ./my_experiment/`
  fetches exactly those, and places each where the folder expects it.
- **A description's visuals.** Some meshes may not be hosted by anyone but their owner (the AR4 arm's, under Annin
  Robotics' licence). Their manifest sits beside the description and names each file at a pinned commit, and
  `mime visuals fetch examples/descriptions/ar4/` downloads them from the owner and converts them on your machine, into the
  cache only. They are for drawing the stage: no result depends on them, and without them the stage draws simple shapes.

`mime assets list` shows each known asset with its size and whether it is cached, `mime assets sources unit_tube_R0.997`
says where it can be fetched from now and under what terms, and `mime assets check` hashes the cache again in full.

**Offline, on a cluster, or behind a lab's mirror.** `MIME_ASSETS_DIR` moves the cache (a shared directory on a cluster);
`MIME_OFFLINE=yes` downloads nothing; a file copied into the cache by hand is accepted by its hash. A lab's own copies go in
the locations file (`MIME_ASSET_LOCATIONS`, else `locations.toml` in your config directory), which is tried first and never
travels with a shared experiment, and an internal server signed by the institution's own certificate authority needs
`MIME_CA_BUNDLE`. [Assets in depth](assets.md) has the rest.

The tutorials in this guide need no fetched asset: the drag surrogate's weights and the material records they use ship with
MIME. [Tutorial A](tutorial_a.md#before-you-run-it) says which part of a tube experiment needs the table.

## Precision follows the dtype of the run

**A run computes in the dtype JAX gives it, float32 by default or float64 under `JAX_ENABLE_X64=1`, and a float32 run holds
only behind declared guards.** MIME's coupled rigs (script A, A0, the hull contact) run and are gated in both dtypes. An
earlier note said that some coupled rigs need float64; it came from one NaN in the earlier line whose cause was never
recorded, and it is withdrawn.

- **The guards of a float32 run:** an integer clock; on a GPU, `JAX_DEFAULT_MATMUL_PRECISION=highest` (in the earlier line a
  run was measurably off on a GPU under reduced-precision (TF32) matmuls); iterative-solve and fixed-point tolerances at or
  above float32's reachable floor, with a stagnation stop; energy terms accumulated in float64; definiteness checked at a
  floor.
- **A rig that needs float64 says so by a declared guard**, not by float64 everywhere, which costs time for nothing where
  the guards hold.
- If you are unsure for a rig, run both dtypes and compare.

A saved experiment's lock records whether it was built in float64 and its matmul precision, and `mime env check <folder>`
applies them. A GPU run whose lock records a matmul precision other than the highest is refused (`env.precision`), because
a reduced precision corrupts float32 physics of small magnitude without a word.
The page on running on Modal (`docs/running_on_modal.md`) says how this holds for a cloud GPU. A controller can fix its own precision:
`arm_tracking` computes in single precision whatever the run's, so a double-precision run with it is mixed, as
[tutorial A0](tutorial_a0.md) says.

Precision is not the only thing a run depends on: the step has limits of its own, which the build report states and
[tutorial A](tutorial_a.md#before-you-run-it) reads.

## Feed every external input

A graph's external inputs come only from a controller's declaration, and a run feeds each one. Code that drives MADDENING
directly, outside `Experiment.build` and `built.run`, must feed each one itself: MADDENING zero-fills an input that nobody
feeds, which is how a graph went to NaN in the earlier line (a zero quaternion). `step({})` is almost
never right.
