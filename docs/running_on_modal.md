# Running on Modal

[Modal](https://modal.com) runs a Python function in a container on a GPU or CPU it rents by the second. Its value is
fast starts: a container whose image is already built starts in seconds. MIME's value is reproducibility: a saved
experiment reruns to the same result. This page is how MIME keeps both.

## The idea

**The environment is built once per MIME release, Python series and variant, never per experiment, and every input to
it is pinned by hash or digest.** Modal caches an image by its definition. The definition depends only on the MIME
release, the variant (`cpu` or `cuda12`) and the Python series, so a student waits for one build per release, and
every later run starts from the cache. Because every input is hashed, that one build holds the same files in every
workspace; the workspace's image ID, which differs, is recorded as information and never compared (below). Anything
that varies per experiment travels beside the image, never inside it:

- the **experiment folder** is mounted when a container starts, so a new experiment rebuilds nothing;
- its **assets** live on a Volume, fetched there once by `mime assets fetch` inside Modal, at the datacenter's
  bandwidth; a run then sets `MIME_OFFLINE`, so it never downloads;
- **XLA's compilation cache** lives on a second Volume, one directory per jaxlib release, platform and, on a GPU, its
  compute capability (`sm_80` for the 40 GB and the 80 GB A100 alike), on a CPU the hash of its flags.

The usual way to reproducibility, an image built for each run, loses the fast starts. Here the image never changes
between runs, so the starts stay fast, and the hashes keep it reproducible.

## Running an experiment

```sh
mime modal fl9_water/ --gpu A100-40GB        # writes fl9_water-modal/modal_app.py beside the folder
modal run fl9_water-modal/modal_app.py::main --duration 5
modal run fl9_water-modal/modal_app.py::main --duration 5 --record swim_speed:umr.speed --every 0.01
modal run fl9_water-modal/modal_app.py::verify
modal volume get mime-runs fl9_water/<run id> .
```

A run's folder is read back with `mime.results.RunResult.read("<run id>")`, where `modal volume get` put it, or
inside a Modal function where the Volume is mounted: the same accessors as the run's own result, each observation's
outputs by name with their units and the flags raised over them. Its `outputs.json` says what every array of
`outputs.npz` is, so the folder is readable without MIME. `--record` names the per-step outputs to keep as series,
separated by commas, and `--every` their spacing in seconds; the outputs the graph reduces are always there.

`mime modal` needs no Modal account and never imports `modal`; `modal run` needs both. The script's own
docstring lists its entrypoints. A run first calls `mime env check`, which refuses an environment other than the one
asked for, naming every difference, and applies the precision the experiment's lock records; then `mime run`, which
writes the outputs, a record of the run and the run's own saved form to the `mime-runs` Volume. `::verify` reruns the
experiment's reference steps there, and `::verify --run <id>` a run saved earlier.

Options: `--gpu` takes an exact type only (`A100-40GB`, `A100-80GB`, `H100!` and the others in
`mime.env.EXACT_GPUS`); a name Modal may serve with another model of GPU (`A100`, `H100`, `B200+`) is refused. For
float64, the A100s, the H100, the H200 and the B200 compute at half their float32 rate by NVIDIA's tables; the others,
the B300 among them, at a small fraction of it. `--deterministic` turns XLA's deterministic GPU mode on; it is off by
default, optional, and the run's lock records whether it was on (the measurement below could not show it is needed).
`--python` picks the image's Python series, by default the lock's.

**A GPU run needs a lock saved at the highest matmul precision**, as float32 physics is corrupted by reduced-precision
(TF32) matmuls: save the experiment with `JAX_DEFAULT_MATMUL_PRECISION=highest` (and `JAX_ENABLE_X64=1` for float64).
`mime modal --gpu` refuses a lock that records anything else (`env.precision`), before anything is written, and so
does `mime env check` on the GPU; a CPU run applies whatever the lock records.

**An A100 request is met by any A100.** Modal's GPU guide (https://modal.com/docs/guide/gpu, read 2026-10-29) offers
one way to stop an upgrade, `H100!`, and says of it only that it avoids the upgrade to an H200; it names no such suffix
for `A100-40GB`, so none is used. Measured, `A100-40GB` was served by an 80 GB card in 5 of 12 fresh containers, so
the check accepts either A100 (the 40 GB SXM, the 80 GB SXM and the 80 GB PCIe card) for either A100 request, and the
lock records the device kind actually served. A claim of bits compares that kind: a run on the other card is another
environment. The upgrade cost nothing in the one bill read: it listed an `A100-40GB` line only, none for the 80 GB
(observed on one bill, 2026-10-29).

## The image

Its layers, in the order that changes least often first:

1. **the base**, `python:<series>-slim-<debian>`, pinned by digest (`mime/data/env/bases.json`);
2. **the variant's pins**, MIME's requirements file copied byte for byte, installed by a pinned uv with
   `--require-hashes --no-deps --no-build`: every package by version and hash, nothing built from source;
3. **MIME's wheel**, by version and its hash on PyPI. Before a release, the script builds a wheel of the checkout
   MIME is installed from and adds it by its hash, so a change of code rebuilds only this small layer;
4. **the lab's distributions** the lock names besides MIME: by hash when the lab gives a hashed file
   (`--lab-requirements`), by version otherwise, and the run records which.

The script pins Modal's image builder version and records it. When Modal stops supporting that version, the script
fails at start with Modal's own error; write it again with a newer MIME.

## What is pinned, and what is only recorded

Pinned: the Python packages (by hash), Python's patch version and the OS (by the base's digest), the CUDA libraries
(by hash, through the `cuda12` extra), uv (by the digest of its image, `mime/data/env/uv.json`: Modal copies it from
`ghcr.io/astral-sh/uv:<release>@sha256:...`), Modal's image builder, and the GPU type. The uv digest is new and has not
yet been run on Modal; if Modal refuses the `<release>@<digest>` reference, the run fails at the build, naming it.

What no pin holds, the lock records, and a load compares: the GPU's device kind and its driver, CUDA and cuDNN as
JAX reports them; the CPU's model and a hash of its flags; the matmul precision; `XLA_FLAGS` and whether they turn the
deterministic mode on; and the runner's own record, which MIME stores and compares and never reads. Here that record
holds what decides the results: the builder version, the base's digest, the hash of the requirements file, uv's
release and digest, the variant, the GPU requested, and for a dev wheel its hash. What does not decide them, the
runner's **information**, is stored and never compared: Modal's client version and the workspace's image ID (which
differs between workspaces for one build).

Two runs are **the same environment**, for a claim of bits, only when everything compared is equal: the pins above (by
the record), the device kind and driver, the CPU model and flags' hash, the precision, `XLA_FLAGS` and the
deterministic mode. A field of a device that either run could not read (a driver `nvidia-smi` did not give) is never
equal to anything. A run's saved form records its own environment, so a later run of it in the same environment is
checked bit for bit, and anywhere else within the lock's tolerances. An experiment saved on a laptop is always another
environment on Modal. A "within tolerances" verdict from `::verify --run` can come from a different CPU host or driver
alone, with no numerical difference: it is a refusal to claim bits, not evidence of a change.

`mime env check` also refuses a module Python imports from outside this environment's site-packages (a `PYTHONPATH`
entry, the user site), whatever the distributions installed say (`env.shadow`). It holds MIME to the version of the wheel
the run pinned (`--mime-version`, which the script passes), not to the installed copy's own metadata; the wheel's
identity is held at install by its hash.

## What was measured

The maintainer ran the measurement plan on Modal on 2026-10-29 (client 1.6.3), with a probe that installs this recipe's
pins; it did not run a script `mime modal` generated, or the uv digest. The raw data is the maintainer's (a report, a results
table and the run's logs). These are observations of one workspace over about 40 minutes, not
guarantees.

| | Observed |
| --- | --- |
| Base by digest | **Accepted**, on the CPU and the GPU, as `FROM python:3.14-slim-trixie@sha256:cd89c30e...`; the tag fallback was not needed |
| First build | 48 s (CPU, the nearest to a clean first build); 19 s (GPU, which reused the CPU's layers) |
| Cache hit | 2 to 3 s |
| Rebuild with the cache ignored | 20 s (CPU), 48 s (GPU) |
| Fresh container, outside it (scheduling and queueing) | median 5.4 s (CPU, 8 starts, 3.6 to 12.5 s); median 7.5 s (GPU, 12 starts, 3.6 to 13.0 s in nine of them) |
| Fresh container, inside it | about 1.5 s (CPU) and 2 s (GPU) of imports and backend start |
| GPU waits | **3 of 12 fresh GPU starts waited 35, 90 and 296 s outside the container**; the cause is not known |
| Controls | the planted pin was refused and the missing GPU refused, each in 2 of 2 runs (a small count) |
| Bits | the BEM-like program's hash was identical in all 14 GPU runs, across three device kinds (`A100-SXM4-40GB`, `A100-SXM4-80GB`, `A100 80GB PCIe`) and six CPU-flag sets, with the deterministic mode or without; and in all 12 CPU runs, across two CPU-flag sets |
| `A100-40GB` served | by an 80 GB card in 5 of 12 fresh containers |
| XLA cache, CPU | works: compiling a program took 1.3 s uncached and 0.3 s with the cache |
| XLA cache, GPU | works only partly: one program's compile fell from 3 to 5 s to under 1.2 s, another stayed at 2 to 3 s, and the cache grew by 9 or 10 entries in every fresh container; the cause was not found |
| Asset `mtime` on a Volume | identical in every container, so the cheap check `locate` makes holds on Volumes (about 30 minutes, one region) |
| Hashing an 898 MB asset | 0.8 s (CPU) to 2.9 s (GPU) per start |

What the bits observation does and does not say: **one dense-assembly-plus-LU program, at one size, agreed bit for bit
in every run**. No run disagreed, so the measurement cannot show that `--xla_gpu_deterministic_ops` is needed, or what
it costs; the flag stays optional. It is not a promise that any experiment reruns bit for bit on a GPU, and the CPU test
never reached an AVX-512 CPU.

**Cost.** The run cost about $0.5 by the maintainer's estimate from run times; one bill read (`A100-40GB` line only)
was $0.1785 for the GPU. To read what a run cost: `modal billing report --for "this week" --show-resources --json`, and
the rates with `modal billing rates --json`.

**Not measured:** a GHCR-hosted image, memory snapshots, the generated script itself, the uv digest on Modal, why the
GPU's cache keeps adding entries, whether the entries of one A100 kind are hits for the other under one `sm_80`
directory (JAX's own key decides, and still holds the accelerator's configuration), what the long GPU waits were, the
first-ever build in a workspace that never pulled the base, and an AVX-512 CPU.

Prebuilt images on GitHub's container registry (the same recipe, built in CI) and memory snapshots wait for their own
measurements.
