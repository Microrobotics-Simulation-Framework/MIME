# Running on Modal

[Modal](https://modal.com) runs a Python function in a container on a GPU or CPU it rents by the second. Its value is
fast starts: a container whose image is already built starts in seconds. MIME's value is reproducibility: a saved
experiment reruns to the same result. This page is how MIME keeps both.

## The idea

**The environment is built once per MIME release, Python series and variant, never per experiment, and every input to
it is pinned by hash or digest.** Modal caches an image by its definition. The definition depends only on the MIME
release, the variant (`cpu` or `cuda12`) and the Python series, so a student waits for one build per release, and
every later run starts from the cache. Because every input is hashed, that one build is the same environment in every
workspace. Anything that varies per experiment travels beside the image, never inside it:

- the **experiment folder** is mounted when a container starts, so a new experiment rebuilds nothing;
- its **assets** live on a Volume, fetched there once by `mime assets fetch` inside Modal, at the datacenter's
  bandwidth; a run then sets `MIME_OFFLINE`, so it never downloads;
- **XLA's compilation cache** lives on a second Volume, one directory per jaxlib release, platform and device.

The usual way to reproducibility, an image built for each run, loses the fast starts. Here the image never changes
between runs, so the starts stay fast, and the hashes keep it reproducible.

## Running an experiment

```sh
mime modal fl9_water/ --gpu A100-40GB        # writes fl9_water-modal/modal_app.py beside the folder
modal run fl9_water-modal/modal_app.py::main --duration 5
modal run fl9_water-modal/modal_app.py::verify
modal volume get mime-runs fl9_water/<run id> .
```

`mime modal` needs no Modal account and never imports `modal`; `modal run` needs both. The script's own
docstring lists its entrypoints. A run first calls `mime env check`, which refuses an environment other than the one
asked for, naming every difference, and applies the precision the experiment's lock records; then `mime run`, which
writes the outputs, a record of the run and the run's own saved form to the `mime-runs` Volume. `::verify` reruns the
experiment's reference steps there, and `::verify --run <id>` a run saved earlier.

Options: `--gpu` takes an exact type only (`A100-40GB`, `A100-80GB`, `H100!` and the others in
`mime.env.EXACT_GPUS`); a name Modal may serve with another model of GPU (`A100`, `H100`, `B200+`) is refused. For
float64, the A100s, the H100, the H200 and the B200 compute at half their float32 rate by NVIDIA's tables; the others,
the B300 among them, at a small fraction of it. `--deterministic` turns XLA's deterministic GPU mode on; it is off by default, and the run's lock
records whether it was on. `--python` picks the image's Python series, by default the lock's.

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
(by hash, through the `cuda12` extra), uv, Modal's image builder, and the GPU type.

What no pin holds, the lock records, and a load compares: the GPU's device kind and its driver, CUDA and cuDNN as
JAX reports them; the CPU's model and a hash of its flags; the matmul precision; `XLA_FLAGS` and whether they turn the
deterministic mode on; and the runner's own record (here Modal's client and builder versions, the base, the variant,
the GPU requested and the image's ID), which MIME stores and compares and never reads. Two runs are **the same
environment**, for a claim of bits, only when all of these are equal too. A run's saved form records its own
environment, so a later run of it on the same hardware is checked bit for bit, and anywhere else within the lock's
tolerances. An experiment saved on a laptop is always another environment on Modal.

## What is not measured yet

Everything above about start times is an estimate, and so is the claim that one environment reruns bit for bit on
Modal: XLA does not promise determinism on a GPU without its deterministic mode. A measurement plan is pending, run by
the maintainer before these are claimed: whether Modal's builder takes an image by digest, the first build and later
cold starts for each variant, reads from Volumes, whether the compilation cache is reused across containers, whether
Modal's CPU workers differ in model and in bits, and whether GPU runs agree bit for bit across cold starts with and
without the deterministic mode. Until then, a bit claim on Modal is a hypothesis this page will replace with what was
measured.

Prebuilt images on GitHub's container registry (the same recipe, built in CI) and memory snapshots wait for those
measurements.
