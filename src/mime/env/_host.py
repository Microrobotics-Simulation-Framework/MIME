"""What a run cannot pin, and so records: the GPU and the versions of its driver and libraries, the CPU, XLA's flags
and its deterministic mode, the matmul precision, and the runner's own record.

Two runs are the same environment for a claim of bits only when these are equal too; the lock records them, and a load
compares them with everything else it records, except the runner's information, which it only stores."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

import jax
from typing_extensions import TypedDict

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._maddening import devices

RUNNER_VARIABLE = "MIME_RUNNER"
RUNNER_INFO_VARIABLE = "MIME_RUNNER_INFO"
XLA_FLAGS_VARIABLE = "XLA_FLAGS"
DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops"
#: The bounds of a runner's record: a few short strings that name where a run was made, never a log.
RUNNER_LIMITS = (32, 256)


class GpuRecord(TypedDict, closed=True):
    """The GPU a run used: JAX's ``device_kind``; the driver's release, as ``nvidia-smi`` reports it (as
    ``jax.print_environment_info`` shows it), ``None`` where it cannot be read; and the CUDA driver's, the CUDA
    runtime's and cuDNN's versions as JAX's CUDA plugin reports them, in CUDA's integer form (``12090`` is 12.9)."""

    device_kind: str
    driver: str | None
    cuda_driver: int | None
    cuda_runtime: int | None
    cudnn: int | None


class CpuRecord(TypedDict, closed=True):
    """The CPU a run used: its model name, and the SHA-256 of its feature flags, sorted, which XLA's CPU code
    generation follows and the compilation cache's key does not hold. ``None`` where the system does not say."""

    model: str | None
    flags_sha256: str | None


@cache
def _driver_release() -> str | None:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    releases = sorted({line.strip() for line in done.stdout.splitlines() if line.strip()})
    return ", ".join(releases) if done.returncode == 0 and releases else None


def _cuda_versions() -> tuple[int | None, int | None, int | None]:
    from jax._src.lib import cuda_versions

    held = cast(Any, cuda_versions)
    if held is None:
        return None, None, None

    def read(name: str) -> int | None:
        try:
            return int(getattr(held, name)())
        except Exception:  # noqa: BLE001  (a library the plugin cannot load reports nothing, which is recorded)
            return None

    return read("cuda_driver_get_version"), read("cuda_runtime_get_version"), read("cudnn_get_version")


def gpu_record() -> GpuRecord | None:
    """The GPU JAX computes on, or ``None`` when its default backend is not a GPU."""
    if jax.default_backend() != "gpu":
        return None
    driver, runtime, cudnn = _cuda_versions()
    return {"device_kind": devices()[0][1], "driver": _driver_release(), "cuda_driver": driver,
            "cuda_runtime": runtime, "cudnn": cudnn}


@cache
def cpu_record() -> CpuRecord:
    """The CPU's model and the hash of its flags, from ``/proc/cpuinfo`` on Linux; elsewhere the model Python reports
    and no flags."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            lines = file.read().splitlines()
    except OSError:
        return {"model": platform.processor() or None, "flags_sha256": None}

    def first(key: str) -> str | None:
        return next((line.split(":", 1)[1].strip() for line in lines
                     if line.split(":", 1)[0].strip() == key and ":" in line), None)

    flags = first("flags") or first("Features")
    model = first("model name") or first("Processor") or platform.processor() or None
    digest = hashlib.sha256(" ".join(sorted(flags.split())).encode()).hexdigest() if flags is not None else None
    return {"model": model, "flags_sha256": digest}


def usable_cpus() -> int | None:
    """How many CPUs this process may run on: its affinity where the system has one (``os.sched_getaffinity``, Linux),
    and otherwise the CPUs Python counts; ``None`` where neither says. XLA's CPU backend sizes its thread pools by the
    affinity when JAX first computes, and partitions a program's work by it, so one program can compute other bits
    under another count (measured: the tube demo's 40 states, on 24 CPUs and on 2). Read when called: an affinity
    changed after JAX's first computation is not the one XLA uses."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count()


@cache
def _lapack_openblas() -> ctypes.CDLL | None:
    """The OpenBLAS that SciPy ships, which is the one JAX's CPU backend runs its LAPACK routines on (``jaxlib`` takes
    them from ``scipy.linalg.cython_lapack``), found beside SciPy and opened by path: the loader returns the library
    SciPy already holds, so its thread pool is the one the compiled steps use. ``None`` where SciPy ships none (macOS
    on Arm uses the system's Accelerate) or the library cannot be opened."""
    spec = importlib.util.find_spec("scipy")
    if spec is None or spec.origin is None:
        return None
    package = Path(spec.origin).parent
    for folder in (package.parent / f"{spec.name}.libs", package / ".dylibs"):
        for path in sorted(folder.glob("*scipy_openblas*")):
            try:
                return ctypes.CDLL(str(path))
            except OSError:
                continue
    return None


def blas_threads() -> int | None:
    """How many threads the BLAS under JAX's CPU LAPACK kernels uses: the thread pool of the OpenBLAS SciPy ships,
    which ``OPENBLAS_NUM_THREADS`` sets (else ``GOTO_NUM_THREADS``, else ``OMP_NUM_THREADS``) and which otherwise is the
    affinity's CPU count [measured]. A routine that large (the MRS drag's solves) splits its work over them, and the
    sums it makes in another order give other last bits: the MRS demos' states differ between one thread and 24 here.
    NumPy ships its own OpenBLAS with its own pool, which this does not read (see MIME-ANO-140). ``None`` where the pool
    cannot be read. Read when called: ``threadpoolctl`` or the library's own setter can change it."""
    held = _lapack_openblas()
    if held is None:
        return None
    try:
        count = int(held.scipy_openblas_get_num_threads())
    except (AttributeError, OSError):
        return None
    return count if count > 0 else None


def xla_flags() -> str:
    return os.environ.get(XLA_FLAGS_VARIABLE, "")


def deterministic(flags: str) -> bool:
    """Whether XLA's flags turn its deterministic GPU mode on: the last ``--xla_gpu_deterministic_ops`` decides, bare
    or ``=true`` (or ``1``) for on."""
    on = False
    for word in flags.split():
        name, _, value = word.partition("=")
        if name == DETERMINISTIC_FLAG:
            on = value.lower() in ("", "true", "1")
    return on


def with_deterministic(flags: str, on: bool) -> str:
    """``flags`` with the deterministic mode set ``on`` or off, every other flag kept in its order."""
    kept = [word for word in flags.split() if word.partition("=")[0] != DETERMINISTIC_FLAG]
    return " ".join([*kept, f"{DETERMINISTIC_FLAG}={'true' if on else 'false'}"])


#: The line ``--deterministic`` (and :func:`deterministic_mode`) says on the CPU, where it changes nothing.
CPU_DETERMINISTIC = ("--deterministic changes nothing on the CPU: XLA's CPU programs repeat bit for bit at the counts the "
                     "lock records, with it or without it")


def deterministic_mode() -> bool:
    """Turn XLA's deterministic GPU mode on for this process, as ``--deterministic`` does for ``mime run`` and ``mime
    verify``, and say whether it is on. XLA reads ``XLA_FLAGS`` once, when JAX's backend starts (its first
    ``jax.devices()`` or computation), and never again [measured: a flag set after the backend started took no effect,
    one set after ``import jax`` and ``import mime`` did], so this sets the flag and starts the backend. On a GPU the
    mode is on, and the lock of what this process saves records it. On the CPU it changes nothing (:data:`CPU_DETERMINISTIC`):
    the flag is taken off again, so the lock records the flags as they were, and it returns ``False``. Once the backend
    has started, the flag would reach the lock and not XLA, so it is refused (``env.deterministic-late``) unless the
    flags already turn the mode on."""
    from jax._src import xla_bridge

    flags = os.environ.get(XLA_FLAGS_VARIABLE)
    if deterministic(flags or ""):
        return jax.default_backend() == "gpu"
    if xla_bridge.backends_are_initialized():
        raise BuildRefused([Refusal(
            "env.deterministic-late", RefusalClass.USER,
            "XLA's deterministic GPU mode was asked for after JAX's backend started, and XLA reads its flags only when "
            "the backend starts: the mode would be recorded and not used",
            f"call mime.env.deterministic_mode() before anything computes (before the first build or jax call), set "
            f"{XLA_FLAGS_VARIABLE}={DETERMINISTIC_FLAG}=true before Python starts, or run from the command line, "
            f"`mime run <folder> --deterministic`")])
    os.environ[XLA_FLAGS_VARIABLE] = with_deterministic(flags or "", True)
    if jax.default_backend() == "gpu":
        return True
    if flags is None:
        del os.environ[XLA_FLAGS_VARIABLE]
    else:
        os.environ[XLA_FLAGS_VARIABLE] = flags
    return False


def matmul_precision() -> str | None:
    return cast(Any, jax.config).jax_default_matmul_precision


def runner_record(environ: Mapping[str, str] | None = None, *, variable: str = RUNNER_VARIABLE) -> dict[str, str]:
    """The runner's own record, from ``MIME_RUNNER``: a JSON object of strings, which MIME stores and compares but
    never reads. Unset or empty is ``{}``; anything else is refused as ``env.runner``. The runner's information
    (``MIME_RUNNER_INFO``, :func:`runner_info`) has the same form and is stored but never compared."""
    raw = (os.environ if environ is None else environ).get(variable, "")
    if not raw.strip():
        return {}
    try:
        held = json.loads(raw)
    except json.JSONDecodeError as error:
        held = error
    count, length = RUNNER_LIMITS
    if (not isinstance(held, dict) or len(cast(dict[object, object], held)) > count
            or not all(isinstance(k, str) and isinstance(v, str) and len(k) <= length and len(v) <= length
                       for k, v in cast(dict[object, object], held).items())):
        raise BuildRefused([Refusal(
            "env.runner", RefusalClass.USER,
            f"{variable} is {raw[:200]!r}, which is not a JSON object of at most {count} strings of at most "
            f"{length} characters",
            f"set {variable} to a JSON object of strings that names where the run is made (its runner, its "
            f"image), `{variable}='{{\"runner\": \"...\"}}'`, or unset it")])
    return dict(sorted(cast(dict[str, str], held).items()))


def runner_info(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """What a runner records about a run that does not decide its results (a workspace's image ID, a client's
    version), from ``MIME_RUNNER_INFO``, in the form of :func:`runner_record`. MIME stores it and a load never compares
    it: two runs that differ only here are the same environment."""
    return runner_record(environ, variable=RUNNER_INFO_VARIABLE)


__all__ = ["CPU_DETERMINISTIC", "DETERMINISTIC_FLAG", "RUNNER_INFO_VARIABLE", "RUNNER_VARIABLE", "XLA_FLAGS_VARIABLE",
           "CpuRecord", "GpuRecord", "blas_threads", "cpu_record", "deterministic", "deterministic_mode", "gpu_record",
           "matmul_precision", "runner_info",
           "runner_record", "usable_cpus", "with_deterministic", "xla_flags"]
