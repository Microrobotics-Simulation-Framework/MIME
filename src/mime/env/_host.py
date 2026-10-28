"""What a run cannot pin, and so records: the GPU and the versions of its driver and libraries, the CPU, XLA's flags
and its deterministic mode, the matmul precision, and the runner's own record.

Two runs are the same environment for a claim of bits only when these are equal too; the lock records them, and a load
compares them with everything else it records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from functools import cache
from typing import Any, cast

import jax
from typing_extensions import TypedDict

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._maddening import devices

RUNNER_VARIABLE = "MIME_RUNNER"
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


def matmul_precision() -> str | None:
    return cast(Any, jax.config).jax_default_matmul_precision


def runner_record(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The runner's own record, from ``MIME_RUNNER``: a JSON object of strings, which MIME stores and compares but
    never reads. Unset or empty is ``{}``; anything else is refused as ``env.runner``."""
    raw = (os.environ if environ is None else environ).get(RUNNER_VARIABLE, "")
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
            f"{RUNNER_VARIABLE} is {raw[:200]!r}, which is not a JSON object of at most {count} strings of at most "
            f"{length} characters",
            f"set {RUNNER_VARIABLE} to a JSON object of strings that names where the run is made (its runner, its "
            f"image), or unset it")])
    return dict(sorted(cast(dict[str, str], held).items()))


__all__ = ["DETERMINISTIC_FLAG", "RUNNER_VARIABLE", "XLA_FLAGS_VARIABLE", "CpuRecord", "GpuRecord", "cpu_record",
           "deterministic", "gpu_record", "matmul_precision", "runner_record", "with_deterministic", "xla_flags"]
