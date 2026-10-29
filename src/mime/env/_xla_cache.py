"""Where XLA's persistent compilation cache is kept: one directory per jaxlib release, platform and what decides the
code compiled for it.

JAX's cache key holds the jaxlib version, the platform, the accelerator's configuration and XLA's flags, but no CPU
feature: XLA compiles for the host's instruction set, so an entry written on one CPU could be read on another. The
directory is therefore split by the hash of the CPU's flags on a CPU, and on a GPU by its compute capability (``sm_80``
for the 40 GB and the 80 GB A100 alike), not by the device kind's string, which Modal varies under one request. JAX's
own key still decides whether an entry is a hit, so a directory shared by two device kinds is never a wrong hit; whether
the entries of one kind are hits for the other is JAX's, and was not measured."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Any, cast

import jax

from mime.build._maddening import compute_capability as capability_of_jax
from mime.build._maddening import devices
from mime.env._host import cpu_record


def _segment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unknown"


def _capability(compute_capability: str) -> str:
    """``sm_80`` for ``"8.0"``: the name CUDA gives the architecture."""
    return "sm_" + compute_capability.replace(".", "")


def xla_cache_dir(root: str | Path, *, jaxlib: str, platform: str, device_kind: str,
                  cpu_flags_sha256: str | None, compute_capability: str | None = None) -> Path:
    """``root/jaxlib-<release>/<platform>/<device>``: on a CPU the first sixteen hex digits of its flags' hash
    (``unknown`` where the system does not say); on a GPU its compute capability as CUDA names it (``sm_80``), or, where
    JAX reports none, the device kind."""
    if platform == "cpu":
        device = (cpu_flags_sha256 or "unknown")[:16]
    else:
        device = _capability(compute_capability) if compute_capability else device_kind
    return Path(root) / f"jaxlib-{_segment(jaxlib)}" / _segment(platform) / _segment(device)


def use_xla_cache(root: str | Path) -> Path:
    """Point JAX's persistent compilation cache at this process's directory under ``root``, creating it."""
    platform, kind = devices()[0]
    held = xla_cache_dir(root, jaxlib=importlib.metadata.version("jaxlib"), platform=platform, device_kind=kind,
                         cpu_flags_sha256=cpu_record()["flags_sha256"], compute_capability=capability_of_jax())
    held.mkdir(parents=True, exist_ok=True)
    point_xla_cache(str(held))
    return held


def point_xla_cache(directory: str | None) -> None:
    """Set the cache's directory (``None``: no persistent cache), and have JAX decide afresh at its next compilation
    whether to use it: JAX decides once per process, at the first compilation, and keeps to it otherwise."""
    from jax.experimental.compilation_cache import compilation_cache

    cast(Any, jax.config).update("jax_compilation_cache_dir", directory)
    compilation_cache.reset_cache()


__all__ = ["point_xla_cache", "use_xla_cache", "xla_cache_dir"]
