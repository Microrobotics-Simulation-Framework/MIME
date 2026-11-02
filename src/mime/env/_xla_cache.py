"""Where XLA's persistent compilation cache is kept: one directory per jaxlib release, platform and what decides the
code compiled for it.

JAX's cache key holds the jaxlib version, the platform, the accelerator's configuration and XLA's flags, but no CPU
feature: XLA compiles for the host's instruction set, so an entry written on one CPU could be read on another. The
directory is therefore split by the hash of the CPU's flags on a CPU, and on a GPU by its compute capability (``sm_80``
for the 40 GB and the 80 GB A100 alike), not by the device kind's string, which Modal varies under one request. JAX's
own key still decides whether an entry is a hit, so a directory shared by two device kinds is never a wrong hit; whether
the entries of one kind are hits for the other is JAX's, and was not measured.

``mime run`` and ``mime serve`` keep the cache in the user's cache directory (:func:`user_xla_cache`), capped in size;
``mime env check --xla-cache`` and the Modal script name their own directory."""

from __future__ import annotations

import contextlib
import importlib.metadata
import re
import shutil
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, cast

import jax
import platformdirs

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


#: What the user's cache may hold, in bytes, before the oldest entries go at the start of the next command.
USER_CACHE_CAP_BYTES = 2 * 1024**3
#: The thresholds the cache is written with (CI's): every program is stored, however quick to compile or small.
_MIN_COMPILE_SECONDS, _MIN_ENTRY_BYTES = 0, -1
_SETTINGS = ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
             "jax_persistent_cache_min_entry_size_bytes")


def user_xla_cache_root() -> Path:
    """Where ``mime run`` and ``mime serve`` keep XLA's cache: ``xla`` in the user's cache directory."""
    return Path(platformdirs.user_cache_dir("mime")) / "xla"


def _files(root: Path) -> list[tuple[float, int, Path]]:
    """Every file under ``root`` as its modification time, its size and its path; one that goes while the directory is
    read (another process pruning) is not there."""
    found: list[tuple[float, int, Path]] = []
    for path in root.rglob("*"):
        try:
            if path.is_file():
                held = path.stat()
                found.append((held.st_mtime, held.st_size, path))
        except OSError:
            continue
    return found


def prune_to_cap(root: Path, cap: int = USER_CACHE_CAP_BYTES) -> int:
    """Delete the oldest files under ``root`` (by when they were written; a read does not touch an entry) until what
    is left is at most ``cap`` bytes; the bytes deleted. A file another process deleted first costs nothing."""
    found = sorted(_files(root))
    total = sum(size for _, size, _ in found)
    freed = 0
    for _, size, path in found:
        if total <= cap:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue
        total -= size
        freed += size
    return freed


def clear_user_xla_cache(root: Path | None = None) -> int:
    """Delete the user's cache; the bytes it held. A cache that is not there holds none."""
    root = user_xla_cache_root() if root is None else root
    held = sum(size for _, size, _ in _files(root)) if root.exists() else 0
    shutil.rmtree(root, ignore_errors=True)
    return held


def _restore(held: dict[str, Any]) -> None:
    for name, value in held.items():
        if name != "jax_compilation_cache_dir":
            cast(Any, jax.config).update(name, value)
    point_xla_cache(held["jax_compilation_cache_dir"])


def _to_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@contextlib.contextmanager
def user_xla_cache(*, root: Path | None = None, cap: int = USER_CACHE_CAP_BYTES,
                   say: Callable[[str], None] = _to_stderr) -> Generator[Path | None]:
    """XLA's persistent compilation cache in the user's cache directory for the length of the block, with the
    thresholds CI uses, and JAX's settings as they were afterwards. The oldest entries are deleted first if the cache
    holds more than ``cap`` bytes. Yields the directory in use, or ``None`` when nothing was changed: JAX's cache
    directory was already set (the user's, or CI's, is not overridden), or the cache cannot be written (a read-only
    home, a full disk), which ``say`` is told once and which costs the block only the cache."""
    config = cast(Any, jax.config)
    if config.jax_compilation_cache_dir is not None:
        yield None
        return
    held = {name: getattr(config, name) for name in _SETTINGS}
    base = user_xla_cache_root() if root is None else root
    chosen: Path | None
    try:
        if base.exists():
            prune_to_cap(base, cap)
        config.update("jax_persistent_cache_min_compile_time_secs", _MIN_COMPILE_SECONDS)
        config.update("jax_persistent_cache_min_entry_size_bytes", _MIN_ENTRY_BYTES)
        chosen = use_xla_cache(base)
        probe = chosen / ".mime-write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        say(f"mime: XLA's compilation cache under {base} cannot be written ({error.strerror or error}); compiling "
            f"without it (--no-xla-cache leaves it out)")
        chosen = None
        _restore(held)
    try:
        yield chosen
    finally:
        if chosen is not None:
            _restore(held)


__all__ = ["USER_CACHE_CAP_BYTES", "clear_user_xla_cache", "point_xla_cache", "prune_to_cap", "use_xla_cache",
           "user_xla_cache", "user_xla_cache_root", "xla_cache_dir"]
