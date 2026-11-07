"""A member's content key (#183, item 7): what a study folder reuses a finished member by. The key holds everything the
member's result is computed from, so that nothing stale is reused and nothing is rerun that need not be:

- the member's point (each axis's path and value, in SI with its unit), as it is overridden;
- the group's structure key (``mime.build._key``: the graph with its parameter values removed, its static data and the
  run's shape);
- the base experiment's input hash;
- the run (its length, what it records, how often);
- the source hashes of what computes it: the bytes of the module of every class its nodes and its entries are made of
  (the MRO's, past the standard library, JAX and NumPy), and of its controller's law; and the versions of MIME, MADDENING
  and JAX, and the precision.

A different value of any of them is a different key, and the member is run again."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mime.results._study_folder import encode_value

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment

#: Packages whose source is not hashed: the standard library's, and the numerical libraries the versions stand for.
NOT_HASHED = frozenset({"builtins", "abc", "typing", "typing_extensions", "dataclasses", "enum", "jax", "jaxlib",
                        "numpy", "scipy", "jaxtyping", "collections", "functools", "types"})


@cache
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_digest(name: str) -> str | None:
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None)
    if not isinstance(file, str) or not Path(file).is_file():
        return None
    stat = Path(file).stat()
    return _file_digest(file, stat.st_mtime_ns, stat.st_size)


def source_hashes(built: BuiltExperiment) -> dict[str, str]:
    """The sha256 of the module of every class that computes ``built``'s members, by module name: its nodes' classes
    (each of their bases too), its entries' classes, and its controller's law's. A module with no file is left out."""
    classes: set[type] = set()
    graph = built.graph
    for name in graph.node_names:
        classes.update(type(graph.get_node(name)).__mro__)
    for plan in built.plans.entries:
        classes.update(type(plan.entry).__mro__)
        classes.update(plan.registered.cls.__mro__)
    if built.law is not None:
        classes.update(type(built.law).__mro__)
    modules = {cls.__module__ for cls in classes if cls.__module__.partition(".")[0] not in NOT_HASHED}
    found = {name: digest for name in sorted(modules) if (digest := _module_digest(name)) is not None}
    return found


def versions() -> dict[str, str]:
    """The versions the numbers depend on besides the sources: MIME, MADDENING, JAX, and the precision in force."""
    import importlib.metadata

    import jax

    from mime import __version__

    held = {"mime": __version__, "jax": jax.__version__, "x64": str(bool(jax.config.jax_enable_x64))}
    try:
        held["maddening"] = importlib.metadata.version("maddening")
    except importlib.metadata.PackageNotFoundError:
        held["maddening"] = "not installed"
    return held


def digest_of(sources: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps({"sources": sources, "versions": versions()}, sort_keys=True).encode()).hexdigest()


def content_key(values: Sequence[tuple[str, object]], structure: str, input_hash: str | None, run: Mapping[str, Any],
                sources: str) -> str:
    """The key of one member: its point, its group's structure key, the base's input hash, the run and the sources."""
    held = {"point": [[path, encode_value(value, path)] for path, value in values], "structure": structure,
            "input": input_hash, "run": run, "sources": sources}
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()


__all__ = ["content_key", "digest_of", "source_hashes", "versions"]
