"""A member's content key (#183, item 7): what a study folder reuses a finished member by. The key holds everything the
member's result is computed from, so that nothing stale is reused:

- the member's point (each axis's path and value, in SI with its unit), as it is overridden;
- the group's structure key (``mime.build._key``: the graph with its parameter values removed, its static data and the
  run's shape);
- the base experiment's input hash (which covers the assets it reads);
- the run (its length, what it records, how often);
- the code: every file of the installed ``mime`` package by its sha256 (bytecode caches aside), the whole tree and not
  only what a member imports, and so too every file of each other package a class of its nodes, entries or law comes
  from (a lab's plugin; a script's own file); and the versions of the packages that stand for the rest of its code
  (MADDENING, OpenMaterica, JAX, jaxlib, NumPy and SciPy), and the precision.

The code's part is coarse on purpose: a change to any file of MIME is a different key, so a code change runs every
member again, whatever it touched. What it cannot see is code with no file (a law defined in a notebook's cell) and a
package outside those named, which a member's classes do not come from."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mime.results._study_folder import encode_value

if TYPE_CHECKING:
    from mime.experiment import BuiltExperiment

#: The distributions whose versions stand for their code in the key.
VERSIONED = ("maddening", "openmaterica", "jax", "jaxlib", "numpy", "scipy")
#: The top-level modules of those distributions (and of the typing helpers their classes are made with), whose classes a
#: member's classes may have among their bases: their versions stand for them, so their files are not hashed.
BY_VERSION = frozenset({"maddening", "materica", "jax", "jaxlib", "numpy", "scipy", "jaxtyping", "typing_extensions"})


@cache
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(path: Path) -> str:
    stat = path.stat()
    return _file_digest(str(path), stat.st_mtime_ns, stat.st_size)


def package_root() -> Path:
    """The installed ``mime`` package's folder: the tree the key hashes."""
    import mime

    return Path(mime.__file__).parent


def _tree(root: Path) -> list[Path]:
    """Every file under ``root``, bytecode caches aside, in a fixed order."""
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and "__pycache__" not in path.relative_to(root).parts and path.suffix != ".pyc")


def _roots(builts: Iterable[BuiltExperiment]) -> dict[str, Path]:
    """What the key hashes, by name: MIME's package, and each other package (or a script's own file) a class of the
    builds' nodes, entries or law comes from, past the standard library and the packages :data:`VERSIONED` stands for."""
    classes: set[type] = set()
    for built in builts:
        graph = built.graph
        for name in graph.node_names:
            classes.update(type(graph.get_node(name)).__mro__)
        for plan in built.plans.entries:
            classes.update(type(plan.entry).__mro__)
            classes.update(plan.registered.cls.__mro__)
        if built.law is not None:
            classes.update(type(built.law).__mro__)
    found = {"mime": package_root()}
    for top in sorted({cls.__module__.partition(".")[0] for cls in classes}):
        if top in found or top in BY_VERSION or top in sys.stdlib_module_names:
            continue
        module = sys.modules.get(top)
        paths = [Path(each) for each in getattr(module, "__path__", ()) or ()]
        file = getattr(module, "__file__", None)
        if paths:
            found.update({top if i == 0 else f"{top}[{i}]": path for i, path in enumerate(paths) if path.is_dir()})
        elif isinstance(file, str) and Path(file).is_file():
            found[top] = Path(file)
    return found


def source_hashes(builts: Iterable[BuiltExperiment] = ()) -> dict[str, str]:
    """The sha256 of every file the key covers, by its package's name and its path in the package
    (``mime/nodes/body/_contact.py``): every file of the installed ``mime`` package, and of each other package a class of
    ``builts``' nodes, entries or law comes from (a module that is one file, by its name)."""
    found: dict[str, str] = {}
    for name, root in _roots(builts).items():
        if root.is_dir():
            found.update({f"{name}/{path.relative_to(root).as_posix()}": _digest(path) for path in _tree(root)})
        else:
            found[name] = _digest(root)
    return found


def versions() -> dict[str, str]:
    """The versions the numbers depend on besides the files hashed: MIME's, those of :data:`VERSIONED`, and the precision
    in force."""
    import jax

    from mime import __version__

    held = {"mime": __version__, "x64": str(bool(jax.config.jax_enable_x64))}
    for name in VERSIONED:
        try:
            held[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            held[name] = "not installed"
    return held


def digest_of(sources: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps({"sources": sources, "versions": versions()}, sort_keys=True).encode()).hexdigest()


def content_key(values: Sequence[tuple[str, object]], structure: str, input_hash: str | None, run: Mapping[str, Any],
                sources: str) -> str:
    """The key of one member: its point, its group's structure key, the base's input hash, the run and the code."""
    held = {"point": [[path, encode_value(value, path)] for path, value in values], "structure": structure,
            "input": input_hash, "run": run, "sources": sources}
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()


__all__ = ["BY_VERSION", "VERSIONED", "content_key", "digest_of", "package_root", "source_hashes", "versions"]
