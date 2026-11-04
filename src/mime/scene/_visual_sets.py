"""A description's meshes: the manifest that lists them (``mime.assets``'s :class:`~mime.assets.Manifest`), found where
the description is kept, never in MIME's own data.

A manifest is looked for, in this order: the pins of the experiment in hand (a loaded one's saved stage names each of
its meshes by its set and sha256, so it rebuilds the same stage whatever is installed now); beside the description in
use (``<description stem>.visuals.json`` or ``visuals.json`` next to it), so choosing a robot brings its recipe; then
every manifest an installed package ships through a ``mime.assets`` entry point. The stage refers to a mesh by its
manifest's name and its file's sha256, and the asset cache is where its converted form lives (``mime.viz``)."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mime.assets import AssetRef, Manifest


def mesh_path(mesh: str) -> str:
    """A description's mesh file name as a manifest lists it (``mime.assets.mesh_path``)."""
    from mime.assets import mesh_path as listed

    return listed(mesh)


def visual_sets() -> Mapping[str, Manifest]:
    """Every manifest the installed packages ship (``mime.assets`` entry points; MIME ships none of its own), by
    name: the visual sets ``mime visuals fetch <name>`` knows without a path."""
    from mime.assets import shared_manifests

    found: dict[str, Manifest] = {}
    for _, manifest in shared_manifests():
        found.setdefault(manifest.name, manifest)
    return MappingProxyType(found)


def manifest_beside(description: Path) -> Manifest | None:
    """The manifest beside a description file, read (once for each version of its file); ``None`` where there is
    none."""
    from mime.assets import beside

    path = beside(description)
    if path is None:
        return None
    seen = path.stat()
    return _read_once(str(path), seen.st_mtime_ns, seen.st_size)


@cache
def _read_once(path: str, mtime_ns: int, size: int) -> Manifest:
    from mime.assets import read_manifest

    del mtime_ns, size  # the cache's key: a changed file is read again
    return read_manifest(Path(path))


@dataclass
class VisualPins:
    """What a loaded experiment's saved stage names of each mesh prim (its set and sha256, or neither), and the
    manifests its lock holds by name: the pins its rebuilt stage keeps. ``notes`` are the lines that say where the
    manifest found now names another file, for the build report."""

    prims: Mapping[str, tuple[str | None, str | None]]
    manifests: Mapping[str, Manifest]
    notes: list[str] = field(default_factory=list[str])


_pins: ContextVar[VisualPins | None] = ContextVar("mime_visual_pins", default=None)


@contextmanager
def pinned_visuals(pins: VisualPins) -> Generator[VisualPins]:
    """Inside this block the stage draws each pinned mesh prim as its saved stage named it."""
    token = _pins.set(pins)
    try:
        yield pins
    finally:
        _pins.reset(token)


def active_visual_pins() -> VisualPins | None:
    return _pins.get()


def mesh_source(mesh: str, description: Path | None = None) -> tuple[Manifest, AssetRef] | None:
    """The manifest and record a description's mesh is: from the manifest beside ``description`` where there is one,
    else from an installed package's manifest; ``None`` where none lists it."""
    held = manifest_beside(description) if description is not None else None
    for manifest in ([held] if held is not None else []) + list(visual_sets().values()):
        found = manifest.file(mesh)
        if found is not None:
            return manifest, found[1]
    return None


__all__ = ["VisualPins", "active_visual_pins", "manifest_beside", "mesh_path", "mesh_source", "pinned_visuals",
           "visual_sets"]
