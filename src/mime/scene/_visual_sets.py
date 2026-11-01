"""The visual sets MIME knows: each a recipe for a description's meshes that MIME does not ship, read from ``mime.data``'s
``visuals/`` folder. A set names its upstream at a pinned commit, each file by the path a description writes it with
and the sha256 of its bytes, its licence, its tier (where its bytes may live, design #198) and its role (presentation:
no result depends on it), and the credit and notice it is shown with. The stage refers to a mesh by its set's name and
its file's sha256, and the asset cache is where its converted form lives (``mime.viz``)."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, cast

VISUALS_FOLDER = "visuals"
VISUAL_SET_FORMAT = "mime.visual_set/1"


@dataclass(frozen=True)
class VisualFile:
    """One file of a visual set: the path a description writes it with, the sha256 and size of its bytes, its git blob
    (which names it in any clone of its upstream), and where it is downloaded from."""

    path: str
    sha256: str
    size: int
    git_blob: str
    location: str


@dataclass(frozen=True)
class VisualSet:
    """A recipe for a description's meshes: where they come from, under what terms, and each file pinned."""

    name: str
    title: str
    upstream: Mapping[str, str]
    licence: str
    licence_url: str
    tier: str
    role: str
    credit: str
    notice: str
    archives: tuple[str, ...]
    archive_checked: str
    files: tuple[VisualFile, ...]

    def file(self, mesh: str) -> VisualFile | None:
        """The file a description's mesh names (``meshes/ar4_mk5/Link_1_Aluminum.STL``, or with its package,
        ``package://<package>/meshes/...``), or ``None``."""
        wanted = mesh_path(mesh)
        return next((held for held in self.files if held.path == wanted), None)

    def text(self) -> str:
        """The credit, licence and notice, as a command prints them before it fetches."""
        return (f"{self.title}\ncredit: {self.credit}\nlicence: {self.licence} ({self.licence_url}); tier "
                f"{self.tier}, role {self.role}\n{self.notice}")


def mesh_path(mesh: str) -> str:
    """A description's mesh file name as a visual set lists it: without a ``package://<package>/`` or ``file://``
    prefix, or a leading ``./``."""
    held = mesh.strip()
    if held.startswith("package://"):
        held = held.removeprefix("package://").partition("/")[2]
    held = held.removeprefix("file://")
    while held.startswith("./"):
        held = held[2:]
    return held


def _read(record: Mapping[str, Any]) -> VisualSet:
    files = tuple(VisualFile(str(f["path"]), str(f["sha256"]), int(f["size"]), str(f["git_blob"]), str(f["location"]))
                  for f in cast(list[Mapping[str, Any]], record["files"]))
    return VisualSet(str(record["name"]), str(record["title"]), MappingProxyType(dict(record["upstream"])),
                     str(record["licence"]), str(record["licence_url"]), str(record["tier"]), str(record["role"]),
                     str(record["credit"]), str(record["notice"]), tuple(map(str, record["archives"])),
                     str(record["archive_checked"]), files)


@cache
def visual_sets() -> Mapping[str, VisualSet]:
    """Every visual set MIME ships a recipe for, by name."""
    found: dict[str, VisualSet] = {}
    folder = importlib.resources.files("mime.data") / VISUALS_FOLDER
    for entry in sorted(folder.iterdir(), key=lambda held: held.name):
        if entry.name.endswith(".json"):
            record = json.loads(entry.read_text(encoding="utf-8"))
            if record.get("format") != VISUAL_SET_FORMAT:
                raise ValueError(f"{entry.name} is not a {VISUAL_SET_FORMAT} record")
            held = _read(record)
            found[held.name] = held
    return MappingProxyType(found)


def mesh_source(mesh: str) -> tuple[VisualSet, VisualFile] | None:
    """The visual set and file a description's mesh is, where a set MIME knows lists it."""
    for held in visual_sets().values():
        found = held.file(mesh)
        if found is not None:
            return held, found
    return None


__all__ = ["VISUALS_FOLDER", "VISUAL_SET_FORMAT", "VisualFile", "VisualSet", "mesh_path", "mesh_source",
           "visual_sets"]
