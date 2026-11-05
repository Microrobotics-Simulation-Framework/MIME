"""The scene a script declares Python-first: its up, its length unit, and the asset layers it references. Declaring
reads no file; the build reads, pins and composes them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mime._content import same_content
from mime._notebook import in_notebook
from mime.scene._frame import unit_vector, up_axis_problem
from mime.scene._geometry import Axis
from mime.terms import CustomLicence, Terms
from mime.things import PrimPath

if TYPE_CHECKING:
    from mime.assets import AssetRef


@dataclass(frozen=True)
class Asset:
    """A USD layer on disk (a robot), or a file an entry reads (a robot's description), pinned by the SHA-256 of its
    bytes. With ``sha256`` given the build refuses a file whose bytes differ; without it the build pins the file as it
    finds it, and the saved form records the pin. ``terms`` are a third-party file's (``mime.terms.Terms``: its
    licence, credit, role and origin), which the saved form records by the file's path and pin; a file without them is
    the author's own, under the experiment's licence.

    Declared by its record (``asset(AssetRef(...))``: a scan one study uses, a file a lab's package names), the file is
    the record's: its pin is the record's sha256, its terms come from the record (licence, credit, role, and the
    record's locations as its origin), and it is read from the asset cache, never downloaded by a build (``asset.missing``
    names the command that fetches it). ``path`` is then where the cache keeps it under the record's name."""

    path: Path
    sha256: str | None = None
    terms: Terms | None = None
    record: AssetRef | None = None

    def read(self) -> bytes:
        if self.record is not None:
            from mime.assets import named_file

            return named_file(self.record).read_bytes()
        return self.path.read_bytes()


def _record_terms(record: AssetRef) -> Terms:
    """The terms a file declared by its record carries: the record's licence (its custom licence where it names one
    alone), its credit, its role, and its locations as the origin its bytes travel by."""
    licence: str | CustomLicence = record.custom.get(record.licence, record.licence)
    return Terms(licence, record.credit or "", record.role or "results", record.locations)


def asset(path: str | PathLike[str] | AssetRef, *, sha256: str | None = None, terms: Terms | None = None) -> Asset:
    """A file the experiment reads: by its path (pinned as the build finds it, or to ``sha256``), with ``terms`` where
    it is a third party's; or by its record, an :class:`~mime.assets.AssetRef` (``sha256`` and ``terms`` then come
    from the record, and giving either is refused)."""
    from mime.assets import AssetRef as Record

    if isinstance(path, Record):
        if sha256 is not None or terms is not None:
            raise ValueError("a file declared by its record takes its sha256 and its terms from the record: write "
                             "asset(record) alone")
        from mime.assets import named_path

        return Asset(named_path(path), path.sha256, _record_terms(path), path)
    return Asset(Path(path), sha256, terms)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class Reference:
    """An asset layer referenced into the scene at a prim path: the asset's default prim appears there."""

    asset: Asset
    at: str


class Scene:
    """The scene, declared once per experiment: ``Scene.new(up=+Z, units="m")``. ``up`` is the world's up, +Y or +Z,
    from which gravity is derived (``-g * up``), and ``units`` the length unit the stage is written in (MIME computes
    in SI whatever it is). ``reference`` brings an asset layer in at a prim path, and ``scene["/World/UMR"]`` names a
    prim a thing binds. An up along another direction raises ``ValueError`` here; everything else is checked at the
    build."""

    def __init__(self, *, up: Any, units: Any) -> None:
        problem = up_axis_problem(up)
        if problem is not None:
            _, message, action = problem
            raise ValueError(f"{message}; {action}")
        self.declared_up = up
        self.units = units
        self._references: list[Reference] = []

    @classmethod
    def new(cls, *, up: Any, units: Any) -> Scene:
        return cls(up=up, units=units)

    @classmethod
    def open(cls, path: str | PathLike[str]) -> Scene:
        """Stage-first authoring, where the stage is the authority and the document holds only its path: not yet.
        Raises ``NotImplementedError`` naming the Python-first way."""
        raise NotImplementedError(
            f"Scene.open({str(path)!r}) would take the stage as the authority (stage-first authoring), which arrives "
            f"with OpenMaterica's USD support; until then a scene is declared Python-first: Scene.new(up=..., "
            f"units=...), with each asset referenced by scene.reference(asset(path), at=...) and each geometry kind "
            f"declared in Python, the stage written from those declarations")

    @property
    def up(self) -> Any:
        """Up as declared, a signed :class:`Axis` when it is three numbers, so ``-scene.up`` is down."""
        if unit_vector(self.declared_up) is None:
            return self.declared_up
        return Axis(*(float(c) for c in self.declared_up))

    @property
    def references(self) -> tuple[Reference, ...]:
        return tuple(self._references)

    def reference(self, layer: Asset | str | PathLike[str], *, at: str) -> PrimPath:
        """Reference ``layer`` (an :func:`asset`, or a path) at the prim path ``at``; returns that prim. The same asset
        referenced at the same prim again (a notebook's cell run twice) is kept once, with a note. Another asset at a
        prim already referenced is kept beside it, since the scene cannot tell what binds the first one's prims, and
        the build refuses the two (``scene.reference-path``), in a notebook saying a line or cell has probably run twice."""
        from mime.errors._frames import warn_at_script  # the errors' package imports the build, which imports this

        held = layer if isinstance(layer, Asset) else asset(layer)
        reference = Reference(held, at)
        earlier = [other for other in self._references if other.at == at]
        if any(same_content(other, reference) for other in earlier):
            warn_at_script(f"scene.reference(...) references the same asset at {at} again: it is kept once",
                           UserWarning)
            return PrimPath(at)
        self._references.append(reference)
        if earlier and in_notebook():
            warn_at_script(f"scene.reference(...) references another asset at {at}, where one is referenced already: "
                           f"the scene cannot tell what binds the earlier one's prims, so both are kept and the build "
                           f"will refuse them; run the lines or cells again from the one that makes the scene, or run them all",
                           UserWarning)
        return PrimPath(at)

    def __getitem__(self, path: str) -> PrimPath:
        return PrimPath(path)
