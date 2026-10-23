"""The scene a script declares Python-first: its up, its length unit, and the asset layers it references. Declaring
reads no file; the build reads, pins and composes them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from mime.scene._frame import unit_vector, up_axis_problem
from mime.scene._geometry import Axis
from mime.things import PrimPath


@dataclass(frozen=True)
class Asset:
    """A USD layer on disk (a robot), pinned by the SHA-256 of its bytes. With ``sha256`` given the build refuses a
    file whose bytes differ; without it the build pins the file as it finds it, and the saved form records the pin."""

    path: Path
    sha256: str | None = None

    def read(self) -> bytes:
        return self.path.read_bytes()


def asset(path: str | PathLike[str], *, sha256: str | None = None) -> Asset:
    return Asset(Path(path), sha256)


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
        """Reference ``layer`` (an :func:`asset`, or a path) at the prim path ``at``; returns that prim."""
        held = layer if isinstance(layer, Asset) else asset(layer)
        self._references.append(Reference(held, at))
        return PrimPath(at)

    def __getitem__(self, path: str) -> PrimPath:
        return PrimPath(path)
