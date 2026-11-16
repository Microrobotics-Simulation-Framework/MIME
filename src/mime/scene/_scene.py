"""The scene a script declares Python-first: its up, its length unit, and the asset layers it references. Declaring
reads no file; the build reads, pins and composes them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any

from mime._content import same_content
from mime._notebook import in_notebook
from mime.scene._frame import unit_vector, up_axis_problem
from mime.scene._geometry import Axis
from mime.things import PrimPath

if TYPE_CHECKING:
    from mime.assets import Asset


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
            f"units=...), with each asset referenced by scene.reference(asset(Path(...)), at=...) and each geometry kind "
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

    def reference(self, layer: Asset | PathLike[str], *, at: str) -> PrimPath:
        """Reference ``layer`` (an :func:`~mime.assets.asset` of a file or a record, or a path) at the prim path ``at``;
        returns that prim. A layer is a file: a record's name is refused (``asset.source``), ``asset(Path(...))`` or
        ``asset(AssetRef(...))`` being the layer. The same asset referenced at the same prim again (a notebook's cell
        run twice) is kept once, with a note. Another asset at a prim already referenced is kept beside it, since the
        scene cannot tell what binds the first one's prims, and the build refuses the two (``scene.reference-path``),
        in a notebook saying a line or cell has probably run twice."""
        from mime.assets import Asset, asset
        from mime.assets._asset import refused_origin
        from mime.errors._frames import warn_at_script  # the errors' package imports the build, which imports this

        held = layer if isinstance(layer, Asset) else asset(layer)
        if held.origin == "named":
            raise refused_origin(held, "a scene's layer", "asset(Path(\"<file>.usda\")) or asset(AssetRef(...))")
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
