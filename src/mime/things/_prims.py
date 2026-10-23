from __future__ import annotations

from typing import override

#: The root prim of every scene, under which each thing has its prim.
WORLD = "/World"


class PrimPath:
    """A prim of the scene by its USD path (``/World/UMR``), as ``scene["/World/UMR"]`` gives it. A thing bound to
    one is named after the path's last element unless it is given a name. The build checks the path."""

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def leaf(self) -> str:
        return self.path.rstrip("/").rpartition("/")[2]

    def __truediv__(self, child: str) -> PrimPath:
        return PrimPath(f"{self.path.rstrip('/')}/{child}")

    @override
    def __str__(self) -> str:
        return self.path

    @override
    def __repr__(self) -> str:
        return f"PrimPath({self.path!r})"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimPath) and other.path == self.path

    @override
    def __hash__(self) -> int:
        return hash((PrimPath, self.path))
