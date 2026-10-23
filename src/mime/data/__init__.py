"""Package data shipped in the wheel: small default assets, each pinned by its sha256; calibrations from a2.

An experiment names a packaged asset with ``asset("dejongh_mlp_v2")``, which holds the name and the sha256 the
experiment is written against. Whatever reads the asset's contents reads them through :func:`asset_bytes`, which
refuses a file whose sha256 is not the one asked for, so a saved experiment that recorded the hash refuses, on load, an
installation that ships other contents under the same name. This is the smallest form of the seam the design writes as
``asset(...)``: assets by name from MIME's own package only. Assets fetched by DOI or URL, an experiment folder's own,
and assets that cannot be redistributed come later.
"""

from __future__ import annotations

import hashlib
import importlib.resources
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType


@dataclass(frozen=True)
class Asset:
    """An asset an experiment names: its ``name`` and the ``sha256`` of the contents it was written against. The saved
    form writes both and the lock records them."""

    name: str
    sha256: str


@dataclass(frozen=True)
class PackagedAsset:
    """An asset shipped in ``mime.data``: its file, the sha256 of its contents, and what it is."""

    file: str
    sha256: str
    description: str


#: Every packaged asset, by name.
PACKAGED: Mapping[str, PackagedAsset] = MappingProxyType({
    "dejongh_mlp_v2": PackagedAsset(
        "mlp_cholesky_weights_v2.npz", "ac1fca5c109fa3e2f27520cbd0ee1c0ce5e85ac31a0b1e93e7b8085c68f0a345",
        "the Cholesky MLP drag surrogate's weights and normalisation for de Jongh screw bodies in a tube, version 2 "
        "(trained in the 0.2 line on regularised-Stokeslet BEM resistances with a cylinder-wall table)"),
})


class AssetError(Exception):
    """An asset that is not packaged, or whose contents are not the ones asked for."""


def asset(name: str) -> Asset:
    """The packaged asset ``name``, with the sha256 MIME pins for it."""
    held = PACKAGED.get(name)
    if held is None:
        raise AssetError(f"no packaged asset is named {name!r}; the packaged assets are {sorted(PACKAGED)}")
    return Asset(name, held.sha256)


def asset_source(held: Asset) -> str:
    """Where the asset is read from, for the lock: ``mime.data/<file>``."""
    packaged = PACKAGED.get(held.name)
    return f"mime.data/{packaged.file}" if packaged is not None else "not packaged"


@cache
def _contents(file: str) -> tuple[bytes, str]:
    data = (importlib.resources.files(__name__) / file).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def asset_problem(held: Asset) -> str | None:
    """Why the contents of ``held`` cannot be read as asked, or ``None`` when they can."""
    packaged = PACKAGED.get(held.name)
    if packaged is None:
        return f"no packaged asset is named {held.name!r}; the packaged assets are {sorted(PACKAGED)}"
    try:
        _, digest = _contents(packaged.file)
    except OSError as error:
        return f"the packaged asset {held.name!r} ({asset_source(held)}) cannot be read: {error}"
    if digest != held.sha256:
        return (f"the packaged asset {held.name!r} ({asset_source(held)}) has sha256 {digest}, and the experiment asks "
                f"for {held.sha256}: the installed MIME ships other contents under that name")
    return None


def asset_bytes(held: Asset) -> bytes:
    """The contents of ``held``, once their sha256 is checked; :class:`AssetError` otherwise."""
    problem = asset_problem(held)
    if problem is not None:
        raise AssetError(problem)
    return _contents(PACKAGED[held.name].file)[0]


__all__ = ["PACKAGED", "Asset", "AssetError", "PackagedAsset", "asset", "asset_bytes", "asset_problem", "asset_source"]
