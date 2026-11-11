"""Package data shipped in the wheel: small default assets, each pinned by its sha256 (surrogate weights, the exact
sphere-plane resistance table, copies of OpenMaterica's example records, three electromagnetic navigation systems'
calibrations).

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
    "sphere_plane_resistance": PackagedAsset(
        "sphere_plane_resistance.json", "346bc025d0b56460a20ba98bdfc9023457cba32ca975d44e363585f80a904f5f",
        "the exact resistance of a sphere near a plane wall at gaps of 1e-3 to 10 radii, five entries at 97 knots and "
        "their midpoints, from the bispherical solutions (Brenner 1961, Jeffery 1915, Dean and O'Neill 1963, O'Neill "
        "1964), computed by MIME (scripts/sphere_plane_table.py): the near-contact law's sphere form reads it"),
    "openmaterica_example_water": PackagedAsset(
        "records/water.json", "36add4875e58ade4810d8924cdc5fa7a0dbce28f621c02e4d98fb3516bc92761",
        "OpenMaterica's example record example/water 1.0.1 (its main at commit eb722a9b): liquid water's density "
        "and viscosity at 293.15 and 310.15 K, from the IAPWS formulations; values CC0-1.0; an example, not reviewed "
        "library data"),
    "openmaterica_example_ndfeb_n45": PackagedAsset(
        "records/ndfeb_n45.json", "40ff82e8a2fb472abcc6cea16bdd594cef78d13c80400d2a94f9926e6b37c4b0",
        "OpenMaterica's example record example/ndfeb_n45 0.1.0 (its tag openmaterica-v0.1.0.dev3): grade N45's "
        "remanence as a vendor's limits, 1.32 to 1.37 T; values CC0-1.0; an example, not reviewed library data"),
    "navion_2": PackagedAsset(
        "emns_calibrations/navion_2.yaml", "ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56",
        "ETH MSRL's published calibration of Navion_2 (3 coils, calibrated multipole model of order one), evaluated by "
        "MIME: Workspace_eMNS's Navion_2_Calibration_24-02-2020.yaml byte for byte; Apache-2.0, Multi-Scale Robotics "
        "Lab, ETH Zurich"),
    "octomag": PackagedAsset(
        "emns_calibrations/octomag.yaml", "d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0",
        "ETH MSRL's published calibration of OctoMag (8 coils, calibrated multipole model of order one), evaluated by "
        "MIME: Tesla_core_public's OctoMag_Calibration.yaml byte for byte; Apache-2.0, Multi-Scale Robotics Lab, ETH Zurich"),
    "cardiomag": PackagedAsset(
        "emns_calibrations/cardiomag.yaml", "31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c",
        "ETH MSRL's published calibration of CardioMag (8 coils, calibrated multipole model of order one, evaluated by "
        "MIME; its Offset block of eight sources is carried and not enabled), Tesla_core_public's C_Mag_Calibration_06-25-2015_no_offset.yaml byte for "
        "byte; Apache-2.0, Multi-Scale Robotics Lab, ETH Zurich"),
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
