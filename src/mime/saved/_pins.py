"""A loaded experiment's pins: the records its lock holds and the meshes its saved stage names, which its rebuild
keeps whatever a shared source holds now: the pin wins."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import cast

from mime.assets import AssetRef, Manifest, pinned, read_manifest_data
from mime.saved._folder import SavedForm
from mime.scene import STAGE_FILE, VISUAL_ASSET, VISUAL_FILE, VISUAL_SHA256, VisualPins, pinned_visuals

#: The lock's key that holds the manifests of the meshes the stage draws (version 6).
VISUALS_KEY = "visuals"


def saved_asset_pins(form: SavedForm) -> tuple[AssetRef, ...]:
    """The records the folder's lock holds, each a pin; a malformed one is the load's to refuse, and pins nothing."""
    found: list[AssetRef] = []
    for record in form.lock.get("assets", []):
        try:
            found.append(AssetRef.from_record(cast(dict[str, object], record)))
        except Exception:  # noqa: BLE001 - the load refuses a malformed record with its reason
            continue
    return tuple(found)


def lock_manifests(form: SavedForm) -> dict[str, Manifest]:
    """The manifests the folder's lock holds of the meshes its stage draws, by name."""
    held = cast(dict[str, object], form.lock).get(VISUALS_KEY, {})
    if not isinstance(held, dict):
        return {}
    found: dict[str, Manifest] = {}
    for name, value in cast(dict[str, object], held).items():
        try:
            found[name] = read_manifest_data(value, f"{form.folder} (the lock's {VISUALS_KEY}: {name})", lock=True)
        except Exception:  # noqa: BLE001 - the load refuses a malformed lock with its reason
            continue
    return found


def saved_visual_pins(form: SavedForm) -> VisualPins:
    """Each mesh prim the saved stage names, with its set and sha256 (or neither), and the lock's manifests."""
    from mime.scene._pxr import Usd

    prims: dict[str, tuple[str | None, str | None]] = {}
    path = form.folder / STAGE_FILE
    if path.is_file():
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
        for prim in stage.Traverse():
            if not prim.GetAttribute(VISUAL_FILE):
                continue
            asset, digest = prim.GetAttribute(VISUAL_ASSET), prim.GetAttribute(VISUAL_SHA256)
            prims[str(prim.GetPath())] = ((str(asset.Get()) if asset and asset.Get() else None),
                                          (str(digest.Get()) if digest and digest.Get() else None))
    return VisualPins(prims, lock_manifests(form))


class KeptPins:
    """The lines a build says of its pins: each asset and each mesh it keeps while a source holds another version."""

    def __init__(self, assets: list[str], visuals: VisualPins | None) -> None:
        self._assets = assets
        self._visuals = visuals

    def lines(self) -> tuple[str, ...]:
        return (*self._assets, *(self._visuals.notes if self._visuals is not None else ()))


@contextmanager
def keeping_pins(form: SavedForm | None) -> Generator[KeptPins]:
    """Inside this block a build keeps the pins of the loaded ``form`` (none for a script's experiment), and what it
    yields gives each line the build report says of a pin a source no longer agrees with."""
    if form is None:
        yield KeptPins([], None)
        return
    with ExitStack() as stack:
        notes = stack.enter_context(pinned(saved_asset_pins(form)))
        visuals = stack.enter_context(pinned_visuals(saved_visual_pins(form)))
        yield KeptPins(notes, visuals)


__all__ = ["VISUALS_KEY", "KeptPins", "keeping_pins", "lock_manifests", "saved_asset_pins", "saved_visual_pins"]
