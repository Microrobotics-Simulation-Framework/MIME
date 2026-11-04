"""The records a name finds: the shared sources' (MIME's own ``mime.data/assets/``, and every installed package's
``mime.assets`` entry point), under the pins of the experiment in hand (``mime.assets._sources``)."""

from __future__ import annotations

from collections.abc import Mapping

from mime.assets._reference import AssetRef
from mime.assets._sources import conflicts, resolve_name, shared_records
from mime.build import BuildRefused, Refusal, RefusalClass


def known() -> Mapping[str, AssetRef]:
    """Every record the shared sources agree on, by name: MIME's own, then each installed source's."""
    return shared_records()


def reference(name: str) -> AssetRef:
    """The record called ``name``: the experiment's pin where it has one, else the shared sources'. Refused where
    nothing holds the name (``asset.unknown``), or where two shared sources hold it as different files and nothing pins
    one (``asset.conflict``)."""
    held = resolve_name(name, known())
    if held is None:
        clashing = sorted(conflicts())
        raise BuildRefused([Refusal("asset.unknown", RefusalClass.USER, f"no asset reference is named {name!r}",
                                    f"use one of them, asset(\"<name>\"): {sorted({*known(), *clashing})}")])
    return held


__all__ = ["known", "reference"]
