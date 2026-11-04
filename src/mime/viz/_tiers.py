"""Which baked meshes a view that may be shared can hold: a glTF file, a web page, a notebook's output
saved with it. Shareable are the ``open`` and ``conditional`` tiers (a conditional set's conditions are printed beside
it); ``referenced`` and ``carried`` meshes, and a mesh whose tier no record states, are not.

Two guards, each failing closed. The overlay a shareable view composes leaves such a mesh out
(``overlay(..., tiers=SHAREABLE_TIERS)``), so its frames' simple shapes stay drawn; and before a composed stage is
converted, :func:`refuse_unshareable` reads where every prim's opinions come from, and refuses a stage that composes a
baked mesh layer of the asset cache whose tier is not shareable or is stated nowhere: a stage opened some other way
(a view layer written for a USD viewer, ``mime visuals view``) cannot carry one out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.viz._visuals import baked_sets, baked_tier, visuals_dir

#: The tiers whose meshes a shareable view holds.
SHAREABLE_TIERS = frozenset({"open", "conditional"})


def unshareable(stage: Any) -> list[tuple[str, str, str | None]]:
    """Every prim of the composed ``stage`` an opinion of a baked mesh layer reaches whose tier is not shareable: its
    path, the layer's file name and the tier (``None`` where no record states one)."""
    cache = visuals_dir().resolve()
    sets = baked_sets()
    found: list[tuple[str, str, str | None]] = []
    judged: dict[str, str | None] = {}
    for prim in stage.TraverseAll():
        for spec in prim.GetPrimStack():
            real = str(spec.layer.realPath or "")
            if not real:
                continue
            path = Path(real).resolve()
            if path.parent != cache:
                continue
            if path.name not in judged:
                name = sets.get(path.name)
                judged[path.name] = baked_tier(path, name) if name is not None else None
            tier = judged[path.name]
            if tier not in SHAREABLE_TIERS:
                found.append((str(prim.GetPath()), path.name, tier))
                break
    return found


def refuse_unshareable(stage: Any, target: Path) -> None:
    """Refuse (``visuals.not-shareable``) a composed stage that holds a baked mesh a shareable view may not."""
    found = unshareable(stage)
    if not found:
        return
    first, layer, tier = found[0]
    stated = f"its tier is {tier}" if tier is not None else "no record states its tier"
    raise ArgumentRefused([Refusal(
        "visuals.not-shareable", RefusalClass.USER,
        f"{target} composes {len(found)} prims from baked meshes of the asset cache that a shareable view may not "
        f"hold, {first} among them (from {layer}: {stated}; only the open and conditional tiers are shared)",
        "name the stage, the recording, or the run or saved folder itself, not a view layer written for a USD viewer: "
        "mime visuals web <folder> leaves such meshes out and draws simple shapes in their place")])


__all__ = ["SHAREABLE_TIERS", "refuse_unshareable", "unshareable"]
