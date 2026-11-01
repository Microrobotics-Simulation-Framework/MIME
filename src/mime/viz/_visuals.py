"""A description's meshes in the asset cache, and the overlay that draws them on a stage.

A visual set (``mime.scene.visual_sets``) is fetched and baked on the user's machine (``mime visuals fetch <set>``):
each file is downloaded from the set's pinned upstream through :func:`mime.assets.resolve` (held to its sha256), read,
decimated to its share of the set's triangle budget and written as a USD layer under ``visuals/`` in the asset cache,
with a record of what it was made from and how (``<key>.json``). Nothing of it is ever written into a saved or a run
folder: the stage names each mesh by its set and its source's sha256 (``mime:visual:asset``, ``mime:visual:sha256``),
and :func:`overlay` writes, into the cache, a layer of overs that references each baked mesh into its prim and hides the
simple shapes that stand in for it. :func:`view` writes a layer that opens a stage or a recording with that overlay, for
any USD viewer; :func:`open_stage` opens one with it in Python."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mime.build._refusals import ArgumentRefused, BuildRefused, Refusal, RefusalClass
from mime.scene import VISUAL_ASSET, VISUAL_ROLE, VISUAL_SHA256, VisualSet, visual_sets
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom
from mime.viz._meshes import TriangleBudget, decimate, mesh_distance, read_stl, write_mesh

CONVERTER = "mime.viz.mesh_import"
CONVERTER_VERSION = "0.1.0"
VISUALS_DIR = "visuals"
#: The vertices each way the conversion record's distance is measured on, drawn with a fixed seed.
DISTANCE_SAMPLES = 1500
MESH_PRIM = "mesh"
FALLBACK = "fallback"


def visuals_dir() -> Path:
    """Where the baked meshes, their records, the overlays and the views live: ``visuals/`` in the asset cache."""
    from mime.assets import cache_dir

    return cache_dir() / VISUALS_DIR


def baked_key(sha256: str, triangles: int) -> str:
    """The name a baked mesh is kept under: the hash of its source, the converter and its version, and its budget."""
    text = json.dumps({"converter": CONVERTER, "source": sha256, "triangles": triangles, "version": CONVERTER_VERSION},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _set(name: str) -> VisualSet:
    held = visual_sets().get(name)
    if held is None:
        raise ArgumentRefused([Refusal(
            "visuals.unknown-set", RefusalClass.USER,
            f"no visual set is named {name!r}; MIME knows {', '.join(sorted(visual_sets())) or 'none'}",
            "fetch one it knows: mime visuals fetch <set>, from mime visuals list")])
    return held


@dataclass(frozen=True)
class Baked:
    """One mesh baked: its file in the set, the layer it was written to, and its record."""

    path: str
    layer: Path
    record: dict[str, Any]


def _source_ref(held: VisualSet, index: int) -> Any:
    from mime.assets import AssetRef

    source = held.files[index]
    stem = source.path.rsplit("/", 1)[-1]
    return AssetRef(name=f"{held.name}.{stem}", sha256=source.sha256, size=source.size, licence=held.licence,
                    locations=(source.location,))


def index_of(held: VisualSet) -> Path:
    return visuals_dir() / f"{held.name}.json"


def fetch(name: str, *, budget: TriangleBudget = TriangleBudget(), say: Callable[[str], None] = print) -> list[Baked]:
    """Fetch and bake the visual set ``name`` into the asset cache: each file from its pinned upstream, its sha256
    checked, read, decimated to its share of ``budget`` and written as a USD layer with its record; and the set's index,
    which the overlay reads. Prints the set's credit, licence and notice first."""
    from mime.assets import resolve

    held = _set(name)
    say(held.text())
    sources = [resolve(_source_ref(held, index), report=say) for index in range(len(held.files))]
    meshes = []
    for index, path in enumerate(sources):
        try:
            meshes.append(read_stl(path.read_bytes()))
        except (ValueError, OSError) as error:
            raise BuildRefused([Refusal(
                "visuals.convert", RefusalClass.ENVIRONMENT,
                f"{held.files[index].path} of {name} could not be read as an STL file: {error}",
                f"fetch it again: mime visuals fetch {name}")]) from error
    shares = budget.shares([len(mesh.triangles) for mesh in meshes])
    baked: list[Baked] = []
    for source, mesh, share in zip(held.files, meshes, shares, strict=True):
        key = baked_key(source.sha256, share)
        layer = visuals_dir() / f"{key}.usdc"
        decimated = decimate(mesh, share)
        write_mesh(layer, decimated.mesh, MESH_PRIM)
        distance = mesh_distance(mesh, decimated.mesh, samples=DISTANCE_SAMPLES)
        record = {
            "format": "mime.baked_mesh/1", "set": held.name, "file": source.path, "converter": CONVERTER,
            "version": CONVERTER_VERSION, "inputs": {"source": {"sha256": source.sha256, "size": source.size,
                                                                "location": source.location}},
            "parameters": {"method": "vertex clustering on a cubic grid", "triangles": share,
                           "per_mesh": budget.per_mesh, "per_stage": budget.per_stage},
            "checks": {"triangles": {"original": int(len(mesh.triangles)), "baked": int(len(decimated.mesh.triangles)),
                                     "within_budget": bool(len(decimated.mesh.triangles) <= share)},
                       "distance_m": {"cell": decimated.cell, "moved_at_most": decimated.moved,
                                      "measured": distance, "samples_each_way": DISTANCE_SAMPLES}},
            "output": {"sha256": hashlib.sha256(layer.read_bytes()).hexdigest(), "bytes": layer.stat().st_size},
            "licence": held.licence, "tier": held.tier, "role": held.role, "credit": held.credit,
            "notice": held.notice}
        (visuals_dir() / f"{key}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        baked.append(Baked(source.path, layer, record))
        say(f"{source.path}: {len(mesh.triangles)} triangles baked to {len(decimated.mesh.triangles)} "
            f"(budget {share}), within {distance * 1e3:.2f} mm of the original as measured")
    index = {"set": held.name, "budget": {"per_mesh": budget.per_mesh, "per_stage": budget.per_stage},
             "files": {item.path: {"sha256": next(f.sha256 for f in held.files if f.path == item.path),
                                   "layer": item.layer.name} for item in baked}}
    index_of(held).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return baked


def baked_layers() -> dict[tuple[str, str], Path]:
    """Every baked mesh the cache holds, by its set and its source's sha256."""
    found: dict[tuple[str, str], Path] = {}
    for held in visual_sets().values():
        try:
            index = json.loads(index_of(held).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in index.get("files", {}).values():
            layer = visuals_dir() / str(item["layer"])
            if layer.is_file():
                found[(held.name, str(item["sha256"]))] = layer
    return found


@dataclass
class Overlay:
    """What the cache holds for a stage: the overlay's path (``None`` when it holds nothing the stage names), the
    meshes it draws, the meshes the stage names that it does not hold, by set, and the notices of the sets it draws."""

    path: Path | None
    drawn: int = 0
    missing: dict[str, int] = field(default_factory=dict[str, int])
    notices: list[str] = field(default_factory=list[str])

    def text(self) -> str:
        lines = [f"{self.drawn} meshes drawn from the asset cache" + (f" ({self.path})" if self.path else "")]
        for name, count in sorted(self.missing.items()):
            lines.append(f"{count} meshes of {name} are not in the asset cache: drawn as simple shapes from the "
                         f"description's frames; to draw them, `mime visuals fetch {name}`")
        return "\n".join(lines + self.notices)


def overlay(stage_path: Path) -> Overlay:
    """The overlay of the stage (or recording) at ``stage_path``: written into the cache, keyed by the composed stage's
    visual prims, referencing each baked mesh into its prim (scaled from metres to the stage's units) and hiding each
    segment's simple shapes where every mesh of its frames is drawn. Nothing is written beside the stage."""
    stage = Usd.Stage.Open(str(stage_path))
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    layers = baked_layers()
    wanted: list[tuple[str, Path | None, str]] = []
    for prim in stage.Traverse():
        asset, digest = prim.GetAttribute(VISUAL_ASSET), prim.GetAttribute(VISUAL_SHA256)
        if asset and digest and asset.Get() and digest.Get():
            key = (str(asset.Get()), str(digest.Get()))
            wanted.append((str(prim.GetPath()), layers.get(key), key[0]))
    found = Overlay(None)
    for _, layer, name in wanted:
        if layer is None:
            found.missing[name] = found.missing.get(name, 0) + 1
        else:
            found.drawn += 1
    if not found.drawn:
        return found
    used = sorted({name for _, layer, name in wanted if layer is not None})
    found.notices = [visual_sets()[name].text() for name in used]
    text = json.dumps([[p, str(layer.name) if layer else None] for p, layer, _ in wanted] + [meters],
                      separators=(",", ":"))
    path = visuals_dir() / "overlays" / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.usda"
    held = Sdf.Layer.CreateAnonymous(".usda")
    held.customLayerData = {"mime:visuals": {"sets": used, "credit": [visual_sets()[n].credit for n in used],
                                             "notice": [visual_sets()[n].notice for n in used]}}
    complete: dict[str, bool] = {}
    for prim_path, layer, _ in wanted:
        segment = str(Sdf.Path(prim_path).GetParentPath().GetParentPath().GetParentPath())
        complete[segment] = complete.get(segment, True) and layer is not None
        if layer is None:
            continue
        spec = Sdf.CreatePrimInLayer(held, Sdf.Path(prim_path).AppendChild(MESH_PRIM))
        spec.specifier = Sdf.SpecifierDef
        spec.referenceList.Prepend(Sdf.Reference(f"../{layer.name}"))
        if meters != 1.0:
            scale = Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale.default = Gf.Vec3d(1.0 / meters, 1.0 / meters, 1.0 / meters)
            order = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Sdf.VariabilityUniform)
            order.default = ["xformOp:scale"]
    for prim in stage.Traverse():
        role = prim.GetAttribute(VISUAL_ROLE)
        if not (role and role.Get() in ("collision", "frames")):
            continue
        segment = str(prim.GetPath().GetParentPath().GetParentPath().GetParentPath())
        if complete.get(segment):
            spec = Sdf.CreatePrimInLayer(held, prim.GetPath())
            visibility = Sdf.AttributeSpec(spec, "visibility", Sdf.ValueTypeNames.Token)
            visibility.default = "invisible"
    path.parent.mkdir(parents=True, exist_ok=True)
    held.Export(str(path))
    found.path = path
    return found


def view(target: Path) -> tuple[Path, Overlay]:
    """A layer in the cache that opens ``target`` (a stage, a recording, or a run or saved folder's) with its overlay:
    the path to give a USD viewer, and the overlay. Without an overlay the view is ``target`` as it is."""
    held = _target(target)
    found = overlay(held)
    if found.path is None:
        return held, found
    resolved = held.resolve()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.subLayerPaths.append(f"../overlays/{found.path.name}")
    layer.subLayerPaths.append(str(resolved))
    source = Sdf.Layer.FindOrOpen(str(resolved))
    if source is not None:
        for key in ("upAxis", "metersPerUnit", "defaultPrim", "startTimeCode", "endTimeCode", "timeCodesPerSecond",
                    "framesPerSecond"):
            if source.pseudoRoot.HasInfo(key):
                layer.pseudoRoot.SetInfo(key, source.pseudoRoot.GetInfo(key))
    name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    path = visuals_dir() / "views" / f"{name}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    layer.Export(str(path))
    return path, found


def _target(target: Path) -> Path:
    """The stage a path stands for: a file as it is; a run folder's recording, else its experiment's stage; a saved
    folder's stage."""
    if target.is_file():
        return target
    for candidate in (target / "recording.usdc", target / "experiment" / "scene" / "scene.usda",
                      target / "scene" / "scene.usda"):
        if candidate.is_file():
            return candidate
    raise ArgumentRefused([Refusal(
        "visuals.no-stage", RefusalClass.USER, f"{target} holds no stage, recording or saved experiment",
        "name a stage, a run folder or a saved folder: mime visuals view <folder>")])


def open_stage(target: Path) -> Any:
    """The stage or recording at ``target`` opened with its overlay (as :func:`view` writes it), for Python."""
    path, _ = view(target)
    return Usd.Stage.Open(str(path))


__all__ = ["CONVERTER", "CONVERTER_VERSION", "Baked", "Overlay", "baked_key", "baked_layers", "fetch", "open_stage",
           "overlay", "view", "visuals_dir"]

