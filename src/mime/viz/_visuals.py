"""A description's meshes in the asset cache, and the overlay that draws them on a stage.

A manifest of meshes (``mime.assets.Manifest``: beside a description, in an installed package, or in a saved folder's
lock) is fetched and baked on the user's machine (``mime visuals fetch <description folder | manifest | folder |
name>``): each record is resolved through :func:`mime.assets.resolve` (held to its sha256), read, decimated to its share
of the triangle budget and written as a USD layer under ``visuals/`` in the asset cache, with a record of what it was
made from and how (``<key>.json``), the terms of its source, and its tier: a converted mesh is a derived file, and
takes the strictest terms of its source (``mime.terms.derived_verdict``). The cache's index of a set
(``visuals/<set>.json``) holds the set's credit, licence and notice, so that the overlay needs no manifest. Nothing of
it is ever written into a saved or a run folder: the stage names each mesh by its set and its source's sha256
(``mime:visual:asset``, ``mime:visual:sha256``), and :func:`overlay` writes, into the cache, a layer of overs that
references each baked mesh into its prim and hides the simple shapes that stand in for it. :func:`view` writes a layer
that opens a stage or a recording with that overlay, for any USD viewer; :func:`open_stage` opens one with it in
Python."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mime._lock_json import read_lock
from mime.assets import AssetRef, Manifest, read_manifest, read_manifest_data
from mime.build._refusals import ArgumentRefused, BuildRefused, Refusal, RefusalClass
from mime.scene import VISUAL_ASSET, VISUAL_ROLE, VISUAL_SHA256, manifest_beside, visual_sets
from mime.terms import derived_verdict
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom, Vt
from mime.viz._meshes import Mesh, TriangleBudget, decimate, one_sided_distances, read_stl, write_mesh

CONVERTER = "mime.viz._meshes"
CONVERTER_VERSION = "0.1.0"
VISUALS_DIR = "visuals"
#: How the conversion record's distances are measured: from every vertex of each mesh to the other's surface, both ways
#: (so that no denser sample of vertices finds more), beside the most any kept vertex moved, the cell's diagonal, which
#: bounds the baked vertices' side only.
DISTANCE_BASIS = "every vertex of each mesh to the other's surface, both ways"
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


#: The cache's index of a baked set: what the overlay reads, the set's terms among it.
INDEX_FORMAT = "mime.visual_index/1"
LOCK_FILE = "mime.lock"
#: The lock's key that holds the manifests a saved experiment's stage draws.
LOCK_VISUALS = "visuals"


def _unknown(target: str) -> ArgumentRefused:
    known = ", ".join(sorted(visual_sets())) or "none by name"
    return ArgumentRefused([Refusal(
        "visuals.unknown-set", RefusalClass.USER,
        f"{target!r} is no manifest MIME finds: not a manifest file, a description with a manifest beside it, a folder "
        f"holding visuals.json or a saved experiment, nor the name of a set an installed package ships ({known})",
        "name the description's folder (mime visuals fetch examples/descriptions/ar4/), its manifest file, a saved "
        "folder, or a set from mime visuals list")])


def lock_manifests(folder: Path) -> list[Manifest]:
    """The manifests a saved folder's lock (or a run folder's experiment's) holds of the meshes its stage draws."""
    for lock in (folder / LOCK_FILE, folder / "experiment" / LOCK_FILE):
        if lock.is_file():
            try:
                held: object = read_lock(lock)
            except (OSError, ValueError):
                return []
            listed: object = cast(dict[str, object], held).get(LOCK_VISUALS, {}) if isinstance(held, dict) else {}
            if not isinstance(listed, dict):
                return []
            return [read_manifest_data(value, f"{lock} ({LOCK_VISUALS}: {name})")
                    for name, value in sorted(cast(dict[str, object], listed).items())]
    return []


def manifests_for(target: str | Path) -> list[Manifest]:
    """The manifests ``target`` names: a manifest file; a description file, with the manifest beside it; a saved or run
    folder, with the manifests its lock holds; a folder holding ``visuals.json``; or the name of a set an installed
    package ships."""
    path = Path(target)
    if path.is_file():
        if path.suffix == ".json":
            return [read_manifest(path)]
        found = manifest_beside(path)
        if found is None:
            raise _unknown(str(target))
        return [found]
    if path.is_dir():
        held = lock_manifests(path)
        if held:
            return held
        if (path / "visuals.json").is_file():
            return [read_manifest(path / "visuals.json")]
        raise _unknown(str(target))
    named = visual_sets().get(str(target))
    if named is None:
        raise _unknown(str(target))
    return [named]


@dataclass(frozen=True)
class Baked:
    """One mesh baked: its file in the set, the layer it was written to, and its record."""

    path: str
    layer: Path
    record: dict[str, Any]


def index_of(name: str) -> Path:
    return visuals_dir() / f"{name}.json"


def fetch(target: str | Path, *, budget: TriangleBudget = TriangleBudget(),
          say: Callable[[str], None] = print) -> list[Baked]:
    """Fetch and bake every manifest ``target`` names (:func:`manifests_for`) into the asset cache."""
    baked: list[Baked] = []
    for manifest in manifests_for(target):
        baked += bake(manifest, budget=budget, say=say)
    return baked


def bake(held: Manifest, *, budget: TriangleBudget = TriangleBudget(),
         say: Callable[[str], None] = print) -> list[Baked]:
    """Fetch and bake one manifest's meshes into the asset cache: each record resolved (its sha256 checked), read,
    decimated to its share of ``budget`` and written as a USD layer with its record; and the set's index, which the
    overlay reads. Prints the set's credit, licence and notice first."""
    from mime.assets import resolve

    say(held.text())
    files = [(each.path, held.record(each.asset)) for each in held.files]
    sources: list[tuple[str, AssetRef]] = [(path, ref) for path, ref in files if ref is not None]
    stored = [resolve(ref, report=say) for _, ref in sources]
    meshes: list[Mesh] = []
    for (path, _), stored_path in zip(sources, stored, strict=True):
        try:
            meshes.append(read_stl(stored_path.read_bytes()))
        except (ValueError, OSError) as error:
            raise BuildRefused([Refusal(
                "visuals.convert", RefusalClass.ENVIRONMENT,
                f"{path} of {held.name} could not be read as an STL file: {error}",
                f"fetch it again: mime visuals fetch {held.where or held.name}")]) from error
    shares = budget.shares([len(mesh.triangles) for mesh in meshes])
    baked: list[Baked] = []
    for (path, source), mesh, share in zip(sources, meshes, shares, strict=True):
        key = baked_key(source.sha256, share)
        layer = visuals_dir() / f"{key}.usdc"
        decimated = decimate(mesh, share)
        write_mesh(layer, decimated.mesh, MESH_PRIM)
        from_original, to_original = one_sided_distances(mesh, decimated.mesh)
        distance = max(from_original, to_original)
        verdict = derived_verdict([(source.licence, source.verdict)])
        record = {
            "format": "mime.baked_mesh/3", "set": held.name, "file": path, "converter": CONVERTER,
            "version": CONVERTER_VERSION, "inputs": {"source": {"sha256": source.sha256, "size": source.size,
                                                                "location": source.locations[0]}},
            "parameters": {"method": "vertex clustering on a cubic grid", "triangles": share,
                           "per_mesh": budget.per_mesh, "per_stage": budget.per_stage},
            "checks": {"triangles": {"original": int(len(mesh.triangles)), "baked": int(len(decimated.mesh.triangles)),
                                     "within_budget": bool(len(decimated.mesh.triangles) <= share)},
                       "distance_m": {"cell": decimated.cell, "moved_at_most": decimated.moved,
                                      "measured": distance, "original_to_baked": from_original,
                                      "baked_to_original": to_original, "basis": DISTANCE_BASIS,
                                      "vertices": {"original": int(len(mesh.points)),
                                                   "baked": int(len(decimated.mesh.points))}}},
            "output": {"sha256": hashlib.sha256(layer.read_bytes()).hexdigest(), "bytes": layer.stat().st_size},
            "licence": source.licence, "tier": verdict.tier, "verdict": verdict.words(), "by": verdict.by,
            "role": source.role or "presentation", "credit": source.credit or held.credit, "notice": held.notice}
        (visuals_dir() / f"{key}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        baked.append(Baked(path, layer, record))
        say(f"{path}: {len(mesh.triangles)} triangles baked to {len(decimated.mesh.triangles)} "
            f"(budget {share}), within {distance * 1e3:.2f} mm of the original, measured on every vertex both ways "
            f"(the original's to the bake {from_original * 1e3:.2f} mm, the bake's to the original "
            f"{to_original * 1e3:.2f} mm; no kept vertex moved more than {decimated.moved * 1e3:.2f} mm)")
    index = {"format": INDEX_FORMAT, "set": held.name, "title": held.title, "credit": held.credit,
             "notice": held.notice, "licence": held.licence, "tier": held.verdict.tier, "role": held.role,
             "text": held.text(), "budget": {"per_mesh": budget.per_mesh, "per_stage": budget.per_stage},
             "files": {item.path: {"sha256": source.sha256, "layer": item.layer.name}
                       for item, (_, source) in zip(baked, sources, strict=True)}}
    index_of(held.name).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return baked


def _indexes() -> dict[str, dict[str, Any]]:
    """Every set's index in the cache, by set: written by :func:`bake`, or by an earlier MIME (no format, no terms)."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(visuals_dir().glob("*.json")) if visuals_dir().is_dir() else ():
        try:
            held: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        index = cast(dict[str, Any], held) if isinstance(held, dict) else None
        if index is not None and index.get("format", INDEX_FORMAT) == INDEX_FORMAT and "set" in index \
                and isinstance(index.get("files"), dict):
            found[str(index["set"])] = index
    return found


def baked_layers() -> dict[tuple[str, str], Path]:
    """Every baked mesh the cache holds, by its set and its source's sha256."""
    found: dict[tuple[str, str], Path] = {}
    for name, index in _indexes().items():
        for item in cast(dict[str, dict[str, Any]], index["files"]).values():
            layer = visuals_dir() / str(item["layer"])
            if layer.is_file():
                found[(name, str(item["sha256"]))] = layer
    return found


def _notice(name: str, indexes: dict[str, dict[str, Any]], manifests: dict[str, Manifest]) -> str:
    """A set's credit, licence and notice: from the cache's index where it is baked, else from the folder's lock or an
    installed package's manifest, else a line saying where its terms are."""
    if name in indexes and isinstance(indexes[name].get("text"), str):
        return str(indexes[name]["text"])
    if name in manifests:
        return manifests[name].text()
    shared = visual_sets().get(name)
    if shared is not None:
        return shared.text()
    return (f"{name}: no manifest of it is here; its terms are in the manifest beside the description the stage was "
            f"built from")


#: What a fetch command names where the description's folder is not known: a placeholder, never quoted.
UNKNOWN_FOLDER = "<the description's folder>"


@dataclass
class Overlay:
    """What the cache holds for a stage: the overlay's path (``None`` when it holds nothing the stage names), the
    meshes it draws, the meshes the stage names that it does not hold, by set, and the notice (credit, licence and
    terms) of every set the stage names, drawn or missing: what to show a user beside the meshes, or beside the simple
    shapes that stand in for them, with how to fetch them."""

    path: Path | None
    drawn: int = 0
    missing: dict[str, int] = field(default_factory=dict[str, int])
    notices: list[str] = field(default_factory=list[str])
    #: What ``mime visuals fetch`` takes to fetch each missing set: the folder whose lock holds its manifest, the set's
    #: name where an installed package ships it, or the description's folder.
    fetch: dict[str, str] = field(default_factory=dict[str, str])
    #: The meshes the cache holds that this overlay leaves out, by set: their tier is not among the tiers asked for (a
    #: view that may be shared draws only meshes whose terms let them travel), or no record states it.
    withheld: dict[str, int] = field(default_factory=dict[str, int])
    #: The sets whose meshes the overlay draws.
    sets: list[str] = field(default_factory=list[str])
    #: Each set's notice (the entries of ``notices``), by set.
    terms: dict[str, str] = field(default_factory=dict[str, str])

    def fetch_command(self, name: str) -> str:
        """The command that fetches the set ``name``, its argument quoted for a shell (a folder with a space in its
        path pastes whole), or with the placeholder where the description's folder is not known."""
        held = self.fetch.get(name)
        argument = shlex.quote(held) if held is not None and held != UNKNOWN_FOLDER else UNKNOWN_FOLDER
        return f"mime visuals fetch {argument}"

    def fetch_commands(self) -> dict[str, str]:
        """The command that fetches each missing set, by set: what ``hello`` sends a client beside the counts."""
        return {name: self.fetch_command(name) for name in sorted(self.missing)}

    def text(self) -> str:
        lines = [f"{self.drawn} meshes drawn from the asset cache" + (f" ({self.path})" if self.path else "")]
        for name, count in sorted(self.withheld.items()):
            lines.append(f"{count} meshes of {name} are left out of this view: their licence does not let them be "
                         f"shared, or no record states that it does; simple shapes from the description's frames are "
                         f"drawn in their place (a USD viewer draws them on this machine: mime visuals view <target>)")
        for name, count in sorted(self.missing.items()):
            lines.append(f"{count} meshes of {name} are not in the asset cache: drawn as simple shapes from the "
                         f"description's frames; to draw them, `{self.fetch_command(name)}`")
        return "\n".join(lines + self.notices)


def baked_sets() -> dict[str, str]:
    """Each baked layer's file name in the cache, with the set whose index names it."""
    return {str(item["layer"]): name for name, index in _indexes().items()
            for item in cast(dict[str, dict[str, Any]], index["files"]).values()}


def baked_tier(layer: Path, name: str, indexes: dict[str, dict[str, Any]] | None = None,
               manifests: dict[str, Manifest] | None = None) -> str | None:
    """The tier of the baked mesh ``layer`` of set ``name``: its record's (``<key>.json`` beside it, written by
    :func:`bake`), else the set's index's, else the verdict of the set's manifest in the folder's lock or in an
    installed package; ``None`` where none states it."""
    record = layer.with_suffix(".json")
    if record.is_file():
        try:
            held: object = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            held = None
        if isinstance(held, dict) and isinstance(tier := cast(dict[str, Any], held).get("tier"), str):
            return tier
    index = (indexes if indexes is not None else _indexes()).get(name, {})
    if isinstance(index.get("tier"), str):
        return str(index["tier"])
    manifest = (manifests or {}).get(name) or visual_sets().get(name)
    return manifest.verdict.tier if manifest is not None else None


def _folder_of(stage_path: Path) -> Path | None:
    """The saved or run folder a stage or recording is in, where it is in one."""
    for folder in (stage_path.parent.parent, stage_path.parent):
        if (folder / LOCK_FILE).is_file() or (folder / "experiment" / LOCK_FILE).is_file():
            return folder
    return None


def overlay(stage_path: Path, *, tiers: frozenset[str] | None = None) -> Overlay:
    """The overlay of the stage (or recording) at ``stage_path``: written into the cache, keyed by the composed stage's
    visual prims and ``tiers``, referencing each baked mesh into its prim (scaled from metres to the stage's units) and
    hiding each segment's simple shapes where every mesh of its frames is drawn. With ``tiers``, a mesh whose tier
    (:func:`baked_tier`) is not one of them, or is stated nowhere, is left out (``Overlay.withheld``), and its segment's
    simple shapes stay drawn. Nothing is written beside the stage."""
    folder = _folder_of(stage_path)
    manifests = {held.name: held for held in lock_manifests(folder)} if folder is not None else {}
    indexes = _indexes()
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
    withheld: set[str] = set()
    if tiers is not None:
        kept: list[tuple[str, Path | None, str]] = []
        for prim_path, layer, name in wanted:
            if layer is not None and baked_tier(layer, name, indexes, manifests) not in tiers:
                found.withheld[name] = found.withheld.get(name, 0) + 1
                withheld.add(prim_path)
                layer = None
            kept.append((prim_path, layer, name))
        wanted = kept
    for prim_path, layer, name in wanted:
        if layer is None:
            if prim_path not in withheld:
                found.missing[name] = found.missing.get(name, 0) + 1
        else:
            found.drawn += 1
    named = sorted({name for _, _, name in wanted})
    found.terms = {name: _notice(name, indexes, manifests) for name in named}
    found.notices = [found.terms[name] for name in named]
    found.fetch = {name: (f"{folder}/" if folder is not None and name in manifests else
                          name if name in visual_sets() else UNKNOWN_FOLDER) for name in named}
    if not found.drawn:
        return found
    used = sorted({name for _, layer, name in wanted if layer is not None})
    found.sets = used
    text = json.dumps([[p, str(layer.name) if layer else None] for p, layer, _ in wanted]
                      + [meters, sorted(tiers) if tiers is not None else None], separators=(",", ":"))
    path = visuals_dir() / "overlays" / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.usda"
    held = Sdf.Layer.CreateAnonymous(".usda")
    held.customLayerData = {"mime:visuals": {"sets": Vt.StringArray(used),
                                             "credit": Vt.StringArray([str(indexes.get(n, {}).get("credit", ""))
                                                                       for n in used]),
                                             "notice": Vt.StringArray([str(indexes.get(n, {}).get("notice", ""))
                                                                       for n in used])}}
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


def view(target: Path, *, tiers: frozenset[str] | None = None) -> tuple[Path, Overlay]:
    """A layer in the cache that opens ``target`` (a stage, a recording, or a run or saved folder's) with its overlay
    (of the meshes of ``tiers`` alone, where given): the path to give a USD viewer, and the overlay. Without an overlay
    the view is ``target`` as it is."""
    held = _target(target)
    found = overlay(held, tiers=tiers)
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
    keyed = str(resolved) if tiers is None else f"{resolved}\n{found.path.name}"
    name = hashlib.sha256(keyed.encode("utf-8")).hexdigest()
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


__all__ = ["CONVERTER", "CONVERTER_VERSION", "INDEX_FORMAT", "LOCK_VISUALS", "Baked", "Overlay", "bake", "baked_key",
           "baked_layers", "baked_sets", "baked_tier", "fetch", "lock_manifests", "manifests_for", "open_stage", "overlay", "view", "visuals_dir"]

