"""The scene's references composed at build: each asset read once, pinned by the SHA-256 of the bytes read, checked,
and opened under ``/World`` in a stage held in memory, so that the build can ask which prims exist and read the design
data the ``mime:`` schema puts on part prims. The bytes read are the bytes the saved form writes."""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mime.scene._derived import GEOMETRY, GEOMETRY_API
from mime.scene._frame import Problem
from mime.scene._pxr import Gf, Sdf, Usd, UsdGeom
from mime.scene._scene import Reference, sha256_hex
from mime.things import WORLD, did_you_mean

#: The ``mime:`` applied schema for a magnet's design data, written as the ``apiSchemas`` token on a part prim plus
#: its namespaced attribute: no compiled or codeless plugin, so any USD reads and writes it.
MAGNET_API = "MimeMagnetAPI"
#: A unit vector in the part's own frame, the direction the part is magnetised along.
MAGNET_DIRECTION = "mime:magnet:direction"
#: Where the saved form writes each referenced layer, relative to the stage's own folder.
LAYERS = "layers"

_CHILD_OF_WORLD = re.compile(rf"{WORLD}/[A-Za-z_][A-Za-z0-9_]*")
_PRIM_PATH = re.compile(rf"{WORLD}(/[A-Za-z_][A-Za-z0-9_]*)+")
_RELATIVE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*")
_UNIT_TOLERANCE = 1e-6


def valid_prim_path(path: str) -> bool:
    """A prim path below ``/World``, each element an identifier."""
    return bool(_PRIM_PATH.fullmatch(path))


def valid_part_path(path: str) -> bool:
    """A part's path below its body's prim, relative (``Magnets``, ``Shell/Magnets``)."""
    return bool(_RELATIVE.fullmatch(path))


def within(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


@dataclass(frozen=True)
class ComposedLayer:
    """A referenced asset as read: where it is referenced, the file it is saved as below the stage's folder, its
    pin, and its bytes."""

    at: str
    file: str
    sha256: str
    content: bytes


@dataclass(frozen=True)
class MagnetDesign:
    """A magnet part's magnetisation direction, in the part's frame and in its body's."""

    part: str
    in_part: tuple[float, float, float]
    in_body: tuple[float, float, float]


class Composition:
    """The composed references: which prims exist, the design data on them, and every problem found composing."""

    def __init__(self, layers: tuple[ComposedLayer, ...], stage: Any, opened: Sequence[Any],
                 problems: list[Problem]) -> None:
        self.layers = layers
        self.problems = problems
        self._stage = stage
        self._opened = list(opened)  # the anonymous layers the stage references stay alive with it

    def has(self, path: str) -> bool:
        if self._stage is None or not valid_prim_path(path):
            return False
        return bool(self._stage.GetPrimAtPath(path).IsValid())

    def provided(self, path: str) -> bool:
        """Whether ``path`` is a referenced prim or lies below one."""
        return any(within(path, layer.at) for layer in self.layers)

    def prims_below(self, root: str) -> list[str]:
        if self._stage is None:
            return []
        found: list[str] = []
        for prim in self._stage.Traverse():
            path = str(prim.GetPath())
            if within(path, root):
                found.append(path)
        return found

    def magnet(self, part: str, body: str) -> tuple[MagnetDesign | None, list[Problem]]:
        """The design data of the part at ``part``, read through the ``mime:`` schema: its direction, or ``None``
        when the part's prim carries none; refused when the prim is missing, or carries the attribute without the
        schema applied, or the schema without the attribute, or a direction that is not a unit vector."""
        if not self.has(part):
            names = [p.removeprefix(f"{body}/") for p in self.prims_below(body) if p != body]
            return None, [("magnet.part-missing", f"body prim {body} has no part prim {part}",
                           f"name a part prim below the body, in parts={{...}}. "
                           f"{did_you_mean(part.removeprefix(f'{body}/'), names)}")]
        prim = self._stage.GetPrimAtPath(part)
        applied = MAGNET_API in list(prim.GetPrimTypeInfo().GetAppliedAPISchemas())
        attribute = prim.GetAttribute(MAGNET_DIRECTION)
        authored = bool(attribute) and bool(attribute.HasAuthoredValue())
        if authored and not applied:
            return None, [("magnet.schema-not-applied", f"part prim {part} writes {MAGNET_DIRECTION} without the "
                                                        f"{MAGNET_API} schema applied",
                           f"apply it on the prim: `prepend apiSchemas = [\"{MAGNET_API}\"]`")]
        if not authored:
            if applied:
                return None, [("magnet.schema-incomplete", f"part prim {part} applies {MAGNET_API} and writes no "
                                                           f"{MAGNET_DIRECTION}",
                               f"write `{MAGNET_DIRECTION}` as a unit vector in the part's frame")]
            return None, []
        value = attribute.Get()
        try:
            components = tuple(float(c) for c in value)
        except (TypeError, ValueError):
            components = ()
        norm = math.sqrt(sum(c * c for c in components)) if len(components) == 3 else math.nan
        if len(components) != 3 or not math.isfinite(norm) or abs(norm - 1.0) > _UNIT_TOLERANCE:
            return None, [("magnet.direction", f"part prim {part}: {MAGNET_DIRECTION} = {value} is not a unit vector",
                           "write the direction as a unit vector in the part's frame, such as `(0, 1, 0)`")]
        in_part = (components[0] / norm, components[1] / norm, components[2] / norm)
        return MagnetDesign(part, in_part, self.in_body(part, body, in_part)), []

    def placement(self, part: str, body: str, meters_per_unit: float) -> tuple[
            tuple[float, float, float], tuple[tuple[float, float, float], ...], list[Problem]]:
        """Where the prim at ``part`` sits in the frame of the prim at ``body``, from the transforms authored between
        them: its origin in metres, and the rotation taking a vector in the part's frame into the body's (a matrix
        acting on column vectors). A transform that scales or shears is refused: a part's solid is declared in its
        own units, and nothing rescales it."""
        stage = self._stage
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(stage.GetPrimAtPath(part),
                                                                   stage.GetPrimAtPath(body))
        origin = matrix.ExtractTranslation()
        rows = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
        rotation = tuple((rows[0][j], rows[1][j], rows[2][j]) for j in range(3))  # USD's matrices act on rows
        product = [[sum(rotation[i][k] * rotation[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        if any(abs(product[i][j] - (1.0 if i == j else 0.0)) > 1e-9 for i in range(3) for j in range(3)):
            return (0.0, 0.0, 0.0), rotation, [("part.scaled", f"part prim {part} is scaled or sheared relative to "
                                                               f"its body {body}",
                                                "author the part with a rotation and a translation only "
                                                "(`xformOp:orient`, `xformOp:translate`)")]
        return (float(origin[0]) * meters_per_unit, float(origin[1]) * meters_per_unit,
                float(origin[2]) * meters_per_unit), rotation, []

    def in_body(self, part: str, body: str, direction: tuple[float, float, float]) -> tuple[float, float, float]:
        """``direction``, a unit vector in the frame of the prim at ``part``, in the frame of the prim at ``body``:
        turned by the transforms authored between them."""
        stage = self._stage
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(stage.GetPrimAtPath(part),
                                                                   stage.GetPrimAtPath(body))
        turned = matrix.TransformDir(Gf.Vec3d(*direction))
        length = math.sqrt(sum(float(c) ** 2 for c in turned))
        return (float(turned[0]) / length, float(turned[1]) / length, float(turned[2]) / length)


def _open(content: bytes, suffix: str) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        held = Path(folder) / f"asset{suffix}"
        held.write_bytes(content)
        return Sdf.Layer.OpenAsAnonymous(str(held))


def _layer_problems(layer: Any, reference: Reference, meters_per_unit: float | None) -> list[Problem]:
    where = f"the asset {reference.asset.path.name} referenced at {reference.at}"
    problems: list[Problem] = []
    if not layer.defaultPrim:
        problems.append(("scene.asset-default-prim", f"{where} sets no defaultPrim, so nothing names the prim to "
                                                     f"reference", "set the layer's `defaultPrim` to the robot's root "
                                                                   "prim"))
    pseudo = layer.pseudoRoot
    if not pseudo.HasInfo("metersPerUnit"):
        problems.append(("scene.asset-units", f"{where} authors no metersPerUnit, so its lengths would be read in "
                                              f"an assumed unit", "author `metersPerUnit` on the asset layer"))
    elif meters_per_unit is not None and not math.isclose(float(pseudo.GetInfo("metersPerUnit")), meters_per_unit,
                                                          rel_tol=1e-12):
        problems.append(("scene.asset-units", f"{where} is in {pseudo.GetInfo('metersPerUnit')} m per unit and the "
                                              f"scene in {meters_per_unit}: nothing compensates the difference",
                         "write the asset in the scene's units, or the scene in the asset's: Scene.new(..., "
                         "units=...)"))
    dependencies = sorted(str(d) for d in layer.GetCompositionAssetDependencies())
    if dependencies:
        problems.append(("scene.asset-dependencies", f"{where} composes other files ({', '.join(dependencies)}), "
                                                     f"which the saved form does not pin or copy yet",
                         "flatten the asset into one layer (`usdcat --flatten <file> -o <one file>`)"))
    kinds = _authored_kinds(layer)
    if kinds:
        problems.append(("scene.stage-first", f"{where} authors a geometry kind on {', '.join(kinds)}: taking a "
                                              f"kind from a stage is stage-first authoring, which arrives with "
                                              f"OpenMaterica's USD support, and in a Python-first scene the "
                                              f"declarations are the one authority for a shape",
                         "declare the part's kind in Python, Body(..., parts={\"<part>\": Part(<kind>(...))}), and "
                         "remove mime:geometry and MimeGeometryAPI from the asset"))
    return problems


def _authored_kinds(layer: Any) -> list[str]:
    """The prims of ``layer`` that author a geometry kind: ``MimeGeometryAPI`` among their API schemas, or a
    ``mime:geometry`` attribute."""
    found: list[str] = []

    def visit(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if not isinstance(spec, Sdf.PrimSpec):
            return
        schemas = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
        applied = GEOMETRY_API in (list(schemas.GetAddedOrExplicitItems()) if schemas is not None else [])
        names = [str(name) for name in spec.attributes.keys()]
        if applied or any(name == GEOMETRY or name.startswith(f"{GEOMETRY}:") for name in names):
            found.append(str(path))

    layer.Traverse(layer.pseudoRoot.path, visit)
    return sorted(found)


def compose(references: Sequence[Reference], meters_per_unit: float | None) -> Composition:
    """Read, pin and check every reference, and compose them under ``/World`` in a stage held in memory."""
    problems: list[Problem] = []
    layers: list[ComposedLayer] = []
    opened: list[Any] = []
    root = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(root)
    UsdGeom.Xform.Define(stage, WORLD)
    files: dict[str, str] = {}
    for reference in references:
        where = f"the asset {reference.asset.path} referenced at {reference.at}"
        if not _CHILD_OF_WORLD.fullmatch(reference.at):
            problems.append(("scene.reference-path", f"{where}: {reference.at!r} is not a prim directly below "
                                                     f"{WORLD}", "reference assets at scene.reference(..., "
                                                                 f"at=\"{WORLD}/<name>\")"))
            continue
        if any(layer.at == reference.at for layer in layers):
            problems.append(("scene.reference-path", f"two assets are referenced at {reference.at}",
                             "reference each asset at its own prim: scene.reference(..., at=\"/World/<name>\") for "
                             "each"))
            continue
        try:
            content = reference.asset.read()
        except OSError as error:
            problems.append(("scene.asset-missing", f"cannot read {where}: {error}",
                             "pass the path of the asset's layer file: asset(\"<file>.usda\")"))
            continue
        digest = sha256_hex(content)
        if reference.asset.sha256 is not None and reference.asset.sha256 != digest:
            problems.append(("scene.asset-changed", f"{where} is pinned at sha256 {reference.asset.sha256} and its "
                                                    f"bytes hash to {digest}: the file changed after it was pinned",
                             "restore the pinned file, or pin the new one with its hash: asset(path, sha256=...)"))
            continue
        file = f"{LAYERS}/{reference.asset.path.name}"
        if files.get(file, digest) != digest:
            problems.append(("scene.asset-name", f"two different assets are both named {reference.asset.path.name}, "
                                                 f"and the saved form writes each as {file}",
                             "give the assets distinct file names, asset(\"<path>\") for each"))
            continue
        files[file] = digest
        layer = _open(content, reference.asset.path.suffix or ".usda")
        if layer is None:
            problems.append(("scene.asset-unreadable", f"{where} is not a USD "
                                                       "layer", "pass a .usda or .usdc file: asset(\"<file>.usda\")"))
            continue
        found = _layer_problems(layer, reference, meters_per_unit)
        problems += found
        if found:
            continue
        opened.append(layer)
        stage.DefinePrim(reference.at).GetReferences().AddReference(layer.identifier)
        layers.append(ComposedLayer(reference.at, file, digest, content))
    return Composition(tuple(layers), stage, opened, problems)


__all__ = ["LAYERS", "MAGNET_API", "MAGNET_DIRECTION", "ComposedLayer", "Composition", "MagnetDesign", "compose",
           "valid_part_path", "valid_prim_path", "within"]
