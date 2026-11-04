# Derived from usd2gltf 0.3.5 (https://github.com/mikelyndon/usd2gltf), MIT, Copyright (c) 2020 mikelyndon.
# Modified for MIME: typed; the traversal kept (a node for every prim it descends into, its children under it, an
# invisible or excluded prim skipped with its subtree) and narrowed to meshes and USD's analytic shapes, which MIME
# tessellates (mime.viz._primitives), with their purposes; one buffer and one animation, its time a time code over the
# stage's time codes per second, slowed; a material from displayColor and displayOpacity; a root node turning the
# stage's up axis to glTF's +Y and scaling its units; the PointInstancer gathering (a NameError in 0.3.5), USDZ,
# materials, skins, lights, cameras and gltflib's export dropped.
# SPDX-License-Identifier: MIT
"""A composed USD stage as a glTF 2.0 scene: its prims' hierarchy as nodes, each mesh and analytic shape as triangles,
each transform's time samples as one animation, written as a GLB (``mime.viz._gltf._writer``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mime.scene._pxr import Usd, UsdGeom
from mime.viz._gltf._common import clamped
from mime.viz._gltf._mesh import Triangles, triangles_of
from mime.viz._gltf._writer import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, Buffers
from mime.viz._gltf._xform import pose_of
from mime.viz._primitives import KINDS, tessellate

#: The purposes drawn: a stage's default prims and its proxies (MIME draws a tube's wall as a proxy mesh).
PURPOSES = ("default", "proxy")
#: The one animation's name.
ANIMATION = "run"
#: A colour for a shape with none.
GREY = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class Settings:
    """How a stage is converted: the stage's metres per unit over glTF's (``unit``: glTF holds ``scale`` of the stage's
    units per glTF unit, the root node's scale); the root's turn (x, y, z, w) from the stage's up axis to +Y; the
    display time per simulated second (``slow``); the facets a turn of a curved shape; and a box, in the stage's units,
    of the scene's whole motion, drawn as eight invisible points so that a viewer frames all of it."""

    scale: float = 1.0
    turn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    slow: float = 1.0
    segments: int = 48
    frame: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    purposes: tuple[str, ...] = PURPOSES


@dataclass
class Converted:
    """The GLB and what it holds: its nodes by the prim each stands for, the prims drawn as meshes, the triangles,
    the animation's length in display seconds, and the prims it could not draw (by type)."""

    glb: bytes
    nodes: dict[str, int] = field(default_factory=dict[str, int])
    drawn: list[str] = field(default_factory=list[str])
    triangles: int = 0
    duration: float = 0.0
    skipped: dict[str, int] = field(default_factory=dict[str, int])


def _attribute(prim: Any, name: str, default: Any) -> Any:
    attribute = prim.GetAttribute(name)
    value = attribute.Get(Usd.TimeCode.EarliestTime()) if attribute else None
    return default if value is None else value


def shape_of(prim: Any, segments: int) -> Triangles | None:
    """A gprim's triangles in its own frame: a mesh's, or an analytic shape's tessellated; ``None`` for another kind."""
    kind = str(prim.GetTypeName())
    if prim.IsA(UsdGeom.Mesh):
        return triangles_of(UsdGeom.Mesh(prim))
    if kind not in KINDS:
        return None
    shape = tessellate(kind, radius=float(_attribute(prim, "radius", 1.0)), height=float(_attribute(prim, "height", 2.0)),
                       size=float(_attribute(prim, "size", 2.0)), axis=str(_attribute(prim, "axis", "Z")),
                       segments=segments)
    if shape is None:
        return None
    return Triangles(shape.points, shape.triangles, shape.normals)


def _constant(prim: Any, name: str) -> float | tuple[float, float, float] | None:
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not primvar or not primvar.HasAuthoredValue():
        return None
    value = primvar.ComputeFlattened(Usd.TimeCode.EarliestTime())
    if value is None or len(value) == 0:
        return None
    return clamped(value[0]) if name == "displayColor" else max(min(float(value[0]), 1.0), 0.0)


class _Builder:
    def __init__(self, stage: Any, settings: Settings) -> None:
        self.stage = stage
        self.settings = settings
        self.buffers = Buffers()
        self.nodes: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self._material_keys: dict[tuple[float, ...], int] = {}
        self.samplers: list[dict[str, Any]] = []
        self.channels: list[dict[str, Any]] = []
        self._inputs: dict[bytes, int] = {}
        self.result = Converted(b"")
        self.rate = float(stage.GetTimeCodesPerSecond()) or 1.0
        self.start = float(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else 0.0

    def material(self, colour: tuple[float, float, float], opacity: float, *, per_vertex: bool) -> int:
        key = (*colour, opacity, float(per_vertex))
        if key not in self._material_keys:
            factor = [1.0, 1.0, 1.0, opacity] if per_vertex else [*colour, opacity]
            held: dict[str, Any] = {"pbrMetallicRoughness": {"baseColorFactor": factor, "metallicFactor": 0.0,
                                                             "roughnessFactor": 0.6}}
            if opacity < 1.0:
                held.update(alphaMode="BLEND", doubleSided=True)
            self.materials.append(held)
            self._material_keys[key] = len(self.materials) - 1
        return self._material_keys[key]

    def mesh(self, prim: Any, shape: Triangles) -> int:
        attributes = {"POSITION": self.buffers.accessor(shape.points.astype(np.float32), target=ARRAY_BUFFER)}
        if shape.normals is not None:
            lengths = np.linalg.norm(shape.normals, axis=1, keepdims=True)
            unit = shape.normals / np.where(lengths > 0.0, lengths, 1.0)
            attributes["NORMAL"] = self.buffers.accessor(unit.astype(np.float32), target=ARRAY_BUFFER)
        if shape.colours is not None:
            attributes["COLOR_0"] = self.buffers.accessor(np.clip(shape.colours, 0.0, 1.0).astype(np.float32),
                                                          target=ARRAY_BUFFER)
        index_type = np.uint16 if len(shape.points) < 65536 else np.uint32
        indices = self.buffers.accessor(shape.triangles.reshape(-1).astype(index_type), target=ELEMENT_ARRAY_BUFFER)
        colour = _constant(prim, "displayColor")
        opacity = _constant(prim, "displayOpacity")
        material = self.material(colour if isinstance(colour, tuple) else GREY,
                                 opacity if isinstance(opacity, float) else 1.0, per_vertex=shape.colours is not None)
        self.meshes.append({"name": str(prim.GetName()),
                            "primitives": [{"attributes": attributes, "indices": indices, "material": material}]})
        self.result.triangles += len(shape.triangles)
        return len(self.meshes) - 1

    def track(self, node: int, path: str, codes: np.ndarray, values: np.ndarray) -> None:
        times = ((codes - self.start) / self.rate * self.settings.slow).astype(np.float32)
        key = times.tobytes()
        if key not in self._inputs:
            self._inputs[key] = self.buffers.accessor(times)
        output = self.buffers.accessor(values.astype(np.float32))
        self.samplers.append({"input": self._inputs[key], "output": output, "interpolation": "LINEAR"})
        self.channels.append({"sampler": len(self.samplers) - 1, "target": {"node": node, "path": path}})
        self.result.duration = max(self.result.duration, float(times.max()))

    def visible(self, prim: Any) -> bool:
        if not prim.IsA(UsdGeom.Imageable):
            return True
        imageable = UsdGeom.Imageable(prim)
        if str(imageable.ComputeVisibility(Usd.TimeCode.EarliestTime())) == str(UsdGeom.Tokens.invisible):
            return False
        return str(imageable.ComputePurpose()) in self.settings.purposes

    def node(self, prim: Any) -> int | None:
        if not prim.IsActive() or not prim.IsDefined() or not self.visible(prim):
            return None
        held: dict[str, Any] = {"name": str(prim.GetName()), "extras": {"usd": str(prim.GetPath())}}
        index = len(self.nodes)
        self.nodes.append(held)
        self.result.nodes[str(prim.GetPath())] = index
        if prim.IsA(UsdGeom.Xformable):
            pose = pose_of(prim)
            for name in ("translation", "rotation", "scale", "matrix"):
                value = getattr(pose, name)
                if value is not None:
                    held[name] = list(value)
            if pose.codes is not None:
                held.pop("matrix", None)
                for path, values in pose.tracks.items():
                    self.track(index, path, pose.codes, values)
        if prim.IsA(UsdGeom.Gprim):
            shape = shape_of(prim, self.settings.segments)
            if shape is not None:
                held["mesh"] = self.mesh(prim, shape)
                self.result.drawn.append(str(prim.GetPath()))
            else:
                kind = str(prim.GetTypeName())
                self.result.skipped[kind] = self.result.skipped.get(kind, 0) + 1
        children = [child for child in (self.node(c) for c in prim.GetChildren()) if child is not None]
        if children:
            held["children"] = children
        return index

    def frame(self) -> int | None:
        box = self.settings.frame
        if box is None:
            return None
        low, high = np.asarray(box[0], dtype=np.float64), np.asarray(box[1], dtype=np.float64)
        corners = np.array([[(high if (k >> axis) & 1 else low)[axis] for axis in range(3)] for k in range(8)])
        position = self.buffers.accessor(corners.astype(np.float32), target=ARRAY_BUFFER)
        self.materials.append({"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 0.0]},
                               "alphaMode": "BLEND"})
        self.meshes.append({"name": "frame", "primitives": [{"attributes": {"POSITION": position}, "mode": 0,
                                                             "material": len(self.materials) - 1}]})
        self.nodes.append({"name": "frame", "mesh": len(self.meshes) - 1,
                           "extras": {"mime": "the box of the whole motion, so that a viewer frames all of it"}})
        return len(self.nodes) - 1


def convert(stage: Any, settings: Settings, *, extras: dict[str, Any] | None = None,
            copyright: str | None = None, roots: Sequence[Any] | None = None,
            each: Callable[[Any], None] | None = None) -> Converted:
    """The composed ``stage`` as a GLB, under one root node (the turn to +Y and the unit scale), from ``roots`` (the
    pseudo-root's children by default); ``extras`` go into ``asset.extras``, ``copyright`` into ``asset.copyright``,
    and ``each`` is called with every prim the traversal starts from."""
    built = _Builder(stage, settings)
    root: dict[str, Any] = {"name": "mime", "rotation": list(settings.turn),
                            "scale": [settings.scale, settings.scale, settings.scale]}
    built.nodes.append(root)
    starts = list(roots) if roots is not None else list(stage.GetPseudoRoot().GetChildren())
    children: list[int] = []
    for prim in starts:
        if each is not None:
            each(prim)
        index = built.node(prim)
        if index is not None:
            children.append(index)
    framed = built.frame()
    if framed is not None:
        children.append(framed)
    if children:
        root["children"] = children
    asset: dict[str, Any] = {"version": "2.0", "generator": "MIME (mime.viz.gltf; usd2gltf 0.3.5's converter, MIT)"}
    if copyright:
        asset["copyright"] = copyright
    if extras:
        asset["extras"] = extras
    document: dict[str, Any] = {"asset": asset, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": built.nodes}
    if built.meshes:
        document["meshes"] = built.meshes
    if built.materials:
        document["materials"] = built.materials
    if built.channels:
        document["animations"] = [{"name": ANIMATION, "samplers": built.samplers, "channels": built.channels}]
    built.result.glb = built.buffers.glb(document)
    return built.result


def up_turn(up: str) -> tuple[float, float, float, float]:
    """The turn (x, y, z, w) from a stage's up axis to glTF's +Y: a quarter turn about X for a Z-up stage."""
    if up.upper() == "Z":
        half = math.sqrt(0.5)
        return (-half, 0.0, 0.0, half)
    return (0.0, 0.0, 0.0, 1.0)


__all__ = ["ANIMATION", "PURPOSES", "Converted", "Settings", "convert", "shape_of", "up_turn"]
