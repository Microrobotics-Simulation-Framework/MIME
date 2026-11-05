"""An apparatus's motor and magnets drawn in a view, as an overlay of the view alone.

A stage holds a magnet part as a bare frame (``MimeMagnetAPI``, its mount, the name of its solid) because the field of a
point dipole needs no shape; its motor is two frames. This module draws them for a reader of the page or the glTF, and
writes nothing into the stage: the shapes are root prims of the opened stage's *session layer* (``APPARATUS_ROOT``,
outside ``/World``, which the stage's writer alone owns), each following its frame's world pose at every time code the
stage has. A saved stage, a recording, a run folder and the asset cache are never written.

What is drawn, and where its size comes from:

- **A magnet part**: its solid, at the size the saved experiment declares (``experiment.mime.yaml`` beside the stage or
  the recording: a cylinder's radius and length, a sphere's radius, a box's size, all in metres). Where the view has no
  saved experiment, or the solid is another kind, or a size is not in metres, a *placeholder* is drawn instead: a
  cylinder along the magnet's declared direction, its length and diameter each ``PLACEHOLDER_FRACTION`` of the scene's
  diagonal.
- **A motor** (a chain segment named ``motor``): a placeholder housing, always, for no description declares its size: a
  cylinder along the motor's axis (the saved experiment's, else z) of the largest extent of the magnets on its rotor
  for length and half of it for radius, ending in the rotor's plane on the side the axis points away from. Which side of
  the rotor the housing is on is not declared either. A motor with no magnet on it has nothing to size it by and is not
  drawn.

A placeholder is translucent, named ``..._placeholder`` in the glTF, and said in the page's notes with its rule."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from mime.scene._pxr import Gf, Usd, UsdGeom

#: The overlay's root prim: every root prim but ``/World`` is reserved for overlays.
APPARATUS_ROOT = "/ApparatusView"
#: A placeholder's length and diameter, as a fraction of the scene's diagonal.
PLACEHOLDER_FRACTION = 1.0 / 40.0
MAGNET_API_DIRECTION = "mime:magnet:direction"
MOUNT = "mime:mount"
SEGMENT = "mime:segment"
DOCUMENT = "experiment.mime.yaml"
_STEEL = (0.78, 0.2, 0.2)
_PLACEHOLDER = (0.95, 0.75, 0.15)
_HOUSING = (0.45, 0.55, 0.7)
_PLACEHOLDER_OPACITY = 0.45


@dataclass(frozen=True)
class ApparatusShape:
    """One shape the view draws for an apparatus: its overlay prim, the prim of the stage it follows, what it stands
    for (``rpm.rotor_magnet``), its kind (``cylinder``, ``sphere``, ``box``), the size it is drawn at (``size_m``: a
    cylinder's radius and length, a sphere's radius, a box's edges), where the size comes from (``"declared"`` or
    ``"placeholder"``) and, for a placeholder, the rule that sized it."""

    prim: str
    follows: str
    name: str
    kind: str
    size_m: tuple[float, ...]
    source: str
    rule: str = ""

    @property
    def placeholder(self) -> bool:
        return self.source == "placeholder"


def saved_document(source: Path) -> dict[str, Any] | None:
    """The saved experiment's document beside the stage or recording at ``source``: a saved folder's
    (``<folder>/scene/scene.usda``), a run folder's (``<run>/experiment/``, beside its recording or its copy of the
    stage); ``None`` for a stage with none. Read with the safe loader."""
    for folder in (source.parent / "experiment", source.parent.parent, source.parent):
        document = folder / DOCUMENT
        if document.is_file():
            try:
                held: object = yaml.safe_load(document.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                return None
            return cast(dict[str, Any], held) if isinstance(held, dict) else None
    return None


def _vector(value: object) -> tuple[float, float, float] | None:
    """Three numbers of a saved list, as floats; ``None`` for anything else."""
    if not isinstance(value, list):
        return None
    numbers: list[float] = []
    for item in cast(list[Any], value):
        if not isinstance(item, (int, float)):
            return None
        numbers.append(float(item))
    return (numbers[0], numbers[1], numbers[2]) if len(numbers) == 3 else None


def _metres(value: object) -> float | tuple[float, ...] | None:
    """A saved quantity in metres (``{"value": ..., "unit": "m"}``); ``None`` for any other unit."""
    if not isinstance(value, dict):
        return None
    held = cast(dict[str, Any], value)
    if held.get("unit") != "m":
        return None
    number = held.get("value")
    if isinstance(number, (int, float)):
        return float(number)
    if isinstance(number, list) and all(isinstance(v, (int, float)) for v in cast(list[Any], number)):
        return tuple(float(v) for v in cast(list[Any], number))
    return None


def _declared(document: Mapping[str, Any] | None, apparatus: str, key: str) -> tuple[str, dict[str, Any]] | None:
    """The solid of the part ``key`` of ``apparatus`` in the saved document: its kind and settings."""
    if document is None:
        return None
    thing: Any = cast(Mapping[str, Any], document.get("things", {})).get(apparatus)
    if not isinstance(thing, dict):
        return None
    part: Any = cast(dict[str, Any], thing).get("parts", {}).get(key)
    solid: Any = cast(dict[str, Any], part).get("solid") if isinstance(part, dict) else None
    if not isinstance(solid, dict):
        return None
    held = cast(dict[str, Any], solid)
    settings = held.get("settings", {})
    return str(held.get("name")), dict(cast(dict[str, Any], settings)) if isinstance(settings, dict) else {}


def _size(kind: str, settings: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, float, float]] | None:
    """The size and axis of a declared solid the view can draw, in metres; ``None`` otherwise."""
    direction = _vector(settings.get("axis")) or (0.0, 0.0, 1.0)
    if kind == "cylinder":
        radius, length = _metres(settings.get("radius")), _metres(settings.get("length"))
        if isinstance(radius, float) and isinstance(length, float) and radius > 0.0 and length > 0.0:
            return (radius, length), direction
    elif kind == "sphere":
        radius = _metres(settings.get("radius"))
        if isinstance(radius, float) and radius > 0.0:
            return (radius,), (0.0, 0.0, 1.0)
    elif kind == "box":
        size = _metres(settings.get("size"))
        if isinstance(size, tuple) and len(size) == 3 and min(size) > 0.0:
            return size, (0.0, 0.0, 1.0)
    return None


def _unit(vector: Sequence[float]) -> np.ndarray:
    held = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(held))
    return held / length if length > 0.0 else np.array([0.0, 0.0, 1.0])


def _turn_z_to(axis: Sequence[float]) -> tuple[float, float, float, float]:
    """The unit quaternion (w, x, y, z) that turns +z to ``axis``."""
    a = _unit(axis)
    if a[2] > 1.0 - 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    if a[2] < -1.0 + 1e-12:
        return 0.0, 1.0, 0.0, 0.0
    w, x, y = 1.0 + a[2], -a[1], a[0]
    norm = math.sqrt(w * w + x * x + y * y)
    return w / norm, x / norm, y / norm, 0.0


def _following(stage: Any, path: str, follows: Any, codes: Sequence[float]) -> Any:
    """The overlay Xform at ``path``, with the world pose of ``follows`` (a prim of the stage) as translate and orient
    samples at each of ``codes`` (once, at the default time, where there are none)."""
    prim = UsdGeom.Xform.Define(stage, path)
    translate = prim.AddTranslateOp()
    orient = prim.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    xformable = UsdGeom.Xformable(follows)
    for code in codes or [None]:
        time = Usd.TimeCode(float(code)) if code is not None else Usd.TimeCode.Default()
        matrix = xformable.ComputeLocalToWorldTransform(time)
        pose = matrix.RemoveScaleShear()
        translate.Set(Gf.Vec3d(matrix.ExtractTranslation()), time)
        orient.Set(Gf.Quatd(pose.ExtractRotationQuat()), time)
    return prim


def _solid(stage: Any, path: str, kind: str, size_units: Sequence[float], axis: Sequence[float],
           offset: Sequence[float], colour: tuple[float, float, float], opacity: float) -> None:
    """The shape below the overlay Xform ``path``: a cylinder along ``axis``, a sphere, or a box, centred at
    ``offset``; ``size_units`` in the stage's units."""
    shape: Any
    if kind == "cylinder":
        shape = UsdGeom.Cylinder.Define(stage, f"{path}/shape")
        shape.CreateRadiusAttr(size_units[0])
        shape.CreateHeightAttr(size_units[1])
        shape.CreateAxisAttr("Z")
    elif kind == "sphere":
        shape = UsdGeom.Sphere.Define(stage, f"{path}/shape")
        shape.CreateRadiusAttr(size_units[0])
    else:
        shape = UsdGeom.Cube.Define(stage, f"{path}/shape")
        shape.CreateSizeAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
    if kind == "cylinder":
        w, x, y, z = _turn_z_to(axis)
        shape.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    elif kind == "box":
        shape.AddScaleOp().Set(Gf.Vec3d(*size_units))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    if opacity < 1.0:
        shape.CreateDisplayOpacityAttr([opacity])


def _attribute(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute else None


def draw(stage: Any, source: Path, scene_m: Callable[[], float]) -> list[ApparatusShape]:
    """Draw the motors and magnets of ``stage`` into its session layer, following their frames, and say what was drawn.
    ``scene_m`` is the scene's diagonal in metres (called only if a placeholder needs it)."""
    meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
    document = saved_document(source)
    codes = _time_codes(stage)
    found: list[ApparatusShape] = []
    magnets: list[tuple[Any, str, str]] = []
    motors: list[Any] = []
    for prim in stage.Traverse():
        if _attribute(prim, MAGNET_API_DIRECTION) is not None:
            parts = prim.GetPath().pathString.split("/")
            owner = stage.GetPrimAtPath(f"/World/{parts[2]}") if len(parts) >= 4 and parts[1] == "World" else None
            if owner is not None and _attribute(owner, "mime:thing") == "apparatus":
                magnets.append((prim, parts[2], parts[-1]))
        elif str(_attribute(prim, SEGMENT) or "").split(" ")[0] == "motor":
            motors.append(prim)
    sizes: dict[str, float] = {}
    if not magnets and not motors:
        return found
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Scope.Define(stage, APPARATUS_ROOT)
        for prim, apparatus, key in magnets:
            name = f"{apparatus}.{key}"
            path = f"{APPARATUS_ROOT}/{apparatus}_{key}"
            held = _declared(document, apparatus, key)
            drawn = _size(held[0], held[1]) if held is not None else None
            if held is not None and drawn is not None:
                (size, axis), kind, source_of, rule = drawn, held[0], "declared", ""
                colour, opacity = _STEEL, 1.0
            else:
                direction = tuple(float(v) for v in _attribute(prim, MAGNET_API_DIRECTION))
                edge = PLACEHOLDER_FRACTION * scene_m()
                size, axis, kind, source_of = (edge / 2.0, edge), direction, "cylinder", "placeholder"
                why = ("the view has no saved experiment to read its size from" if document is None else
                       f"its solid ({held[0] if held else 'none declared'}) is not one whose declared size the view reads")
                rule = (f"{why}, so a cylinder along its direction, its length and diameter each 1/{round(1 / PLACEHOLDER_FRACTION)} "
                        f"of the scene's diagonal")
                colour, opacity = _PLACEHOLDER, _PLACEHOLDER_OPACITY
                path += "_placeholder"
            _following(stage, path, prim, codes)
            _solid(stage, path, kind, [v / meters for v in size], axis, (0.0, 0.0, 0.0), colour, opacity)
            found.append(ApparatusShape(path, prim.GetPath().pathString, name, kind, tuple(size), source_of, rule))
            sizes[prim.GetPath().pathString] = max(size) if kind == "box" else max([2.0 * size[0], *size[1:]])
        for motor in motors:
            on = [extent for at, extent in sizes.items() if at.startswith(motor.GetPath().pathString + "/")]
            base = next((child for child in motor.GetChildren() if _attribute(child, "mime:frame") == "base"), None)
            if not on or base is None:
                continue
            apparatus = motor.GetPath().pathString.split("/")[2]
            axis = _motor_axis(document, apparatus, motor.GetName())
            length = max(on)
            path = f"{APPARATUS_ROOT}/{apparatus}_{motor.GetName()}_housing_placeholder"
            _following(stage, path, base, codes)
            back = tuple(-0.5 * length * v / meters for v in _unit(axis))
            _solid(stage, path, "cylinder", [0.5 * length / meters, length / meters], axis, back, _HOUSING,
                   _PLACEHOLDER_OPACITY)
            found.append(ApparatusShape(
                path, base.GetPath().pathString, f"{apparatus}.{motor.GetName()}", "cylinder", (0.5 * length, length),
                "placeholder",
                "no description declares a motor's size or which side of the rotor it is on, so a cylinder along its "
                "axis, its length the largest extent of the magnets on its rotor, its radius half that, ending in the "
                "rotor's plane on the side the axis points away from"))
    return found


def _motor_axis(document: Mapping[str, Any] | None, apparatus: str, segment: str) -> tuple[float, float, float]:
    """The motor's axis in its frame as the saved experiment declares it; z where it is not declared."""
    thing: Any = cast(Mapping[str, Any], document.get("things", {})).get(apparatus) if document is not None else None
    chain: Any = cast(dict[str, Any], thing).get("chain", []) if isinstance(thing, dict) else []
    for entry in cast(list[Any], chain) if isinstance(chain, list) else []:
        if isinstance(entry, dict) and cast(dict[str, Any], entry).get("segment") == segment:
            settings: Any = cast(dict[str, Any], entry).get("settings", {})
            found = _vector(cast(dict[str, Any], settings).get("axis")) if isinstance(settings, dict) else None
            if found is not None:
                return found
    return 0.0, 0.0, 1.0


def _time_codes(stage: Any) -> list[float]:
    """Every time code of the stage's range, one apart (the recording's samples): none for a stage with no range."""
    if not stage.HasAuthoredTimeCodeRange():
        return []
    start, end = float(stage.GetStartTimeCode()), float(stage.GetEndTimeCode())
    return [start + k for k in range(int(round(end - start)) + 1)] if end > start else [start]


__all__ = ["APPARATUS_ROOT", "PLACEHOLDER_FRACTION", "ApparatusShape", "draw", "saved_document"]
