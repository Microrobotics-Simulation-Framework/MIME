"""A stage, a recording, or a run or saved folder's, as one glTF 2.0 binary (GLB) that may be shared: the scene's
shapes as triangles, the recording's poses as one animation, and only the baked meshes whose terms let them travel.

glTF is +Y up and in metres, and a viewer frames what it holds, so the GLB's one root node turns the stage's up axis
to +Y and scales its units to a glTF unit of a power of ten metres chosen so that the scene's diagonal is 1 to 10
units (a 20 um ball is not left at 2e-5), the unit in ``asset.extras.mime``. A recording shorter than four simulated
seconds is slowed to about eight seconds of display (``slow`` display seconds per simulated second, in the extras
too, so that a page reads simulated time back), and eight invisible points span the whole motion's bounds, so that a
viewer frames all of it rather than the first pose. The meshes come from the overlay of the asset cache restricted to
the shareable tiers (``mime.viz._tiers``): a mesh left out is drawn as its frames' simple shapes, counted in
``GltfResult.withheld`` and said in its text. A vector's arrow is drawn at most ``ARROW_BODY_FACTOR`` times the largest
extent of the body it is drawn at, and at most ``ARROW_CAP_FRACTION`` of the diagonal of the scene without its arrows
(the lesser; the scene's alone for an arrow at no body), in the view alone and said in its text: the stage and the
recording hold the arrow at the length its prim says. Nothing is written into a saved or a run folder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mime.scene import ARROW, VECTOR_SCALE, VECTOR_UNITS, VECTOR_VALUE
from mime.scene._pxr import Gf, Usd, UsdGeom
from mime.viz._apparatus import APPARATUS_ROOT, ApparatusShape, draw, time_codes
from mime.viz._gltf import Settings, convert, up_turn
from mime.viz._primitives import SEGMENTS
from mime.viz._tiers import SHAREABLE_TIERS, refuse_unshareable
from mime.viz._visuals import _target, view  # pyright: ignore[reportPrivateUsage]

#: A recording shorter than this many simulated seconds is slowed to about ``SHOWN_S`` of display.
SLOWED_BELOW_S = 4.0
SHOWN_S = 8.0
#: The most time codes the motion's bounds are read at (evenly spaced, the first and last among them).
BOUNDS_SAMPLES = 512
#: The longest an arrow is drawn, as a fraction of the diagonal of the scene's motion without the arrows. The scene
#: without them is the choice because the viewer frames all of what the view holds: an arrow as long as the field's
#: value says would make the frame its own and shrink the bodies and the walls it is there to point at. This cap alone
#: does not make an arrow readable beside a body a thousandth of the scene (an arm and a millimetre body), which is
#: what ``ARROW_BODY_FACTOR`` is for; this one is the arrow's cap where it points at no body, and the bound on the others.
ARROW_CAP_FRACTION = 0.25
#: The longest an arrow is drawn at a body, as a multiple of the body's largest extent (the longest side of the box of
#: its own geometry in the stage, whatever its pose): long enough to read its direction beside the body, short enough
#: to stay near it.
ARROW_BODY_FACTOR = 2.0
#: How close, as a fraction of the body's largest extent, a vector prim's origin must lie to a body prim's origin (at the
#: first and the last time code) for the arrow to be drawn at that body: the recording places both from the same pose.
AT_BODY_TOLERANCE = 1e-6
VECTOR_NAME = "mime:vector:name"
_PREFIXES = ((1.0, "m"), (1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm"))


@dataclass(frozen=True)
class ArrowCap:
    """One arrow drawn shorter than its prim says: the longest it is drawn (``drawn_m``), the longest it would be
    (``would_be_m``), both in metres, and whether the body it is drawn at (``"body"``) or the scene (``"scene"``) set the
    cap."""

    drawn_m: float
    would_be_m: float
    rule: str


@dataclass(frozen=True)
class ArrowKey:
    """One vector the view draws as an arrow, for the page's legend: its prim, its name and unit as the stage states
    them, its colour, the length it is drawn at per unit (``metres_per_unit``) and, where the cap shortens it, the cap."""

    prim: str
    name: str
    unit: str
    colour: tuple[float, float, float]
    metres_per_unit: float
    cap: ArrowCap | None = None
    #: The most the view draws it long, in metres, where a cap holds it (whether or not it is shortened now).
    limit_m: float | None = None


@dataclass(frozen=True)
class BodyTrack:
    """A body's recorded position as the viewer's camera sees it: the display seconds of each sampled time code and the
    body prim's origin then, in the glTF scene's own coordinates (the unit and the up axis applied), and the body's
    largest extent in the same unit (``None`` for a body with no geometry)."""

    prim: str
    times: tuple[float, ...]
    positions: tuple[tuple[float, float, float], ...]
    extent: float | None = None


@dataclass
class GltfResult:
    """A GLB and what it says: the stage it came from, the experiment's name, the glTF unit in metres, the display
    seconds per simulated second, the simulated seconds the recording spans and its samples, the scene's diagonal in
    metres over its whole motion, the triangles, the meshes the overlay drew and left out (by set), the notices of
    every set the stage names, and the prims it could not draw (by type)."""

    glb: bytes
    source: Path
    name: str
    unit_m: float
    slow: float
    simulated_s: float
    samples: int
    across_m: float
    triangles: int
    drawn: int = 0
    withheld: dict[str, int] = field(default_factory=dict[str, int])
    missing: dict[str, int] = field(default_factory=dict[str, int])
    notices: list[str] = field(default_factory=list[str])
    skipped: dict[str, int] = field(default_factory=dict[str, int])
    fetch: dict[str, str] = field(default_factory=dict[str, str])
    #: The arrows drawn shorter than their prims say, by prim path.
    capped: dict[str, ArrowCap] = field(default_factory=dict[str, ArrowCap])
    #: Every vector drawn as an arrow, for a legend.
    arrows: list[ArrowKey] = field(default_factory=list[ArrowKey])
    #: Each body of the stage by name: where it is along the recording.
    bodies: dict[str, BodyTrack] = field(default_factory=dict[str, BodyTrack])
    #: The motors and magnets drawn for the view alone (:mod:`mime.viz._apparatus`), and whether they were asked for.
    shapes: list[ApparatusShape] = field(default_factory=list[ApparatusShape])
    apparatus_shapes: bool = True

    def lines(self) -> list[str]:
        """What a reader of the view is told: the meshes left out and missing, then every set's terms."""
        held: list[str] = []
        for name, count in sorted(self.withheld.items()):
            held.append(f"{count} meshes of {name} are not in this view: their licence does not let them be shared "
                        f"(or no record says it does), so simple shapes from its frames are drawn in their place; "
                        f"a USD viewer draws them on this machine (mime visuals view)")
        for name, count in sorted(self.missing.items()):
            held.append(f"{count} meshes of {name} are not in the asset cache, so simple shapes from its frames are "
                        f"drawn in their place (mime visuals fetch {self.fetch.get(name, name)})")
        for kind, count in sorted(self.skipped.items()):
            held.append(f"{count} prims of type {kind} are not drawn: the view draws meshes, spheres, cylinders, "
                        f"cubes and capsules")
        if self.capped:
            names = sorted(self.capped)
            noun = "arrow is" if len(names) == 1 else "arrows are"
            shown = ", ".join(f"{name.removeprefix('/World/')} drawn at {length_text(self.capped[name].drawn_m)}, not "
                              f"{length_text(self.capped[name].would_be_m)}" for name in names[:3])
            more = f" and {len(names) - 3} more" if len(names) > 3 else ""
            held.append(f"{len(names)} {noun} drawn capped, so that an arrow stays beside what it points at: "
                        f"{ARROW_BODY_FACTOR:g} times the largest extent of its body, or {ARROW_CAP_FRACTION:.0%} of the "
                        f"scene's size without its arrows, whichever is less ({shown}{more}). A capped arrow still points "
                        f"along its vector; the stage and the recording hold its full length.")
        for shape in self.shapes:
            if shape.placeholder:
                held.append(f"{shape.name} is drawn as a placeholder, a {shape.kind} of {_size_text(shape)}, for the view "
                            f"alone: {shape.rule}; the stage holds no such shape.")
        return held + self.notices

    def size_text(self) -> str:
        """The scene's diagonal over its whole motion, in the SI unit that puts it between 1 and 1000."""
        return length_text(self.across_m)


def _size_text(shape: ApparatusShape) -> str:
    sizes = [length_text(v) for v in shape.size_m]
    if shape.kind == "cylinder":
        return f"radius {sizes[0]}, length {sizes[1]}"
    return f"radius {sizes[0]}" if shape.kind == "sphere" else " by ".join(sizes)


def length_text(metres: float) -> str:
    for factor, unit in _PREFIXES:
        if metres >= factor or factor == _PREFIXES[-1][0]:
            return f"{metres / factor:.3g} {unit}"
    return f"{metres:.3g} m"


def _codes(stage: Any) -> list[float]:
    if not stage.HasAuthoredTimeCodeRange():
        return []
    start, end = float(stage.GetStartTimeCode()), float(stage.GetEndTimeCode())
    if end <= start:
        return [start]
    count = int(round(end - start)) + 1
    if count <= BOUNDS_SAMPLES:
        return [start + k for k in range(count)]
    return [float(v) for v in np.linspace(start, end, BOUNDS_SAMPLES)]


def _outside(stage: Any, left: Sequence[Any]) -> list[Any]:
    """The prims that together hold everything below the pseudo-root but the subtrees of ``left`` (prims): the whole
    tree's children where ``left`` is empty. A prim that has a left prim below it is walked through, not measured."""
    paths = {prim.GetPath() for prim in left}
    above = {prefix for path in paths for prefix in path.GetPrefixes()[:-1]}
    found: list[Any] = []

    def walk(prim: Any) -> None:
        if prim.GetPath() in paths:
            return
        if prim.GetPath() in above:
            for child in prim.GetChildren():
                walk(child)
        else:
            found.append(prim)

    for child in stage.GetPseudoRoot().GetChildren():
        walk(child)
    return found


def motion_bounds(stage: Any, purposes: tuple[str, ...], leaving: Sequence[Any] = (),
                  only: Sequence[Any] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """The union of the drawn prims' world bounds over the stage's time codes, in its units, leaving out the subtrees
    of the prims ``leaving``, or of just the subtrees of the prims ``only``; ``None`` when it draws nothing."""
    tokens = [UsdGeom.Tokens.default_ if purpose == "default" else getattr(UsdGeom.Tokens, purpose) for purpose in purposes]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    measured = list(only) if only is not None else _outside(stage, leaving) if leaving else [stage.GetPseudoRoot()]
    for code in _codes(stage) or [Usd.TimeCode.Default()]:
        time = Usd.TimeCode(code) if isinstance(code, float) else code
        cache = UsdGeom.BBoxCache(time, tokens, useExtentsHint=False)
        for prim in measured:
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            low = np.minimum(low, np.asarray(box.GetMin(), dtype=np.float64))
            high = np.maximum(high, np.asarray(box.GetMax(), dtype=np.float64))
    if not np.all(np.isfinite(low)):
        return None
    return low, high


@dataclass
class Arrows:
    """The stage's vector arrows against their caps, in the stage's units, by prim path: the longest each is drawn over
    the motion (``longest``), the most each is drawn (``caps``, with ``rules``: ``"body"`` or ``"scene"``, whichever set
    it; an arrow with no cap is absent), the arrows' scale at most ``limits``, the scene's cap (``scene``; ``None`` where
    the scene without arrows has no extent to be a fraction of), and the box of the motion as the view draws it
    (``bounds``: with every arrow at most its cap, or as the stage has it where none is longer than its cap)."""

    scene: float | None = None
    longest: dict[str, float] = field(default_factory=dict[str, float])
    caps: dict[str, float] = field(default_factory=dict[str, float])
    rules: dict[str, str] = field(default_factory=dict[str, str])
    limits: dict[str, float] = field(default_factory=dict[str, float])
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def capped(self) -> dict[str, float]:
        return {path: length for path, length in self.longest.items() if path in self.caps and length > self.caps[path]}


def _drawn(prim: Any) -> bool:
    imageable = UsdGeom.Imageable(prim)
    return str(imageable.ComputeVisibility(Usd.TimeCode.EarliestTime())) != str(UsdGeom.Tokens.invisible)


def _rest(prim: Any, meters: float) -> float:
    """An arrow's length at scale 1, in the stage's units: its cylinder's height."""
    child = prim.GetChild(ARROW)
    height = child.GetAttribute("height").Get() if child and child.GetAttribute("height") else None
    return float(height) if height is not None else 1.0 / meters


def _origin(prim: Any, code: float | None) -> np.ndarray:
    time = Usd.TimeCode(code) if code is not None else Usd.TimeCode.Default()
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    return np.asarray(matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=np.float64)


def _bodies(stage: Any, tokens: list[Any]) -> list[tuple[Any, float]]:
    """Each body's prim and the largest extent of its own geometry in the stage's units (the longest side of its
    box in its own axes, so that a turn does not change it); bodies with no geometry are left out."""
    found: list[tuple[Any, float]] = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.EarliestTime(), tokens, useExtentsHint=False)
    for prim in stage.Traverse():
        thing = prim.GetAttribute("mime:thing")
        if not thing or thing.Get() != "body":
            continue
        box = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        size = float(max(box.GetSize())) if not box.IsEmpty() else 0.0
        if size > 0.0 and math.isfinite(size):
            found.append((prim, size))
    return found


def _body_of(prim: Any, bodies: list[tuple[Any, float]], codes: list[float]) -> float | None:
    """The largest extent of the body the vector ``prim`` is drawn at: the body whose origin the prim's coincides with
    at the first and the last time code (a recording places both from the one pose; the stage, at the body's start),
    the nearest where two do; ``None`` where there is none."""
    ends = [codes[0], codes[-1]] if codes else [None]
    mine = [_origin(prim, code) for code in ends]
    best: tuple[float, float] | None = None
    for body, size in bodies:
        gap = max(float(np.linalg.norm(_origin(body, code) - at)) for code, at in zip(ends, mine))
        if gap <= AT_BODY_TOLERANCE * size and (best is None or gap < best[0]):
            best = (gap, size)
    return best[1] if best is not None else None


def arrows_of(stage: Any, purposes: tuple[str, ...]) -> Arrows:
    """Each drawn vector prim's longest arrow over the motion, and the cap on it: the lesser of ``ARROW_BODY_FACTOR``
    times the largest extent of the body it is drawn at and ``ARROW_CAP_FRACTION`` of the diagonal of the motion
    without any arrow (the scene's alone where it is drawn at no body). The box of the motion is the stage's own where
    no arrow is longer than its cap, and else the motion without arrows with every arrow at most its cap."""
    prims = [prim for prim in stage.Traverse() if prim.HasAttribute(VECTOR_VALUE)]
    if not prims:
        return Arrows()
    tokens = [UsdGeom.Tokens.default_ if purpose == "default" else getattr(UsdGeom.Tokens, purpose) for purpose in purposes]
    meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
    scene = motion_bounds(stage, purposes, prims)
    drawn = [prim for prim in prims if _drawn(prim)]
    diagonal = float(np.linalg.norm(scene[1] - scene[0])) if scene is not None else 0.0
    fraction = ARROW_CAP_FRACTION * diagonal if math.isfinite(ARROW_CAP_FRACTION) and diagonal > 0.0 else None
    found = Arrows(scene=fraction)
    codes = _codes(stage)
    bodies = _bodies(stage, tokens) if math.isfinite(ARROW_BODY_FACTOR) else []
    low, high = (scene if scene is not None else (np.full(3, np.inf), np.full(3, -np.inf)))
    low, high = low.copy(), high.copy()
    for prim in drawn:
        path = str(prim.GetPath())
        size = _body_of(prim, bodies, codes) if bodies else None
        cap, rule = fraction, "scene"
        if size is not None and (cap is None or ARROW_BODY_FACTOR * size < cap):
            cap, rule = ARROW_BODY_FACTOR * size, "body"
        rest = _rest(prim, meters)
        if cap is not None:
            found.caps[path], found.rules[path], found.limits[path] = cap, rule, cap / rest
        xformable = UsdGeom.Xformable(prim)
        longest = 0.0
        for code in codes or [Usd.TimeCode.Default()]:
            time = Usd.TimeCode(code) if isinstance(code, float) else code
            matrix = xformable.ComputeLocalToWorldTransform(time)
            origin = np.asarray(matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=np.float64)
            tip = np.asarray(matrix.Transform(Gf.Vec3d(0.0, 0.0, rest)), dtype=np.float64)
            length = float(np.linalg.norm(tip - origin))
            longest = max(longest, length)
            held = origin + (tip - origin) * (min(1.0, cap / length) if cap is not None and length > 0.0 else 1.0)
            low, high = np.minimum(low, np.minimum(origin, held)), np.maximum(high, np.maximum(origin, held))
        found.longest[path] = longest
    if found.capped():
        found.bounds = (low, high) if np.all(np.isfinite(low)) else None
    else:
        found.bounds = _joined(scene, motion_bounds(stage, purposes, only=prims))
    return found


def _joined(first: tuple[np.ndarray, np.ndarray] | None,
            second: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray] | None:
    """The box of two boxes."""
    if first is None or second is None:
        return first or second
    return np.minimum(first[0], second[0]), np.maximum(first[1], second[1])


def slowed(simulated_s: float) -> float:
    """The display seconds per simulated second: about ``SHOWN_S`` of display for a recording shorter than
    ``SLOWED_BELOW_S``, to two significant figures; else 1."""
    if not simulated_s > 0.0 or simulated_s >= SLOWED_BELOW_S:
        return 1.0
    return float(f"{SHOWN_S / simulated_s:.2g}")


def _name(stage: Any, source: Path) -> str:
    for layer in stage.GetLayerStack():
        held: Any = layer.customLayerData or {}
        recording: Any = held.get("mime:recording")
        if isinstance(recording, dict) and recording.get("experiment"):  # pyright: ignore[reportUnknownMemberType]
            return str(recording.get("experiment"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    folder = source.parent.parent if source.parent.name == "scene" else source.parent
    return folder.name or source.stem


def _turned(turn: tuple[float, float, float, float], vector: np.ndarray) -> np.ndarray:
    """``vector`` (rows of three) turned by the unit quaternion ``turn`` (x, y, z, w)."""
    imaginary, real = np.asarray(turn[:3]), turn[3]
    across = np.cross(imaginary, vector)
    return vector + 2.0 * np.cross(imaginary, across + real * vector)


def body_tracks(stage: Any, tokens: list[Any], *, scale: float, turn: tuple[float, float, float, float],
                shown: float) -> dict[str, BodyTrack]:
    """Each body prim's origin at *every* time code of the stage's range (the recording's own samples, not the at most
    ``BOUNDS_SAMPLES`` the motion's box is read at: a camera that follows must be on the body at each frame the
    animation draws), in the glTF's coordinates (the stage's units scaled by ``scale``, the up axis turned by ``turn``)
    against the display seconds (``shown`` per simulated second) the converter's animation uses; one sample at the
    default time without a range."""
    codes = time_codes(stage)
    rate = float(stage.GetTimeCodesPerSecond()) or 1.0
    start = float(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else 0.0
    extents = {str(prim.GetPath()): size for prim, size in _bodies(stage, tokens)}
    found: dict[str, BodyTrack] = {}
    for prim in stage.Traverse():
        thing = prim.GetAttribute("mime:thing")
        if not thing or thing.Get() != "body":
            continue
        named = prim.GetAttribute("mime:name")
        name = str(named.Get()) if named and named.Get() else prim.GetName()
        times: list[float] = []
        points: list[np.ndarray] = []
        for code in codes or [None]:
            times.append((code - start) / rate * shown if code is not None else 0.0)
            points.append(_origin(prim, code))
        placed = _turned(turn, np.stack(points) * scale)
        extent = extents.get(str(prim.GetPath()))
        found[name] = BodyTrack(str(prim.GetPath()), tuple(times),
                                tuple((float(p[0]), float(p[1]), float(p[2])) for p in placed),
                                extent * scale if extent is not None else None)
    return found


def _arrow_keys(stage: Any, capped: dict[str, ArrowCap], limits_m: dict[str, float]) -> list[ArrowKey]:
    keys: list[ArrowKey] = []
    for prim in stage.Traverse():
        if not prim.HasAttribute(VECTOR_VALUE):
            continue
        named, unit, per = prim.GetAttribute(VECTOR_NAME), prim.GetAttribute(VECTOR_UNITS), prim.GetAttribute(VECTOR_SCALE)
        child = prim.GetChild(ARROW)
        colour = UsdGeom.PrimvarsAPI(child).GetPrimvar("displayColor").Get() if child else None
        rgb = (float(colour[0][0]), float(colour[0][1]), float(colour[0][2])) if colour else (0.8, 0.8, 0.8)
        path = str(prim.GetPath())
        keys.append(ArrowKey(path, str(named.Get()) if named and named.Get() else prim.GetName(),
                             str(unit.Get()) if unit and unit.Get() else "", rgb,
                             float(per.Get()) if per and per.Get() else 1.0, capped.get(path), limits_m.get(path)))
    return keys


def gltf(target: str | Path, *, slow: float | None = None, segments: int = SEGMENTS,
         apparatus_shapes: bool = True) -> GltfResult:
    """``target`` (a stage or recording file, a run folder or a saved folder) as a GLB that may be shared: the overlay
    of the shareable tiers composed, the composed stage refused if any other baked mesh reaches it
    (``visuals.not-shareable``), the up axis turned to +Y, the units scaled, the recording slowed by ``slow`` (chosen
    when ``None``: :func:`slowed`) and framed over its whole motion. With ``apparatus_shapes`` the apparatus's motors
    and magnets are drawn for the view alone (:mod:`mime.viz._apparatus`); nothing is written into the stage."""
    source = _target(Path(target))
    path, found = view(source, tiers=SHAREABLE_TIERS)
    stage = Usd.Stage.Open(str(path))
    refuse_unshareable(stage, Path(target))
    purposes = ("default", "proxy")

    def scene_m() -> float:
        """The scene's diagonal in metres without its arrows and without the view's own shapes."""
        vectors = [prim for prim in stage.Traverse() if prim.HasAttribute(VECTOR_VALUE)]
        overlay = stage.GetPrimAtPath(APPARATUS_ROOT)
        box = motion_bounds(stage, purposes, [*vectors, *([overlay] if overlay else [])])
        return float(np.linalg.norm(box[1] - box[0])) * (float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0) if box else 0.0

    shapes = draw(stage, source, cache(scene_m)) if apparatus_shapes else []
    arrows = arrows_of(stage, purposes)
    bounds = arrows.bounds if arrows.bounds is not None else motion_bounds(stage, purposes)
    meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
    across = float(np.linalg.norm(bounds[1] - bounds[0])) * meters if bounds is not None else meters
    unit = 10.0 ** math.floor(math.log10(across)) if across > 0.0 else meters
    codes = _codes(stage)
    rate = float(stage.GetTimeCodesPerSecond()) or 1.0
    simulated = (codes[-1] - codes[0]) / rate if len(codes) > 1 else 0.0
    shown = slowed(simulated) if slow is None else slow
    frame = None
    if bounds is not None:
        low, high = bounds
        frame = ((float(low[0]), float(low[1]), float(low[2])), (float(high[0]), float(high[1]), float(high[2])))
    turn = up_turn(str(UsdGeom.GetStageUpAxis(stage)))
    settings = Settings(scale=meters / unit, turn=turn, slow=shown,
                        segments=segments, frame=frame, purposes=purposes, scale_limits=arrows.limits)
    samples = int(round(float(stage.GetEndTimeCode()) - float(stage.GetStartTimeCode()))) + 1 \
        if stage.HasAuthoredTimeCodeRange() else 0
    name = _name(stage, source)
    drawn_shapes = [{"prim": s.prim, "name": s.name, "kind": s.kind, "size_m": list(s.size_m), "source": s.source}
                    for s in shapes]
    extras = {"mime": {"name": name, "unit_m": unit, "slow": shown, "simulated_s": simulated, "samples": samples,
                       "time_codes_per_second": rate, "withheld": dict(found.withheld), "missing": dict(found.missing),
                       "notices": list(found.notices), "apparatus_shapes": drawn_shapes}}
    credit = "\n".join(found.terms[name] for name in found.sets if name in found.terms)
    converted = convert(stage, settings, extras=extras, copyright=credit or None)
    capped = {prim: ArrowCap(arrows.caps[prim] * meters, length * meters, arrows.rules[prim])
              for prim, length in arrows.capped().items()}
    tokens = [UsdGeom.Tokens.default_ if purpose == "default" else getattr(UsdGeom.Tokens, purpose) for purpose in purposes]
    tracks = body_tracks(stage, tokens, scale=meters / unit, turn=turn, shown=shown)
    return GltfResult(converted.glb, source, name, unit, shown, simulated, samples, across, converted.triangles,
                      found.drawn, dict(found.withheld), dict(found.missing), list(found.notices),
                      dict(converted.skipped), dict(found.fetch), capped, _arrow_keys(stage, capped, {p: c * meters for p, c in arrows.caps.items()}), tracks, shapes,
                      apparatus_shapes)


__all__ = ["ARROW_BODY_FACTOR", "ARROW_CAP_FRACTION", "ArrowCap", "ArrowKey", "BodyTrack", "GltfResult", "body_tracks", "arrows_of", "gltf", "length_text", "motion_bounds", "slowed"]
