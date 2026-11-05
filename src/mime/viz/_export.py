"""A stage, a recording, or a run or saved folder's, as one glTF 2.0 binary (GLB) that may be shared: the scene's
shapes as triangles, the recording's poses as one animation, and only the baked meshes whose terms let them travel.

glTF is +Y up and in metres, and a viewer frames what it holds, so the GLB's one root node turns the stage's up axis
to +Y and scales its units to a glTF unit of a power of ten metres chosen so that the scene's diagonal is 1 to 10
units (a 20 um ball is not left at 2e-5), the unit in ``asset.extras.mime``. A recording shorter than four simulated
seconds is slowed to about eight seconds of display (``slow`` display seconds per simulated second, in the extras
too, so that a page reads simulated time back), and eight invisible points span the whole motion's bounds, so that a
viewer frames all of it rather than the first pose. The meshes come from the overlay of the asset cache restricted to
the shareable tiers (``mime.viz._tiers``): a mesh left out is drawn as its frames' simple shapes, counted in
``GltfResult.withheld`` and said in its text. A vector's arrow is drawn at most ``ARROW_CAP_FRACTION`` of the scene's
diagonal long (the scene without its arrows), in the view alone and said in its text: the stage and the recording hold
the arrow at the length its prim says. Nothing is written into a saved or a run folder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mime.scene import ARROW, VECTOR_VALUE
from mime.scene._pxr import Gf, Usd, UsdGeom
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
#: value says (8.3 m per tesla at 10 mT: 8 cm beside a millimetre body) would make the frame its own, and shrink the
#: bodies and the walls it is there to point at; a quarter of what they alone need keeps the frame within 1.5 times
#: theirs and the arrow longer than any body in it that is not itself the scene. A body's size is not the measure:
#: a 20 um ball in a centimetre tube would have its arrow drawn at 60 um, which cannot be seen.
ARROW_CAP_FRACTION = 0.25
_PREFIXES = ((1.0, "m"), (1e-3, "mm"), (1e-6, "µm"), (1e-9, "nm"))


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
    #: The arrows drawn shorter than their prims say: each one's name and the longest it would be, in metres, and the
    #: length they are drawn at most.
    capped: dict[str, float] = field(default_factory=dict[str, float])
    arrow_cap_m: float = 0.0

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
            shown = ", ".join(f"{name} ({length_text(self.capped[name])})" for name in names[:3])
            more = f" and {len(names) - 3} more" if len(names) > 3 else ""
            noun = "arrow is" if len(names) == 1 else "arrows are"
            held.append(f"{len(names)} {noun} drawn capped at {length_text(self.arrow_cap_m)}, "
                        f"{ARROW_CAP_FRACTION:.0%} of the scene's size without its arrows, so that an arrow does not "
                        f"outgrow what it points at: {shown}{more}, each with the length its vector would have. A capped "
                        f"arrow still points along its vector; the stage and the recording hold its full length.")
        return held + self.notices

    def size_text(self) -> str:
        """The scene's diagonal over its whole motion, in the SI unit that puts it between 1 and 1000."""
        return length_text(self.across_m)


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


def motion_bounds(stage: Any, purposes: tuple[str, ...],
                  leaving: Sequence[Any] = ()) -> tuple[np.ndarray, np.ndarray] | None:
    """The union of the drawn prims' world bounds over the stage's time codes, in its units, leaving out the subtrees
    of the prims ``leaving``; ``None`` when it draws nothing."""
    tokens = [UsdGeom.Tokens.default_ if purpose == "default" else getattr(UsdGeom.Tokens, purpose) for purpose in purposes]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    measured = _outside(stage, leaving) if leaving else [stage.GetPseudoRoot()]
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
    """The stage's vector arrows against a cap: the drawn ones (``vectors``), the longest each is drawn over the
    motion in the stage's units (``longest``, by prim path), the cap in the stage's units (``cap``; ``None`` where the
    scene without arrows has no extent to be a fraction of), the arrows' scale at most ``limits`` (by prim path), and
    the box of the motion with every arrow at most the cap (``bounds``; ``None`` when none is longer than the cap)."""

    cap: float | None = None
    longest: dict[str, float] = field(default_factory=dict[str, float])
    limits: dict[str, float] = field(default_factory=dict[str, float])
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def capped(self) -> dict[str, float]:
        return {path: length for path, length in self.longest.items() if self.cap is not None and length > self.cap}


def _drawn(prim: Any) -> bool:
    imageable = UsdGeom.Imageable(prim)
    return str(imageable.ComputeVisibility(Usd.TimeCode.EarliestTime())) != str(UsdGeom.Tokens.invisible)


def _rest(prim: Any, meters: float) -> float:
    """An arrow's length at scale 1, in the stage's units: its cylinder's height."""
    child = prim.GetChild(ARROW)
    height = child.GetAttribute("height").Get() if child and child.GetAttribute("height") else None
    return float(height) if height is not None else 1.0 / meters


def arrows_of(stage: Any, purposes: tuple[str, ...]) -> Arrows:
    """Each drawn vector prim's longest arrow over the motion, and the cap on them: ``ARROW_CAP_FRACTION`` of the
    diagonal of the motion without any arrow. The bounds of the motion with the arrows capped are held only where
    one is longer than the cap; a scene with none is framed as before."""
    prims = [prim for prim in stage.Traverse() if prim.HasAttribute(VECTOR_VALUE)]
    if not prims:
        return Arrows()
    meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
    bodies = motion_bounds(stage, purposes, prims)
    drawn = [prim for prim in prims if _drawn(prim)]
    diagonal = float(np.linalg.norm(bodies[1] - bodies[0])) if bodies is not None else 0.0
    cap = ARROW_CAP_FRACTION * diagonal if math.isfinite(ARROW_CAP_FRACTION) and diagonal > 0.0 else None
    found = Arrows(cap=cap)
    low, high = (bodies if bodies is not None else (np.full(3, np.inf), np.full(3, -np.inf)))
    low, high = low.copy(), high.copy()
    for prim in drawn:
        rest = _rest(prim, meters)
        if cap is not None:
            found.limits[str(prim.GetPath())] = cap / rest
        xformable = UsdGeom.Xformable(prim)
        longest = 0.0
        for code in _codes(stage) or [Usd.TimeCode.Default()]:
            time = Usd.TimeCode(code) if isinstance(code, float) else code
            matrix = xformable.ComputeLocalToWorldTransform(time)
            origin = np.asarray(matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=np.float64)
            tip = np.asarray(matrix.Transform(Gf.Vec3d(0.0, 0.0, rest)), dtype=np.float64)
            length = float(np.linalg.norm(tip - origin))
            longest = max(longest, length)
            held = origin + (tip - origin) * (min(1.0, cap / length) if cap is not None and length > 0.0 else 1.0)
            low, high = np.minimum(low, np.minimum(origin, held)), np.maximum(high, np.maximum(origin, held))
        found.longest[str(prim.GetPath())] = longest
    if found.capped() and np.all(np.isfinite(low)):
        found.bounds = (low, high)
    return found


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


def gltf(target: str | Path, *, slow: float | None = None, segments: int = SEGMENTS) -> GltfResult:
    """``target`` (a stage or recording file, a run folder or a saved folder) as a GLB that may be shared: the overlay
    of the shareable tiers composed, the composed stage refused if any other baked mesh reaches it
    (``visuals.not-shareable``), the up axis turned to +Y, the units scaled, the recording slowed by ``slow`` (chosen
    when ``None``: :func:`slowed`) and framed over its whole motion."""
    source = _target(Path(target))
    path, found = view(source, tiers=SHAREABLE_TIERS)
    stage = Usd.Stage.Open(str(path))
    refuse_unshareable(stage, Path(target))
    purposes = ("default", "proxy")
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
    settings = Settings(scale=meters / unit, turn=up_turn(str(UsdGeom.GetStageUpAxis(stage))), slow=shown,
                        segments=segments, frame=frame, purposes=purposes, scale_limits=arrows.limits)
    samples = int(round(float(stage.GetEndTimeCode()) - float(stage.GetStartTimeCode()))) + 1 \
        if stage.HasAuthoredTimeCodeRange() else 0
    name = _name(stage, source)
    extras = {"mime": {"name": name, "unit_m": unit, "slow": shown, "simulated_s": simulated, "samples": samples,
                       "time_codes_per_second": rate, "withheld": dict(found.withheld), "missing": dict(found.missing),
                       "notices": list(found.notices)}}
    credit = "\n".join(found.terms[name] for name in found.sets if name in found.terms)
    converted = convert(stage, settings, extras=extras, copyright=credit or None)
    capped = {path.rsplit("/", 1)[-1]: length * meters for path, length in arrows.capped().items()}
    return GltfResult(converted.glb, source, name, unit, shown, simulated, samples, across, converted.triangles,
                      found.drawn, dict(found.withheld), dict(found.missing), list(found.notices),
                      dict(converted.skipped), dict(found.fetch), capped,
                      (arrows.cap or 0.0) * meters if capped else 0.0)


__all__ = ["ARROW_CAP_FRACTION", "GltfResult", "arrows_of", "gltf", "length_text", "motion_bounds", "slowed"]
