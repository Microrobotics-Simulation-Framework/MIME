"""A stage, a recording, or a run or saved folder's, as one glTF 2.0 binary (GLB) that may be shared: the scene's
shapes as triangles, the recording's poses as one animation, and only the baked meshes whose terms let them travel.

glTF is +Y up and in metres, and a viewer frames what it holds, so the GLB's one root node turns the stage's up axis
to +Y and scales its units to a glTF unit of a power of ten metres chosen so that the scene's diagonal is 1 to 10
units (a 20 um ball is not left at 2e-5), the unit in ``asset.extras.mime``. A recording shorter than four simulated
seconds is slowed to about eight seconds of display (``slow`` display seconds per simulated second, in the extras
too, so that a page reads simulated time back), and eight invisible points span the whole motion's bounds, so that a
viewer frames all of it rather than the first pose. The meshes come from the overlay of the asset cache restricted to
the shareable tiers (``mime.viz._tiers``): a mesh left out is drawn as its frames' simple shapes, counted in
``GltfResult.withheld`` and said in its text. Nothing is written into a saved or a run folder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mime.scene._pxr import Usd, UsdGeom
from mime.viz._gltf import Settings, convert, up_turn
from mime.viz._primitives import SEGMENTS
from mime.viz._tiers import SHAREABLE_TIERS, refuse_unshareable
from mime.viz._visuals import _target, view  # pyright: ignore[reportPrivateUsage]

#: A recording shorter than this many simulated seconds is slowed to about ``SHOWN_S`` of display.
SLOWED_BELOW_S = 4.0
SHOWN_S = 8.0
#: The most time codes the motion's bounds are read at (evenly spaced, the first and last among them).
BOUNDS_SAMPLES = 512
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


def motion_bounds(stage: Any, purposes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    """The union of the drawn prims' world bounds over the stage's time codes, in its units; ``None`` when it draws
    nothing."""
    tokens = [UsdGeom.Tokens.default_ if purpose == "default" else getattr(UsdGeom.Tokens, purpose) for purpose in purposes]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for code in _codes(stage) or [Usd.TimeCode.Default()]:
        time = Usd.TimeCode(code) if isinstance(code, float) else code
        cache = UsdGeom.BBoxCache(time, tokens, useExtentsHint=False)
        box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        low = np.minimum(low, np.asarray(box.GetMin(), dtype=np.float64))
        high = np.maximum(high, np.asarray(box.GetMax(), dtype=np.float64))
    if not np.all(np.isfinite(low)):
        return None
    return low, high


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
    bounds = motion_bounds(stage, purposes)
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
                        segments=segments, frame=frame, purposes=purposes)
    samples = int(round(float(stage.GetEndTimeCode()) - float(stage.GetStartTimeCode()))) + 1 \
        if stage.HasAuthoredTimeCodeRange() else 0
    name = _name(stage, source)
    extras = {"mime": {"name": name, "unit_m": unit, "slow": shown, "simulated_s": simulated, "samples": samples,
                       "time_codes_per_second": rate, "withheld": dict(found.withheld), "missing": dict(found.missing),
                       "notices": list(found.notices)}}
    credit = "\n".join(found.terms[name] for name in found.sets if name in found.terms)
    converted = convert(stage, settings, extras=extras, copyright=credit or None)
    return GltfResult(converted.glb, source, name, unit, shown, simulated, samples, across, converted.triangles,
                      found.drawn, dict(found.withheld), dict(found.missing), list(found.notices),
                      dict(converted.skipped), dict(found.fetch))


__all__ = ["GltfResult", "gltf", "length_text", "motion_bounds", "slowed"]
