"""The recording layer: a run's poses and field vectors as time samples over its run folder's own stage.

``recording.usdc`` sits in the run folder, and its one sublayer is the folder's ``experiment/scene/scene.usda``, by a
path inside the folder: the recording is its experiment's by construction, and the folder still resolves when it is
moved. It holds overs on the bound prims (``BuiltExperiment.binding``): a pose's ``translate`` and ``orient`` (double
precision: a float32 state's values exactly), and a vector's ``mime:vector:value`` with its arrow (a metre long in the
stage) placed at the body, turned along the vector and scaled, whole, to its length. Its time code is the sample's
index; ``timeCodesPerSecond`` and ``framesPerSecond`` are the samples per simulated second, and ``customLayerData``
holds ``mime:recording`` with every sample's step. It is written as the oldest crate version usd-core writes for what
it holds (0.8.0), which OpenUSD 24.08 reads, and exported fresh, with no bytes of an earlier save."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mime.build import StageBinding
from mime.scene import STAGE_FILE, VECTOR_SCALE, VECTOR_VALUE, up_axis
from mime.scene._pxr import Gf, Sdf, Vt

RECORDING_FILE = "recording.usdc"
RECORDING_FORMAT = "mime.recording/1"
#: The run folder's saved form, whose stage the recording is over.
RUN_STAGE = f"experiment/{STAGE_FILE}"


@dataclass(frozen=True)
class RecordingSettings:
    """What a recording is written with, from the built experiment: the stage's up (a unit vector along +Y or +Z) and
    metres per unit, the base step (s), the experiment's name and its input hash."""

    up: tuple[float, float, float]
    meters_per_unit: float
    dt: float
    experiment: str
    input_hash: str


class StageSamples:
    """The bound fields of a run's state, taken at chunk ends: the step of each sample and each field's values,
    ``every`` base steps apart within a stretch of recording."""

    def __init__(self, binding: StageBinding, every: int) -> None:
        self.binding = binding
        self.every = every
        self.steps: list[int] = []
        self.values: dict[tuple[str, str], list[np.ndarray]] = {key: [] for key in binding.fields()}

    @property
    def first(self) -> int:
        return self.steps[0] if self.steps else 0

    def take(self, step: int, state: Mapping[str, Mapping[str, Any]]) -> None:
        """The bound fields as they stand after ``step`` steps; a step already taken is not taken twice."""
        if self.steps and self.steps[-1] == step:
            return
        self.steps.append(step)
        for node, name in self.values:
            self.values[(node, name)].append(np.array(state[node][name], copy=True))

    def stacked(self, key: tuple[str, str]) -> np.ndarray:
        return np.stack(self.values[key]) if self.values[key] else np.zeros((0,))


def _prim_spec(layer: Any, path: str) -> Any:
    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(path))
    return spec


def _attribute(prim: Any, name: str, kind: Any) -> Any:
    return prim.attributes[name] if name in prim.attributes else Sdf.AttributeSpec(prim, name, kind)


def _turn(vector: np.ndarray) -> Any:
    """The unit quaternion turning +Z onto ``vector``, the shorter way (identity for a zero vector)."""
    length = float(np.linalg.norm(vector))
    if not length > 0.0:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    d = vector / length
    w = 1.0 + float(d[2])
    if w < 1e-12:
        return Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
    q = np.array([w, -float(d[1]), float(d[0]), 0.0])
    q /= np.linalg.norm(q)
    return Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))


def recording_layer(samples: StageSamples, settings: RecordingSettings, *, stage: str = RUN_STAGE,
                    scales: Mapping[str, float] | None = None) -> Any:
    """The recording as an anonymous layer, its sublayer ``stage`` (a path relative to where it will be written)."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.subLayerPaths.append(stage)
    count = len(samples.steps)
    rate = 1.0 / (samples.every * settings.dt)
    layer.startTimeCode, layer.endTimeCode = 0.0, float(max(count - 1, 0))
    layer.timeCodesPerSecond = layer.framesPerSecond = rate
    root = layer.pseudoRoot
    root.SetInfo("upAxis", up_axis(settings.up))
    root.SetInfo("metersPerUnit", settings.meters_per_unit)
    root.SetInfo("defaultPrim", "World")
    layer.customLayerData = {"mime:recording": {
        "format": RECORDING_FORMAT, "experiment": settings.experiment, "input_hash": settings.input_hash,
        "dt_s": settings.dt, "every_steps": samples.every, "samples": count,
        "steps": Vt.IntArray([int(step) for step in samples.steps]), "stage": stage}}
    scale = settings.meters_per_unit
    held = {key: samples.stacked(key) for key in samples.values}
    for entry in samples.binding.entries:
        values = held[(entry.node, entry.field)]
        rows = values if entry.index is None else values[:, entry.index]
        prim = _prim_spec(layer, entry.prim)
        if entry.kind == "pose":
            translate = _attribute(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            orient = _attribute(prim, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            for code, row in enumerate(np.asarray(rows, dtype=np.float64)):
                layer.SetTimeSample(translate.path, float(code), Gf.Vec3d(*(row[:3] / scale)))
                layer.SetTimeSample(orient.path, float(code), Gf.Quatd(float(row[3]), Gf.Vec3d(*row[4:7])))
        else:
            at = held[entry.at] if entry.at is not None else None
            per_unit = (scales or {}).get(entry.prim, 1.0)
            value = _attribute(prim, VECTOR_VALUE, Sdf.ValueTypeNames.Double3)
            translate = _attribute(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            orient = _attribute(prim, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            scaled = _attribute(prim, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            visibility = _attribute(prim, "visibility", Sdf.ValueTypeNames.Token)
            visibility.default = "inherited"
            for code, row in enumerate(np.asarray(rows, dtype=np.float64)):
                layer.SetTimeSample(value.path, float(code), Gf.Vec3d(*row))
                if at is not None:
                    layer.SetTimeSample(translate.path, float(code),
                                        Gf.Vec3d(*(np.asarray(at[code], dtype=np.float64)[:3] / scale)))
                layer.SetTimeSample(orient.path, float(code), _turn(row))
                length = float(np.linalg.norm(row)) * per_unit
                layer.SetTimeSample(scaled.path, float(code), Gf.Vec3d(length, length, length))
    return layer


def write_recording(folder: Path, binding: StageBinding, samples: StageSamples, settings: RecordingSettings) -> Path:
    """Write ``recording.usdc`` into the run ``folder``, over its ``experiment/scene/scene.usda``, from ``samples`` of
    the ``binding``'s fields; each vector's arrow scaled as its prim in that stage says."""
    del binding  # the samples hold it
    scales: dict[str, float] = {}
    stage_layer = Sdf.Layer.FindOrOpen(str(folder / RUN_STAGE))
    if stage_layer is not None:
        for entry in samples.binding.entries:
            spec = stage_layer.GetAttributeAtPath(Sdf.Path(entry.prim).AppendProperty(VECTOR_SCALE))
            if spec is not None and isinstance(spec.default, float) and math.isfinite(spec.default):
                scales[entry.prim] = float(spec.default)
    layer = recording_layer(samples, settings, scales=scales)
    path = folder / RECORDING_FILE
    path.unlink(missing_ok=True)
    if not layer.Export(str(path)):
        raise OSError(f"usd-core could not write {path}")
    return path


def crate_version(path: Path) -> tuple[int, int, int] | None:
    """The crate version a ``.usdc`` file was written in, from its header (``PXR-USDC`` then three version bytes)."""
    with path.open("rb") as stream:
        head = stream.read(16)
    if head[:8] != b"PXR-USDC":
        return None
    return int(head[8]), int(head[9]), int(head[10])


__all__ = ["RECORDING_FILE", "RECORDING_FORMAT", "RUN_STAGE", "RecordingSettings", "StageSamples", "crate_version",
           "recording_layer", "write_recording"]
