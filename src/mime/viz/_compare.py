"""The ghosted comparison of a study's members: ``compare.usda`` in the study folder, one layer that sublayers the
reference member's stage and holds every member as its own prim, so that a viewer lists, labels and switches members on
and off, and mutes the whole comparison by muting one sublayer.

::

    /World/<thing>                    the reference member's things, animated (the stage's own, over its stage)
    /Compare/<member>                 an Xform for each member (``m<point>``, as its arrays are named in the study folder):
                                      a ``visibility`` attribute, and ``customData`` ``mime_compare`` naming the member, its
                                      coordinates (``axis<k>``: path, value and unit) and whether it is the reference
    /Compare/<member>/<thing>         a member's things: an internal reference to ``/World/<thing>`` (the reference's
                                      geometry), with time samples written from the member's recorded poses

The reference's prim is listed with the others and holds no things (its things are ``/World``): it is always shown. The
layer's own ``customLayerData`` ``mime:compare`` lists the members, the reference and the time base.

**The time base.** Every sample is a multiple of ``stage_every``, a whole number of every member's base step (the study
refuses another, ``study.stage-every``, with the nearest valid one); the time code is the sample's index and
``timeCodesPerSecond`` is samples per simulated second, as a run's own recording is. **What it references** lies inside
the study folder: the reference group's stage (``experiment/scene/scene.usda`` or the group's own build). A member of a
group whose build differs is drawn with the reference's geometry, and the layer says so. The writer of a stage still emits
only ``/World``; this is the overlay on a root prim of its own (``mime.scene``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mime.build._binding import Bound
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.results._study_folder import BASE_DIR, COMPARE_FILE, GROUPS_DIR
from mime.scene import STAGE_FILE, VECTOR_SCALE, up_axis
from mime.scene._pxr import Sdf, Vt
from mime.viz._recording import write_entries

if TYPE_CHECKING:
    from mime.results._stage import StageRecord
    from mime.results._study import StudyResult

COMPARE_FORMAT = "mime.compare/1"
COMPARE_ROOT = "Compare"
GHOST_OPACITY = 0.35


def member_name(point: int) -> str:
    return f"m{point}"


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> ArgumentRefused:
    return ArgumentRefused([Refusal(code, category, message, action)])


def _bound(record: StageRecord) -> list[Bound]:
    found: list[Bound] = []
    for e in record.entries():
        at = e.get("at")
        found.append(Bound(str(e["prim"]), str(e["node"]), str(e["field"]), e["index"], str(e["kind"]), str(e["thing"]),
                           str(e["name"]), str(e["unit"]), (str(at["node"]), str(at["field"])) if at else None))
    return found


def _things(entries: Sequence[Bound]) -> list[str]:
    """The top-level prims below ``/World`` the entries move: what a member's prim references."""
    return list(dict.fromkeys(entry.prim.split("/")[2] for entry in entries))


def _coordinates(study: StudyResult, index: int) -> dict[str, Any]:
    from mime.units import Quantity, unit_symbol

    found: dict[str, Any] = {}
    for number, path in enumerate(study.axes):
        value = study.coordinate(path, index)
        if isinstance(value, Quantity):
            found[f"axis{number}"] = {
                "path": path, "unit": unit_symbol(value.dimension),
                "value": Vt.DoubleArray([float(v) for v in np.atleast_1d(np.asarray(value.value, dtype=np.float64))])
                if np.ndim(value.value) else float(cast(float, value.value))}
        else:
            found[f"axis{number}"] = {"path": path, "value": str(value), "unit": ""}
    return found


def _stage_of(study: StudyResult, group: int) -> str:
    own = f"{GROUPS_DIR}/{group}/{BASE_DIR}/{STAGE_FILE}"
    return own if group and study.folder is not None and (study.folder / own).is_file() else f"{BASE_DIR}/{STAGE_FILE}"


def _ghost_path(name: str) -> Callable[[str], str]:
    """Where one member's things sit: ``/World/<thing>/...`` of the stage is ``/Compare/<member>/<thing>/...``."""
    def path_of(prim: str) -> str:
        return f"/{COMPARE_ROOT}/{name}/" + prim.split("/", 2)[2]

    return path_of


def compare_layer(study: StudyResult, reference: int = 0, *, stage: str | None = None) -> Any:
    """The overlay as an anonymous layer, its sublayer the reference's stage (a path relative to where it is written)."""
    count = len(study)
    if not -count <= reference < count:
        raise _refuse("compare.reference", RefusalClass.USER, f"the reference is a member number 0 to {count - 1}, not {reference!r}",
                       "name a member of the study: result.write_compare(reference=0)")
    reference %= count
    records: list[StageRecord] = []
    for index in range(count):
        record = study.member(index).stage
        if record is None:
            raise _refuse("compare.stage", RefusalClass.USER, f"member {index} kept no stage samples, so there is nothing to ghost",
                           "keep them in the sweep: Study(exp).sweep(..., stage_every=ms(10)), and run it again")
        records.append(record)
    first = records[reference]
    times = [np.asarray(r.steps, dtype=np.float64) * r.dt for r in records]
    if any(len(t) != len(times[reference]) or not np.allclose(t, times[reference], rtol=0, atol=1e-12) for t in times):
        raise _refuse("compare.stage", RefusalClass.USER, "the members' stage samples are not at the same simulated times (another duration, "
                       "a run stopped early, or another stage_every)", "compare members run for the same duration: `Study(exp).sweep(..., duration=s(0.05))`")
    sample = float(times[reference][1] - times[reference][0]) if len(times[reference]) > 1 else 0.0
    if not sample > 0.0:
        raise _refuse("compare.stage", RefusalClass.USER, "the members kept fewer than two stage samples", "run longer than stage_every=")
    entries = _bound(first)
    if any([e.prim for e in _bound(r)] != [e.prim for e in entries] for r in records):
        raise _refuse("compare.scene", RefusalClass.USER, "the members bind different things to their stages",
                       "compare members of the same scene: `result.write_compare(reference=0)`")
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.subLayerPaths.append(stage or f"{BASE_DIR}/{STAGE_FILE}")
    layer.startTimeCode, layer.endTimeCode = 0.0, float(len(times[reference]) - 1)
    layer.timeCodesPerSecond = layer.framesPerSecond = 1.0 / sample
    root = layer.pseudoRoot
    root.SetInfo("upAxis", up_axis(cast(tuple[float, float, float], tuple(first.meta["up"]))))
    root.SetInfo("metersPerUnit", float(first.meta["meters_per_unit"]))
    root.SetInfo("defaultPrim", "World")
    names = [member_name(study.points[i]) for i in range(count)]
    layer.customLayerData = {"mime:compare": {
        "format": COMPARE_FORMAT, "experiment": study.experiment, "reference": names[reference], "members": Vt.StringArray(names),
        "stage_every_s": sample, "samples": len(times[reference]),
        "geometry": "every member is drawn with the reference's geometry"}}
    scale = float(first.meta["meters_per_unit"])
    reference_layer = Sdf.Layer.FindOrOpen(str(study.folder / (stage or f"{BASE_DIR}/{STAGE_FILE}"))) \
        if study.folder is not None else None
    scales: dict[str, float] = {}
    if reference_layer is not None:
        for entry in entries:
            spec = reference_layer.GetAttributeAtPath(Sdf.Path(entry.prim).AppendProperty(VECTOR_SCALE))
            if spec is not None and isinstance(spec.default, float) and math.isfinite(spec.default):
                scales[entry.prim] = float(spec.default)
    write_entries(layer, entries, {key: np.asarray(v, dtype=np.float64) for key, v in first.fields.items()}, scale, scales)
    compare = Sdf.PrimSpec(root, COMPARE_ROOT, Sdf.SpecifierDef, "Xform")
    for index in range(count):
        member = Sdf.PrimSpec(compare, names[index], Sdf.SpecifierDef, "Xform")
        flag = index == reference
        member.customData = {"mime_compare": {
            "member": index, "point": study.points[index], "name": names[index], "reference": flag,
            "coordinates": _coordinates(study, index)}}
        visibility = Sdf.AttributeSpec(member, "visibility", Sdf.ValueTypeNames.Token)
        visibility.default = "inherited"
        if flag:
            continue
        opacity = Sdf.AttributeSpec(member, "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray)
        opacity.default = Vt.FloatArray([GHOST_OPACITY])
        for thing in _things(entries):
            ghost = Sdf.PrimSpec(member, thing, Sdf.SpecifierDef)
            ghost.referenceList.Prepend(Sdf.Reference("", Sdf.Path(f"/World/{thing}")))
        write_entries(layer, entries, {k: np.asarray(v, dtype=np.float64) for k, v in records[index].fields.items()},
                      scale, scales, path_of=_ghost_path(names[index]))
    return layer


def write_compare(study: StudyResult, reference: int = 0) -> Path:
    """Write ``compare.usda`` into the study's folder (see the module). The folder is the study's; a result in memory has
    none (``compare.folder``)."""
    if study.folder is None:
        raise _refuse("compare.folder", RefusalClass.USER, "this study is in memory, and the overlay's references lie inside a study folder",
                       "run it into a folder, study.run_to(folder), and write the overlay from that result")
    group = study.groups[reference % len(study)]
    layer = compare_layer(study, reference, stage=_stage_of(study, group))
    path = study.folder / COMPARE_FILE
    path.unlink(missing_ok=True)
    if not layer.Export(str(path)):
        raise OSError(f"usd-core could not write {path}")
    return path


__all__ = ["COMPARE_FORMAT", "COMPARE_ROOT", "GHOST_OPACITY", "compare_layer", "member_name", "write_compare"]
