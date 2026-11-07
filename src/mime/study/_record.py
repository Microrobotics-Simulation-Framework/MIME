"""A study's manifest as a plan gives it (``study.json``, format ``mime-study-outputs/1``): the base experiment, the axes
with their units and reasons, every point, the run, the groups, each member's overrides and content key, the points
dropped and the seed policy. Members read ``pending`` until a run finishes them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mime.results._study_folder import (
    BASE_DIR,
    PENDING,
    READING,
    SPEC_FILE,
    STUDY_FORMAT,
    STUDY_VERSION,
    encode_value,
)
from mime.study import _content
from mime.study._axes import STUDY_REASON, Axis, value_text
from mime.study._protocol import Ramp
from mime.units import Quantity, unit_symbol

if TYPE_CHECKING:
    from mime.study._study import StudyPlan


def run_record(plan: StudyPlan) -> dict[str, Any]:
    """The run every member takes: its length, the steps in it, what it records, how often, and the batch."""
    assert plan.duration is not None
    return {"duration": encode_value(plan.duration, "duration"), "steps": plan.steps, "record": list(plan.record),
            "every": None if plan.every is None else encode_value(plan.every, "every"), "batch": plan.batch}


def axis_record(axis: Axis, plan: StudyPlan) -> dict[str, Any]:
    values = [point.values[axis] for point in plan.points]
    quantities = [v for v in values if isinstance(v, Quantity)]
    unit = unit_symbol(quantities[0].dimension) if quantities and len(quantities) == len(values) else None
    return {"path": axis.path, "kind": axis.kind.value, "unit": unit, "reason": plan.reasons.get(axis.path, STUDY_REASON)}


def seed_record(plan: StudyPlan) -> dict[str, Any]:
    return {"policy": "experiment", "seed": plan.seed,
            "note": "every member runs with the experiment's own seed; no entry of this release draws noise, so no "
                    "member differs by a draw"}


def protocol_record(plan: StudyPlan) -> dict[str, Any] | None:
    """A ramp's command and numbers, as the study's identity holds them; ``None`` from rest, which leaves the identity
    of every study run from rest as it was."""
    protocol = plan.protocol
    if not isinstance(protocol, Ramp):
        return None
    owner, command = protocol.path
    return {"ramp": f"{owner}.commands.{command}", "to": encode_value(protocol.to, "to"),
            "rate": encode_value(protocol.rate, "rate"), "hold": encode_value(protocol.hold, "hold")}


def study_id(plan: StudyPlan, input_hash: str | None) -> str:
    """The study's identity: the base's input hash, the axes with their reasons, every point, the run, the drop, the
    seed policy and a ramp's protocol. The same spec on the same base is the same study, whatever has been run of it."""
    held = {"input": input_hash, "axes": [axis_record(axis, plan) for axis in plan.axes],
            "points": [[encode_value(point.values[axis], axis.path) for axis in plan.axes] for point in plan.points],
            "run": run_record(plan), "drop_refused": plan.drop_refused, "seed": seed_record(plan)}
    if (ramp := protocol_record(plan)) is not None:
        held["protocol"] = ramp
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()[:16]


def member_keys(plan: StudyPlan, input_hash: str | None) -> dict[int, str]:
    """The content key of every point that runs, by point index."""
    run = run_record(plan)
    keys: dict[int, str] = {}
    builts = [group.built for group in plan.groups if group.built is not None]
    digest = _content.digest_of(_content.source_hashes(builts))
    for group in plan.groups:
        for index in group.members:
            point = plan.points[index]
            keys[index] = _content.content_key([(axis.path, point.values[axis]) for axis in plan.axes], group.key,
                                               input_hash, run, digest)
    return keys


def document_of(plan: StudyPlan, input_hash: str | None, *, compiles: Sequence[int] = (),
                seconds: Sequence[float | None] = (), status: Mapping[int, str] | None = None,
                group_experiments: Mapping[int, str] | None = None,
                group_files: Mapping[int, str | None] | None = None) -> dict[str, Any]:
    """The manifest of ``plan``: every member ``pending`` unless ``status`` (by point) says otherwise, and each group's
    file the sha256 ``group_files`` gives it (none where it gives none)."""
    keys = member_keys(plan, input_hash)
    group_of = {index: number for number, group in enumerate(plan.groups) for index in group.members}
    effective = {axis.path: plan.reasons.get(axis.path, STUDY_REASON) for axis in plan.axes}
    members: list[dict[str, Any]] = []
    for point in plan.points:
        if point.index not in keys:
            continue
        members.append({
            "member": len(members), "point": point.index, "group": group_of[point.index],
            "status": (status or {}).get(point.index, PENDING), "content_key": keys[point.index],
            "overrides": [{"path": axis.path, "value": encode_value(point.values[axis], axis.path),
                           "reason": effective[axis.path]} for axis in plan.axes]})
    from mime import __version__

    return {
        "format": STUDY_FORMAT, "version": STUDY_VERSION, "id": study_id(plan, input_hash),
        "experiment": {"name": plan.experiment, "input_hash": input_hash, "profile": plan.profile},
        "axes": [axis_record(axis, plan) for axis in plan.axes],
        "points": [{"index": point.index, "values": [encode_value(point.values[axis], axis.path) for axis in plan.axes]}
                   for point in plan.points],
        "run": run_record(plan), "drop_refused": plan.drop_refused,
        "dropped": [{"index": point.index, "coordinates": ", ".join(f"{axis.path} = {value_text(point.values[axis])}"
                                                                    for axis in plan.axes),
                     "refusals": [f"[{r.code}] {r.message}" for r in point.refusals]} for point in plan.refused],
        "seed": seed_record(plan),
        "groups": [{"index": number, "key": group.key, "reasons": list(group.reasons), "members": list(group.members),
                    "batches": list(group.batches), "batched": group.batched, "compiles": group.compiles,
                    "seconds": seconds[number] if number < len(seconds) else None,
                    "experiment": (group_experiments or {}).get(number),
                    "sha256": (group_files or {}).get(number)} for number, group in enumerate(plan.groups)],
        "members": members,
        "lineage": {"base": BASE_DIR, "spec": SPEC_FILE, "mime": __version__},
        "reading": READING}



__all__ = ["axis_record", "document_of", "member_keys", "protocol_record", "run_record", "seed_record", "study_id"]
