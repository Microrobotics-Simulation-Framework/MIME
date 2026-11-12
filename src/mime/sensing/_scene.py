"""The whole scene a camera sees, from the experiment's roster: every body's parts and every structure become the
camera's subjects, nothing listed by the user. Bodies are moving (their pose is read from the graph every step);
structures are static. ``hide`` names what the camera leaves out, and ``absorbers`` the attenuation per metre of what is
not opaque; every other subject takes the stated default (opaque, with the build note ``appearance defaulted for:``).

A subject without a closed form (route 1) is refused with its reason, as the offline renderer refuses it
(:func:`mime.sensing.route_of`), until the sampled route lands; nothing is dropped silently."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mime.build._refusals import ArgumentRefused
from mime.registry import Roster
from mime.sensing._subjects import Appearance, Look, Subject, route_of

#: The first line of the camera's refusal of a subject without a closed form, and the code it carries.
NO_FORM = "camera.no-closed-form"


@dataclass(frozen=True)
class Found:
    """The subjects of a roster, the bodies among them (each with the subjects of its parts), what was hidden, and what
    could not be imaged (name, code, reason)."""

    subjects: tuple[Subject, ...]
    bodies: Mapping[str, tuple[str, ...]]
    hidden: tuple[str, ...]
    refused: tuple[tuple[str, str, str], ...]
    unknown_hidden: tuple[str, ...]


def _list(value: Any) -> Any:
    """``value`` as plain Python numbers and lists, for a JSON string."""
    if isinstance(value, Mapping):
        return {str(k): _list(v) for k, v in cast(Mapping[Any, Any], value).items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_list(v) for v in cast(Sequence[Any], value)]
    if isinstance(value, (bool, str)) or value is None:
        return value
    return float(value)


def _in_body(part_settings: Mapping[str, Any], rotation: Sequence[Sequence[float]],
             centre: Sequence[float]) -> dict[str, Any]:
    """A part's settings with its axis turned into the body's frame and its centroid there."""
    settings = {k: _list(v) for k, v in part_settings.items()}
    if "axis" in settings:
        axis = np.asarray(settings["axis"], dtype=np.float64)
        settings["axis"] = (np.asarray(rotation, dtype=np.float64) @ axis).tolist()
    settings["centre"] = [float(c) for c in centre]
    return settings


def _appearance(name: str, absorbers: Mapping[str, float]) -> Appearance | None:
    return Appearance.attenuating(absorbers[name]) if name in absorbers else None


def subjects_of(roster: Roster, hide: Sequence[str], absorbers: Mapping[str, float]) -> Found:
    """Every body part and structure of ``roster`` as a subject, less ``hide``."""
    hidden: list[str] = []
    refused: list[tuple[str, str, str]] = []
    subjects: list[Subject] = []
    bodies: dict[str, tuple[str, ...]] = {}

    def refuse(name: str, kind: str, error: ArgumentRefused) -> None:
        refused.append((name, error.refusals[0].code, error.refusals[0].message))

    for name, body in sorted(roster.bodies.items()):
        if name in hide:
            hidden.append(name)
            continue
        made: list[str] = []
        for part in body.parts:
            subject_name = name if len(body.parts) == 1 else f"{name}.{part.name}"
            try:
                route_of(part.kind)
            except ArgumentRefused as error:
                refuse(subject_name, part.kind, error)
                continue
            subjects.append(Subject(subject_name, part.kind, _in_body(part.settings, part.rotation, part.centre),
                                    appearance=_appearance(subject_name, absorbers), moving=True))
            made.append(subject_name)
        bodies[name] = tuple(made)
    for name, structure in sorted(roster.structures.items()):
        if name in hide:
            hidden.append(name)
            continue
        try:
            route_of(structure.kind)
        except ArgumentRefused as error:
            refuse(name, structure.kind, error)
            continue
        settings = {k: _list(v) for k, v in structure.settings.items()}
        region = "wall" if structure.kind == "tube" else "solid"
        subjects.append(Subject(name, structure.kind, settings, region, appearance=_appearance(name, absorbers)))
    named = set(roster.bodies) | set(roster.structures)
    return Found(tuple(subjects), bodies, tuple(hidden), tuple(refused),
                 tuple(n for n in hide if n not in named))


def subject_spec(subject: Subject, window: int | None) -> dict[str, Any]:
    """A subject as plain data for the camera node's JSON constant (the node rebuilds it)."""
    appearance = subject.appearance
    attenuation = float(appearance.attenuation) if appearance is not None and appearance.look is Look.ATTENUATING \
        else None
    return {"name": subject.name, "kind": subject.kind, "settings": _list(subject.settings), "region": subject.region,
            "moving": subject.moving, "pose": _list(subject.pose), "coverage": subject.coverage, "window": window,
            "attenuation": attenuation}


def subject_of(spec: Mapping[str, Any]) -> Subject:
    """The subject :func:`subject_spec` wrote."""
    attenuation = spec["attenuation"]
    pose = spec["pose"]
    return Subject(spec["name"], spec["kind"], spec["settings"], spec["region"],
                   Appearance.attenuating(attenuation) if attenuation is not None else None, None, spec["moving"],
                   ((pose[0][0], pose[0][1], pose[0][2]), (pose[1][0], pose[1][1], pose[1][2], pose[1][3])),
                   spec["coverage"], spec["window"])


def dumps(spec: Mapping[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, allow_nan=False)


def numbers(spec: object) -> tuple[float, ...]:
    """Every float of a camera's spec, in the order of its keys (sorted) and lists: the numbers the node consumes when it
    is built (the subjects' settings and appearances, the response), which it holds as a leaf its step does not read."""
    if isinstance(spec, Mapping):
        held = cast(Mapping[str, object], spec)
        return tuple(x for key in sorted(held) for x in numbers(held[key]))
    if isinstance(spec, (list, tuple)):
        return tuple(x for each in cast(Sequence[object], spec) for x in numbers(each))
    return (float(spec),) if isinstance(spec, float) else ()


def frame_window(reach: float, width: int, height: int) -> int:
    """A window's half-size in pixels from a silhouette's reach, widened as the offline renderer widens it; half the
    frame where the reach is not a finite number (a sphere behind the camera at the start)."""
    if not math.isfinite(reach) or reach <= 0.0:
        return max(1, min(width, height) // 2)
    return math.ceil(1.1 * reach) + 2
