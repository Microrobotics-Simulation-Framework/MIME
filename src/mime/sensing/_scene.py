"""The whole scene a camera sees, from the experiment's roster: every body's parts and every structure become the
camera's subjects, nothing listed by the user. Bodies are moving (their pose is read from the graph every step);
structures are static. ``hide`` names what the camera leaves out, and ``absorbers`` the attenuation per metre of what is
not opaque; every other subject takes the stated default (opaque, with the build note ``appearance defaulted for:``).

A subject takes route 1 where its kind has a closed form and route 2 (sampled) where it declares a cheap sign
function; one with neither, or a creased one without its crease measure, is refused with its reason, as the offline
renderer refuses it (:func:`mime.sensing.route_of`); nothing is dropped silently. The exceptions are static: a scenery
mesh the stage holds and a structure of a kind the stage draws with no closed form (an ellipsoid, a box, a helix, a
screw) are traced as meshes (:class:`TracedMesh`, ``mime.sensing._mesh``), once a run, in the static structure, before
route 2: a drawn structure stays the mesh the tracer measured."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from mime.build._refusals import ArgumentRefused
from mime.registry import Roster, RosterDrawing
from mime.scene import SceneryMesh
from mime.sensing._mesh import closed_problem, inside, signed_volume, triangles_of
from mime.sensing._subjects import ROUTES, Appearance, Display, Look, Subject, from_stage, route_of

#: The code of the refusal of an unowned gprim that is not a mesh (a cube, a capsule, curves): the camera traces the
#: stage's meshes, and reads no other gprim.
MESH_UNTRACED = "camera.mesh-untraced"


@dataclass(frozen=True, eq=False)
class TracedMesh:
    """A static mesh the camera traces: its ``name`` (a scenery prim's path, or a structure's name), its ``source``
    (``"scenery"``, or the kind a structure's drawing is of), its triangles' corners in the world (``(T, 3, 3)``,
    metres, counter-clockwise seen from outside), its declared ``appearance`` (``absorbers``; ``None`` where none is),
    the stage's ``display`` primvars for it, whether the stage animates it, how far a decimation moved it (metres;
    ``None``: not recorded), a drawing's facets a turn and sagitta (metres; zero for an exact drawing or a scenery
    mesh), and why it is not a closed, consistently wound solid (``None``: it is; checked only where it attenuates, the
    one look that needs it)."""

    name: str
    source: str
    triangles: np.ndarray
    appearance: Appearance | None = None
    display: Display | None = None
    animated: bool = False
    decimation: float | None = None
    segments: int = 0
    sagitta: float = 0.0
    open: str | None = None

    def look(self) -> Appearance | None:
        """Its appearance: declared, else from the stage's display primvars, else ``None`` (the stated default)."""
        if self.appearance is not None:
            return self.appearance
        return from_stage(self.display) if self.display is not None else None


def _rotated(q: Sequence[float], v: np.ndarray) -> np.ndarray:
    """``v`` (``(..., 3)``) turned by the unit quaternion ``q`` ``(w, x, y, z)``."""
    w, x, y, z = (float(c) for c in q)
    matrix = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return np.asarray(v, dtype=np.float64) @ matrix.T


def _faces(counts: Sequence[int], indices: Sequence[int]) -> np.ndarray:
    """Each polygon fanned from its first corner into triangles, ``(m, 3)`` indices."""
    found: list[tuple[int, int, int]] = []
    start = 0
    for count in counts:
        face = indices[start:start + count]
        start += count
        found += [(face[0], face[k], face[k + 1]) for k in range(1, count - 1)]
    return np.asarray(found, dtype=np.int64).reshape(-1, 3)


def drawn_mesh(name: str, kind: str, drawn: RosterDrawing, appearance: Appearance | None) -> TracedMesh:
    """A structure's drawing as a traced mesh in the world, with its sagitta, about ``r (1 - cos(sqrt(2) pi / n))`` for
    ``n`` facets a turn and ``r`` the drawing's largest distance from its own axis (a facet's corners span about
    ``2 pi / n`` each way, so its diagonal's; an ellipsoid of equal semi-axes at 48 facets measured 0.17 um on 40 um
    against this 0.17 um, ``tests/sensing/test_mesh_swap.py``); zero for an exact drawing."""
    points = np.asarray(drawn.drawing.points, dtype=np.float64).reshape(-1, 3)
    faces = _faces(drawn.drawing.face_vertex_counts, drawn.drawing.face_vertex_indices)
    world = _rotated(drawn.orientation, points) + np.asarray(drawn.translation, dtype=np.float64)
    radius = float(np.max(np.hypot(points[:, 0], points[:, 1]))) if len(points) else 0.0
    sagitta = 0.0 if drawn.exact else radius * (1.0 - math.cos(math.sqrt(2.0) * math.pi / drawn.segments))
    held = TracedMesh(name, kind, triangles_of(world, faces), appearance, segments=drawn.segments, sagitta=sagitta)
    return _checked(held, world, faces)


def scenery_mesh(held: SceneryMesh, appearance: Appearance | None) -> TracedMesh:
    """A scenery mesh as a traced mesh: its triangles in the world, its display primvars, whether it is animated and
    how far a decimation moved it."""
    traced = TracedMesh(held.path, "scenery", triangles_of(held.points, held.triangles), appearance,
                        Display(held.colour, held.opacity), held.animated, held.decimation)
    return _checked(traced, held.points, held.triangles)


def _checked(held: TracedMesh, points: np.ndarray, faces: np.ndarray) -> TracedMesh:
    """``held`` with why it is not a closed solid, checked where it attenuates."""
    look = held.look()
    if look is None or look.look is not Look.ATTENUATING:
        return held
    return dataclasses.replace(held, open=closed_problem(points, faces))


@dataclass(frozen=True)
class Found:
    """The subjects of a roster, the bodies among them (each with the subjects of its parts), what was hidden, and what
    could not be imaged (name, code, reason)."""

    subjects: tuple[Subject, ...]
    bodies: Mapping[str, tuple[str, ...]]
    hidden: tuple[str, ...]
    refused: tuple[tuple[str, str, str], ...]
    unknown_hidden: tuple[str, ...]
    #: The stage's display primvars of the subjects that take any (by subject name), as the roster holds them.
    display: Mapping[str, Display] = field(default_factory=dict[str, Display])
    #: What reducing a non-constant display primvar did, by the line that says so, to the subjects it was done for.
    reduced: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    #: The static meshes the camera traces: the stage's scenery meshes and the structures of drawn kinds.
    meshes: tuple[TracedMesh, ...] = ()
    #: The closed attenuating mesh each body's subject lies in at the start, by subject name: its parent medium.
    inside: Mapping[str, str] = field(default_factory=dict[str, str])


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


def _placed_in_body(part_settings: Mapping[str, Any], rotation: Sequence[Sequence[float]],
                    centre: Sequence[float]) -> dict[str, Any]:
    """A sampled part's settings, in its own frame as its kind takes them, with where its frame sits in the body's: its
    centroid (``centre``) and the rotation from its frame to the body's (``rotation``, rows)."""
    settings = {k: _list(v) for k, v in part_settings.items()}
    settings["centre"] = [float(c) for c in centre]
    settings["rotation"] = [[float(x) for x in row] for row in rotation]
    return settings


def _appearance(name: str, absorbers: Mapping[str, float]) -> Appearance | None:
    return Appearance.attenuating(absorbers[name]) if name in absorbers else None


def subjects_of(roster: Roster, hide: Sequence[str], absorbers: Mapping[str, float]) -> Found:
    """Every body part and structure of ``roster`` as a subject, less ``hide``."""
    hidden: list[str] = []
    refused: list[tuple[str, str, str]] = []
    subjects: list[Subject] = []
    bodies: dict[str, tuple[str, ...]] = {}
    display: dict[str, Display] = {}
    reduced: dict[str, list[str]] = {}

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
                route = route_of(part.kind)
            except ArgumentRefused as error:
                refuse(subject_name, part.kind, error)
                continue
            settings = _in_body(part.settings, part.rotation, part.centre) if route == ROUTES[0] else \
                _placed_in_body(part.settings, part.rotation, part.centre)
            subjects.append(Subject(subject_name, part.kind, settings, appearance=_appearance(subject_name, absorbers),
                                    moving=True))
            made.append(subject_name)
            if part.display is not None and part.display != (None, None) and subject_name not in absorbers:
                display[subject_name] = Display(*part.display)
                for line in part.display_reduced:
                    reduced.setdefault(line, []).append(subject_name)
        bodies[name] = tuple(made)
    meshes: list[TracedMesh] = []
    for name, structure in sorted(roster.structures.items()):
        if name in hide:
            hidden.append(name)
            continue
        try:
            route = route_of(structure.kind)
        except ArgumentRefused as error:
            if structure.drawn is None:
                refuse(name, structure.kind, error)
                continue
            route = None
        if route != ROUTES[0] and structure.drawn is not None:
            meshes.append(drawn_mesh(name, structure.kind, structure.drawn, _appearance(name, absorbers)))
            continue
        settings = {k: _list(v) for k, v in structure.settings.items()}
        region = "wall" if structure.kind == "tube" else "solid"
        subjects.append(Subject(name, structure.kind, settings, region, appearance=_appearance(name, absorbers)))
    for held in roster.scenery:
        if held.path in hide:
            hidden.append(held.path)
            continue
        if isinstance(held, SceneryMesh):
            meshes.append(scenery_mesh(held, _appearance(held.path, absorbers)))
            for line in held.reduced:
                reduced.setdefault(line, []).append(held.path)
            continue
        refused.append((held.path, MESH_UNTRACED,
                        f"the stage holds this {held.type_name}, and the camera traces the stage's meshes alone (a "
                        f"{held.type_name} is not read)" + "".join(f"; {line}" for line in held.reduced)))
    inside = _inside(roster, subjects, meshes)
    subjects = [dataclasses.replace(s, around=_medium(meshes, inside[s.name])) if s.name in inside else s
                for s in subjects]
    named = set(roster.bodies) | set(roster.structures) | {held.path for held in roster.scenery}
    return Found(tuple(subjects), bodies, tuple(hidden), tuple(refused),
                 tuple(n for n in hide if n not in named), display,
                 {line: tuple(names) for line, names in reduced.items()}, tuple(meshes), inside)


def _medium(meshes: Sequence[TracedMesh], name: str) -> float:
    look = next(each for each in meshes if each.name == name).look()
    assert look is not None
    return float(look.attenuation)


def _inside(roster: Roster, subjects: Sequence[Subject], meshes: Sequence[TracedMesh]) -> dict[str, str]:
    """The closed attenuating mesh each moving subject's centre lies in at its body's pose at the start (the smallest,
    where several hold it), by subject name: what it displaces along its chord."""
    solids = [each for each in meshes if each.open is None and (look := each.look()) is not None
              and look.look is Look.ATTENUATING]
    if not solids:
        return {}
    owner = {f"{body}.{part.name}" if len(held.parts) > 1 else body: (held, part)
             for body, held in roster.bodies.items() for part in held.parts}
    found: dict[str, str] = {}
    for subject in subjects:
        if not subject.moving or subject.name not in owner:
            continue
        held, part = owner[subject.name]
        if held.pose is None:
            continue
        centre = np.asarray(held.pose[:3]) + _rotated(held.pose[3:], np.asarray(part.centre, dtype=np.float64))
        holding = [each for each in solids if inside(each.triangles, centre)]
        if holding:
            found[subject.name] = min(holding, key=lambda each: signed_volume(each.triangles)).name
    return found


def subject_spec(subject: Subject, window: int | None, display: Display | None = None) -> dict[str, Any]:
    """A subject as plain data for the camera node's JSON constant (the node rebuilds it). ``display`` is the stage's
    display primvars for a subject with no declared appearance: written only where there is one, so a scene whose
    stage holds none writes the same constant as before."""
    appearance = subject.appearance
    attenuation = float(appearance.attenuation) if appearance is not None and appearance.look is Look.ATTENUATING \
        else None
    spec: dict[str, Any] = {
        "name": subject.name, "kind": subject.kind, "settings": _list(subject.settings), "region": subject.region,
        "moving": subject.moving, "pose": _list(subject.pose), "coverage": subject.coverage, "window": window,
        "attenuation": attenuation}
    if display is not None and attenuation is None and (display.colour is not None or display.opacity is not None):
        spec["display"] = {"colour": _list(display.colour), "opacity": display.opacity}
    if subject.around is not None:
        spec["around"] = float(subject.around)
    return spec


def mesh_spec(held: TracedMesh) -> dict[str, Any]:
    """A traced mesh's look as plain data for the camera node's JSON constant (its triangles are the node's arrays)."""
    look = held.look() or Appearance.opaque()
    return {"name": held.name, "look": look.look.value,
            "attenuation": float(look.attenuation) if look.look is Look.ATTENUATING else None,
            "blocks": float(look.blocks) if look.look is Look.FLAT else None}


def mesh_look(spec: Mapping[str, Any]) -> Appearance:
    """The appearance :func:`mesh_spec` wrote."""
    if spec["look"] == Look.ATTENUATING.value:
        return Appearance.attenuating(spec["attenuation"])
    if spec["look"] == Look.FLAT.value:
        return Appearance(Look.FLAT, blocks=spec["blocks"])
    return Appearance.opaque()


def subject_of(spec: Mapping[str, Any]) -> Subject:
    """The subject :func:`subject_spec` wrote."""
    attenuation = spec["attenuation"]
    pose = spec["pose"]
    held = spec.get("display")
    appearance = Appearance.attenuating(attenuation) if attenuation is not None else (
        from_stage(Display(held["colour"], held["opacity"])) if held is not None else None)
    return Subject(spec["name"], spec["kind"], spec["settings"], spec["region"], appearance, None, spec["moving"],
                   ((pose[0][0], pose[0][1], pose[0][2]), (pose[1][0], pose[1][1], pose[1][2], pose[1][3])),
                   spec["coverage"], spec["window"], around=spec.get("around"))


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
