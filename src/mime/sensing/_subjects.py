"""What a camera images: each subject's closed form by its kind, its appearance and where that came from, and its parent
medium.

**The route is chosen by capability**, deterministically: a kind with a closed form registered here takes route 1;
the sampled route (route 2) is not built yet, so any other kind is refused with its reason, and ``route=`` is an
override only. Each kind registers its closed form by name (:func:`register_closed_form`): the camera reads that
capability, never a shape's formula.

**Appearance** belongs to materials. A subject's declared appearance wins; where none is declared it comes from the
stage's display primvars (``displayOpacity``, read with inheritance by ``mime.scene.display_primvars``); with neither it
is opaque and mid-grey, and the resolution says so ("appearance defaulted for: ..."), never leaving a body invisible.
Each subject's source is reported. The backlit one-bin camera reads a declared appearance as an attenuation per metre
(Beer-Lambert along its chord) or opaque, and the stage's display opacity as a flat absorber (the share of the
backlight it blocks where it covers a pixel, whatever its chord): display colours are a surface's reflected colour,
which backlit transmission does not see, and are kept for the display view."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.scene import display_primvars
from mime.sensing._forms import CAPSULE, CYLINDER, SPHERE, Form


class Source(Enum):
    """Where a subject's appearance came from."""

    DECLARED = "declared"
    STAGE = "the stage's display primvars"
    DEFAULT = "defaulted"


class Look(Enum):
    OPAQUE = "opaque"
    ATTENUATING = "attenuating"
    FLAT = "flat"


#: The stated default display colour of a subject with no appearance: mid-grey (for the display view; a backlit camera
#: sees an opaque body as its shadow, whatever its colour).
MID_GREY = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Appearance:
    """How a subject changes the backlight where it covers a pixel: ``OPAQUE`` (blocks it), ``ATTENUATING`` (Beer-Lambert
    along its chord, ``attenuation`` per metre, a JAX leaf) or ``FLAT`` (blocks the share ``blocks`` whatever its
    chord: the stage's display opacity read by a backlit camera). ``colour`` is a display colour, kept for the display
    view and not read by the backlit camera."""

    look: Look
    attenuation: Any = 0.0
    blocks: Any = 0.0
    colour: tuple[float, float, float] = MID_GREY

    @staticmethod
    def opaque(colour: tuple[float, float, float] = MID_GREY) -> Appearance:
        return Appearance(Look.OPAQUE, colour=colour)

    @staticmethod
    def attenuating(per_metre: Any, colour: tuple[float, float, float] = MID_GREY) -> Appearance:
        """Beer-Lambert: the transmitted share ``exp(-per_metre L)`` along a chord of length ``L``."""
        return Appearance(Look.ATTENUATING, attenuation=per_metre, colour=colour)

    @staticmethod
    def clear() -> Appearance:
        """Nothing absorbs: an empty lumen."""
        return Appearance(Look.ATTENUATING, attenuation=0.0)


@dataclass(frozen=True)
class Display:
    """A prim's display primvars as the stage holds them (``displayColor``, ``displayOpacity``), each ``None`` where
    neither the prim nor a parent authors it."""

    colour: tuple[float, float, float] | None = None
    opacity: float | None = None


#: The tube's two imaged regions: the wall (the outer cylinder, which its lumen displaces) and the lumen.
TUBE_REGIONS = ("wall", "lumen")


@dataclass(frozen=True)
class Placed:
    """A closed form placed in its subject's frame: the form, its parameters (JAX leaves), the offset of its own origin
    and the rotation turning its own +z onto its axis (fixed at declaration: no frame across every axis is continuous)."""

    form: Form
    parameters: Mapping[str, Any]
    centre: tuple[float, float, float]
    turn: tuple[tuple[float, ...], ...]


def turn_onto(axis: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """The rotation matrix turning +z onto ``axis`` (normalised here) the shorter way, in float64."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, a))
    if c < -1.0 + 1e-15:
        matrix = np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(z, a)
        skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        matrix = np.eye(3) + skew + skew @ skew / (1.0 + c)
    return tuple(tuple(float(x) for x in row) for row in matrix)


def _vector(settings: Mapping[str, Any], name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = settings.get(name, default)
    return (float(value[0]), float(value[1]), float(value[2]))


def _sphere(settings: Mapping[str, Any], region: str) -> Placed:
    return Placed(SPHERE, {"radius": settings["radius"]}, (0.0, 0.0, 0.0), turn_onto((0.0, 0.0, 1.0)))


def _capsule(settings: Mapping[str, Any], region: str) -> Placed:
    return Placed(CAPSULE, {"radius": settings["radius"], "height": settings["height"]}, (0.0, 0.0, 0.0),
                  turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


def _cylinder(settings: Mapping[str, Any], region: str) -> Placed:
    return Placed(CYLINDER, {"radius": settings["radius"], "length": settings["length"]}, (0.0, 0.0, 0.0),
                  turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


def _tube(settings: Mapping[str, Any], region: str) -> Placed:
    radius = settings["inner_radius"] + settings["wall"] if region == "wall" else settings["inner_radius"]
    return Placed(CYLINDER, {"radius": radius, "length": settings["length"]},
                  _vector(settings, "centre", (0.0, 0.0, 0.0)), turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


type ClosedForm = Callable[[Mapping[str, Any], str], Placed]

_closed_forms: dict[str, tuple[ClosedForm, tuple[str, ...]]] = {}

#: Why each of MIME's kinds without a closed form is refused, and what comes instead.
NO_CLOSED_FORM = {
    "ellipsoid": "the ellipsoid has no run-time shape yet, so no chord to take in closed form; it gains one with its "
                 "run-time shape",
    "box": "the box has no run-time shape yet, so no chord to take in closed form; it gains one with its run-time shape",
    "screw": "the screw's surface has no closed-form chord; it is imaged by the sampled route, which needs a cheap sign "
             "function and the measure of its caps' crease, and is not built yet",
    "helix": "the helix has no closed-form chord; it is imaged by the sampled route, which is not built yet",
}


def register_closed_form(kind: str, closed_form: ClosedForm, regions: tuple[str, ...] = ("solid",)) -> None:
    """Give the kind ``kind`` route 1: ``closed_form(settings, region)`` places its closed form for each of its imaged
    ``regions``, from its settings in SI. Registering the same function again changes nothing."""
    held = _closed_forms.get(kind)
    if held is not None and held[0] is not closed_form:
        raise ValueError(f"kind {kind!r} already has a closed form registered")
    _closed_forms[kind] = (closed_form, regions)


register_closed_form("sphere", _sphere)
register_closed_form("capsule", _capsule)
register_closed_form("cylinder", _cylinder)
register_closed_form("tube", _tube, TUBE_REGIONS)

ROUTES = ("closed form",)


def route_of(kind: str, route: str | None = None) -> str:
    """The route a subject of ``kind`` takes: chosen by its capability, ``route`` an override; refused (with its reason)
    where the kind has none, or the override names a route not built."""
    if route is not None and route not in ROUTES:
        raise ArgumentRefused([Refusal(
            "camera.route", RefusalClass.USER, f"the route {route!r} is not built: only route 1 ({ROUTES[0]!r}) is",
            "leave route= out: the route is chosen by the kind's capability",
            why="the sampled route (route 2) comes later")])
    if kind not in _closed_forms:
        reason = NO_CLOSED_FORM.get(kind, f"no closed form is registered for the kind {kind!r}")
        raise ArgumentRefused([Refusal(
            "camera.no-closed-form", RefusalClass.USER, f"a {kind} cannot be imaged yet: {reason}",
            "image a sphere, a capsule, a cylinder or a straight tube (Subject(name, \"sphere\", {\"radius\": ...})), "
            "or register the kind's closed form: register_closed_form(kind, closed_form)")])
    return ROUTES[0]


@dataclass(frozen=True)
class Subject:
    """One imaged region: its ``name``, its geometry ``kind`` (a registered kind's name) and ``settings`` (in SI, in the
    subject's own frame, as the kind's methods take them), its ``region`` of the kind (``"solid"``, a tube's ``"wall"``
    or ``"lumen"``), its declared ``appearance`` (``None``: from the stage, else the default), the ``parent`` medium it
    sits in (a subject's name; its displaced medium), whether it is ``moving`` (its pose given per sub-frame), its fixed
    ``pose`` otherwise (a position and a unit quaternion ``(w, x, y, z)``, the world's identity by default), the
    ``coverage`` its silhouette's pixels take (a sphere's exact conic, ``"disc"``, by default, always in a window; the box
    edge, ``"box"``, every other kind's and a sphere's declared opt-in) and an optional square ``window`` (its half-size
    in pixels) its pixels are rendered in: for exact disc coverage sized from its silhouette at its pose where none is
    declared, its overflow flagged."""

    name: str
    kind: str
    settings: Mapping[str, Any]
    region: str = "solid"
    appearance: Appearance | None = None
    parent: str | None = None
    moving: bool = False
    pose: tuple[tuple[float, float, float], tuple[float, float, float, float]] = ((0.0, 0.0, 0.0),
                                                                                   (1.0, 0.0, 0.0, 0.0))
    coverage: str = ""
    window: int | None = None
    route: str | None = None
    #: The depth of penetration outside its parent within which a subject counts as inside it (a contact's tolerance:
    #: the robot rides the floor), as a share of its enclosing radius.
    tolerance: float = 1e-3
    placed: Placed = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        route_of(self.kind, self.route)
        forms, regions = _closed_forms[self.kind]
        problems: list[str] = []
        if self.region not in regions:
            problems.append(f"a {self.kind} images the regions {', '.join(regions)}, not {self.region!r}")
        if not self.coverage:
            object.__setattr__(self, "coverage", "disc" if self.kind == "sphere" else "box")
        if self.coverage not in ("box", "disc") or (self.coverage == "disc" and self.kind != "sphere"):
            problems.append(f"coverage {self.coverage!r}: \"box\", or \"disc\" for a sphere (its default)")
        if self.window is not None and self.window < 1:
            problems.append("a window is at least one pixel")
        if problems:
            raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER, f"subject {self.name!r}: {problem}",
                                           "write it as its kind declares it: Subject(name, kind, settings, region=, "
                                           "coverage=\"box\" or \"disc\" for a sphere, window= on a moving one)")
                                   for problem in problems])
        object.__setattr__(self, "placed", forms(self.settings, self.region))

    def enclosing(self) -> float:
        """The radius of the smallest ball about the subject's frame's origin that holds it."""
        placed = self.placed
        return float(np.linalg.norm(placed.centre)) + placed.form.enclosing(
            {k: float(v) for k, v in placed.parameters.items()})

    def feature(self) -> float:
        """The subject's smallest silhouette radius: what the small-silhouette flag reads."""
        return float(self.placed.parameters["radius"])


@dataclass(frozen=True)
class Resolved:
    """The subjects with their appearances, each one's source, and the build note for those defaulted."""

    subjects: tuple[Subject, ...]
    sources: Mapping[str, Source]
    notes: tuple[str, ...]


def _from_stage(display: Display) -> Appearance | None:
    if display.opacity is None and display.colour is None:
        return None
    opacity = 1.0 if display.opacity is None else float(display.opacity)
    colour = display.colour or MID_GREY
    return Appearance.opaque(colour) if opacity >= 1.0 else Appearance(Look.FLAT, blocks=opacity, colour=colour)


def resolve(subjects: Sequence[Subject], stage: Mapping[str, Display] | None = None) -> Resolved:
    """Each subject's appearance: its declared one, else the stage's display primvars for it (by subject name), else
    the default (opaque, mid-grey), with the source of each and the note naming those defaulted. Subjects named twice,
    or with a parent no subject is, are refused."""
    names = [s.name for s in subjects]
    problems = [f"{name!r} names two subjects" for name in sorted({n for n in names if names.count(n) > 1})]
    problems += [f"{s.name!r}'s parent {s.parent!r} is no subject" for s in subjects
                 if s.parent is not None and s.parent not in names]
    if problems:
        raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER, problem,
                                       "give each subject one name, and a parent among them: Subject(..., parent=name)")
                                   for problem in problems])
    held: list[Subject] = []
    sources: dict[str, Source] = {}
    defaulted: list[str] = []
    for subject in subjects:
        appearance, source = subject.appearance, Source.DECLARED
        if appearance is None:
            appearance = _from_stage((stage or {}).get(subject.name, Display()))
            source = Source.STAGE
        if appearance is None:
            appearance, source = Appearance.opaque(), Source.DEFAULT
            defaulted.append(subject.name)
        sources[subject.name] = source
        held.append(Subject(subject.name, subject.kind, subject.settings, subject.region, appearance, subject.parent,
                            subject.moving, subject.pose, subject.coverage, subject.window, subject.route,
                            subject.tolerance))
    boxed = [s.name for s in subjects if s.kind == "sphere" and s.coverage == "box"]
    notes = ((f"appearance defaulted for: {', '.join(defaulted)}",) if defaulted else ()) + \
        ((f"the box edge in place of exact disc coverage (opt-in) for: {', '.join(boxed)}",) if boxed else ())
    return Resolved(tuple(held), sources, notes)


def order(subjects: Sequence[Subject]) -> list[Subject]:
    """The subjects with every parent before its children (refusing a cycle)."""
    by_name = {s.name: s for s in subjects}
    placed: list[Subject] = []
    seen: set[str] = set()
    for subject in subjects:
        chain: list[Subject] = []
        current: Subject | None = subject
        while current is not None and current.name not in seen:
            if current in chain:
                raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER,
                                               f"{subject.name!r}'s parents form a cycle",
                                               "give each subject a parent outside its own children: "
                                               "Subject(..., parent=name)")])
            chain.append(current)
            current = by_name.get(current.parent) if current.parent is not None else None
        for held in reversed(chain):
            seen.add(held.name)
            placed.append(held)
    return placed


def display(prim: Any) -> Display:
    """A stage prim's display primvars (its own or the nearest parent's, ``mime.scene.display_primvars``)."""
    colour, opacity = display_primvars(prim)
    return Display(colour, opacity)
