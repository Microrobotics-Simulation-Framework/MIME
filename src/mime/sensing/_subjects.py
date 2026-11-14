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

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.registry import Solid, implements
from mime.scene import display_primvars
from mime.sensing import _sampled as sampled_route
from mime.sensing._forms import CAPSULE, CYLINDER, SPHERE, Form
from mime.sensing._sampled import Sampled


class AppearanceSource(Enum):
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
    """A form placed in its subject's frame: the form, its parameters (JAX leaves), the offset of its own origin and
    the rotation turning its own frame into the subject's (a closed form's own +z onto its axis; fixed at declaration:
    no frame across every axis is continuous). A body on route 2 holds what its march needs (``sampled``) and its
    smallest feature (``feature``)."""

    form: Form
    parameters: Mapping[str, Any]
    centre: tuple[float, float, float]
    turn: tuple[tuple[float, ...], ...]
    sampled: Sampled | None = None
    feature: float | None = None


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
    return Placed(SPHERE, {"radius": settings["radius"]}, _vector(settings, "centre", (0.0, 0.0, 0.0)),
                  turn_onto((0.0, 0.0, 1.0)))


def _capsule(settings: Mapping[str, Any], region: str) -> Placed:
    return Placed(CAPSULE, {"radius": settings["radius"], "height": settings["height"]},
                  _vector(settings, "centre", (0.0, 0.0, 0.0)), turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


def _cylinder(settings: Mapping[str, Any], region: str) -> Placed:
    return Placed(CYLINDER, {"radius": settings["radius"], "length": settings["length"]},
                  _vector(settings, "centre", (0.0, 0.0, 0.0)), turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


def _tube(settings: Mapping[str, Any], region: str) -> Placed:
    radius = settings["inner_radius"] + settings["wall"] if region == "wall" else settings["inner_radius"]
    return Placed(CYLINDER, {"radius": radius, "length": settings["length"]},
                  _vector(settings, "centre", (0.0, 0.0, 0.0)), turn_onto(_vector(settings, "axis", (0.0, 0.0, 1.0))))


type ClosedForm = Callable[[Mapping[str, Any], str], Placed]

_closed_forms: dict[str, tuple[ClosedForm, tuple[str, ...]]] = {}

#: Why each of MIME's kinds without a closed form takes no route yet, and what comes instead: the creased ones until
#: their crease measure (camera PR 5b).
NO_CLOSED_FORM = {
    "box": "the box's body is three slabs' intersection, creased at its edges, and declares no crease measure; it gains "
           "route 1 with its run-time shape",
    "screw": "the screw's body is its thread's region and the span between its caps, creased at the caps' rims, and the "
             "sampled route needs the rims' exact measure (the first-order march alone is 40 % off in motion at a "
             "tilted pose there, at any sample count), which comes with the next piece of the camera",
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

ROUTES = ("closed form", "sampled")


def _sampled_refusal(kind: str) -> Refusal | None:
    """Why ``kind`` cannot take route 2, or ``None``."""
    found = sampled_route.kind_named(kind)
    if found is None or not sampled_route.has_sign(found):
        reason = f"no closed form is registered for the kind {kind!r} and it declares no cheap sign function"
        return Refusal(
            "camera.no-route", RefusalClass.USER, f"a {kind} cannot be imaged: {reason}",
            "image a kind with a closed form (a sphere, a capsule, a cylinder, a straight tube) or one that declares a "
            "cheap sign function (Geometry.sign_function: the ellipsoid, the helix), or register the kind's closed form: "
            "register_closed_form(kind, closed_form)")
    if sampled_route.creased(found) and not implements(found, "crease_measure"):
        reason = NO_CLOSED_FORM.get(kind, f"the {kind}'s body is the intersection of its regions "
                                          f"{', '.join(found.regions)}, creased where they meet")
        return Refusal(
            "camera.crease-unmeasured", RefusalClass.USER,
            f"a {kind} cannot be imaged by the sampled route without its crease measure: {reason}",
            f"give the kind its crease measure (Geometry.crease_measure), or hide it: hide=(...)",
            why="the first-order march reads a crease's edge wrongly at any sample count (camera truth 1)")
    return None


def route_of(kind: str, route: str | None = None) -> str:
    """The route a subject of ``kind`` takes: chosen by its capability (route 1 where a closed form is registered,
    else route 2 where its kind declares a cheap sign function and, if creased, its crease measure), ``route`` an
    override; refused, with its reason, where it has neither, or the override names a route it cannot take."""
    if route is not None and route not in ROUTES:
        raise ArgumentRefused([Refusal(
            "camera.route", RefusalClass.USER, f"the route {route!r} is not a route: they are {', '.join(map(repr, ROUTES))}",
            "leave route= out: the route is chosen by the kind's capability")])
    if route == ROUTES[0] and kind not in _closed_forms:
        raise ArgumentRefused([Refusal(
            "camera.no-closed-form", RefusalClass.USER, f"a {kind} has no closed form registered (route 1)",
            "leave route= out: the route is chosen by the kind's capability, or register_closed_form(kind, ...)")])
    if route == ROUTES[0] or (route is None and kind in _closed_forms):
        return ROUTES[0]
    refused = _sampled_refusal(kind)
    if refused is not None:
        raise ArgumentRefused([refused])
    return ROUTES[1]


def _sampled(kind: str, settings: Mapping[str, Any], region: str) -> Placed:
    """A route-2 body placed in its subject's frame: its kind made from its settings, its region (a kind's only one,
    an uncreased body), its frame's offset (``centre``) and its rotation into the subject's frame (``rotation``, rows),
    and its march's numbers."""
    found = sampled_route.kind_named(kind)
    assert found is not None
    made = sampled_route.made_kind(found, settings)
    names = {item.name for item in dataclasses.fields(found)}  # type: ignore[arg-type]
    parameters = {name: settings[name] for name in names}
    enclosing = float(made.enclosing_radius(settings)) if isinstance(made, Solid) else math.inf
    feature = sampled_route.smallest_length(found, settings)
    rotation = settings.get("rotation", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    marched = Sampled(made, found.regions[0], feature, enclosing, sampled_route.samples_for(enclosing, feature))
    return Placed(sampled_route.form(marched), parameters, _vector(settings, "centre", (0.0, 0.0, 0.0)),
                  tuple(tuple(float(x) for x in row) for row in rotation), marched, feature)


@dataclass(frozen=True)
class Subject:
    """One imaged region: its ``name``, its geometry ``kind`` (a registered kind's name) and ``settings`` (in SI, in the
    subject's own frame, as the kind's methods take them), its ``region`` of the kind (``"solid"``, a tube's ``"wall"``
    or ``"lumen"``), its declared ``appearance`` (``None``: from the stage, else the default), the ``parent`` medium it
    sits in (a subject's name; its displaced medium), whether it is ``moving`` (its pose given per sub-frame), its fixed
    ``pose`` otherwise (a position and a unit quaternion ``(w, x, y, z)``, the world's identity by default), the
    ``coverage`` its silhouette's pixels take (a sphere's exact conic, ``"disc"``, by default, always in a window; the box
    edge, ``"box_edge"``, every other kind's and a sphere's declared opt-in) and an optional square ``window`` (its half-size
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
        if route_of(self.kind, self.route) == ROUTES[1]:
            self._place_sampled()
            return
        forms, regions = _closed_forms[self.kind]
        problems: list[str] = []
        if self.region not in regions:
            problems.append(f"a {self.kind} images the regions {', '.join(regions)}, not {self.region!r}")
        if not self.coverage:
            object.__setattr__(self, "coverage", "disc" if self.kind == "sphere" else "box_edge")
        if self.coverage not in ("box_edge", "disc") or (self.coverage == "disc" and self.kind != "sphere"):
            problems.append(f"coverage {self.coverage!r}: \"box_edge\", or \"disc\" for a sphere (its default)")
        if self.window is not None and self.window < 1:
            problems.append("a window is at least one pixel")
        if problems:
            raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER, f"subject {self.name!r}: {problem}",
                                           "write it as its kind declares it: Subject(name, kind, settings, region=, "
                                           "coverage=\"box_edge\" or \"disc\" for a sphere, window=)")
                                   for problem in problems])
        object.__setattr__(self, "placed", forms(self.settings, self.region))

    def _place_sampled(self) -> None:
        problems: list[str] = []
        if self.region != "solid":
            problems.append(f"a {self.kind} images its body, region \"solid\", not {self.region!r}")
        if not self.coverage:
            object.__setattr__(self, "coverage", "box_edge")
        if self.coverage != "box_edge":
            problems.append(f"coverage {self.coverage!r}: the sampled route takes the box edge, \"box_edge\"")
        if self.window is not None and self.window < 1:
            problems.append("a window is at least one pixel")
        found = sampled_route.kind_named(self.kind)
        missing = [] if found is None else [item.name for item in dataclasses.fields(found)  # type: ignore[arg-type]
                                            if item.name not in self.settings]
        if missing:
            problems.append(f"its settings lack {', '.join(missing)}, which the {self.kind} takes")
        if problems:
            raise ArgumentRefused([Refusal("camera.subject", RefusalClass.USER, f"subject {self.name!r}: {problem}",
                                           "write it as its kind declares it: Subject(name, kind, settings in SI, "
                                           "window=)") for problem in problems])
        object.__setattr__(self, "placed", _sampled(self.kind, self.settings, self.region))

    def enclosing(self) -> float:
        """The radius of the smallest ball about the subject's frame's origin that holds it."""
        placed = self.placed
        if placed.sampled is not None:
            return float(np.linalg.norm(placed.centre)) + placed.sampled.enclosing
        return float(np.linalg.norm(placed.centre)) + placed.form.enclosing(
            {k: float(v) for k, v in placed.parameters.items()})

    def feature(self) -> float:
        """The subject's smallest silhouette radius: what the small-silhouette flag reads."""
        if self.placed.feature is not None:
            return self.placed.feature
        return float(self.placed.parameters["radius"])


@dataclass(frozen=True)
class Resolved:
    """The subjects with their appearances, each one's source, and the build note for those defaulted."""

    subjects: tuple[Subject, ...]
    sources: Mapping[str, AppearanceSource]
    notes: tuple[str, ...]


def from_stage(display: Display) -> Appearance | None:
    if display.opacity is None and display.colour is None:
        return None
    opacity = 1.0 if display.opacity is None else float(display.opacity)
    colour = display.colour or MID_GREY
    return Appearance.opaque(colour) if opacity >= 1.0 else Appearance(Look.FLAT, blocks=opacity, colour=colour)


def resolve(subjects: Sequence[Subject], stage: Mapping[str, Display] | None = None) -> Resolved:
    """Each subject's appearance: its declared one, else the stage's display primvars for it (by subject name), else
    the default (opaque, mid-grey), with the source of each and the note naming those defaulted. Subjects named twice,
    or with a parent no subject is, and a stage's display opacity outside 0 to 1, are refused."""
    names = [s.name for s in subjects]
    problems = [f"{name!r} names two subjects" for name in sorted({n for n in names if names.count(n) > 1})]
    problems += [f"{s.name!r}'s parent {s.parent!r} is no subject" for s in subjects
                 if s.parent is not None and s.parent not in names]
    refused = [Refusal("camera.subject", RefusalClass.USER, problem,
                       "give each subject one name, and a parent among them: Subject(..., parent=name)")
               for problem in problems]
    refused += [Refusal("camera.subject", RefusalClass.USER,
                        f"subject {name!r}: its display opacity on the stage is {held.opacity!r}, outside 0 to 1",
                        "give the stage's display opacity from 0 to 1, or declare the subject's appearance: "
                        "Subject(..., appearance=Appearance.opaque())")
                for name, held in sorted((stage or {}).items())
                if held.opacity is not None and not 0.0 <= float(held.opacity) <= 1.0]
    if refused:
        raise ArgumentRefused(refused)
    held: list[Subject] = []
    sources: dict[str, AppearanceSource] = {}
    defaulted: list[str] = []
    for subject in subjects:
        appearance, source = subject.appearance, AppearanceSource.DECLARED
        if appearance is None:
            appearance = from_stage((stage or {}).get(subject.name, Display()))
            source = AppearanceSource.STAGE
        if appearance is None:
            appearance, source = Appearance.opaque(), AppearanceSource.DEFAULT
            defaulted.append(subject.name)
        sources[subject.name] = source
        held.append(Subject(subject.name, subject.kind, subject.settings, subject.region, appearance, subject.parent,
                            subject.moving, subject.pose, subject.coverage, subject.window, subject.route,
                            subject.tolerance))
    boxed = [s.name for s in subjects if s.kind == "sphere" and s.coverage == "box_edge"]
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
