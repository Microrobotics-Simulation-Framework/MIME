"""The registry walk: every registered entry, MIME's toys and the lab fixture's, and every kind of thing has a
round-trip case, and every case round-trips: built, saved, loaded in a fresh process and rebuilt, it gives the same
derived hash and the same first steps, bit for bit. An entry registered without a case fails here. Every geometry kind
that samples its surface passes the kind-conformance harness (``mime.testing.kind_problems``) at each declaration a
case makes of it, and a kind no case declares fails; the de Jongh helix passes it at every count of starts it accepts,
at the least and the most modulation and twist (either hand) it accepts and at FL-9's. Every kind that draws itself
passes the drawing check (``mime.testing.drawing_problems``) at the writer's resolution, at each declaration, and the
helix at the same settings, and the check holds every drawn kind's vertices to its surface and its volume and area to
their closed forms (a drawn kind it cannot hold to all of them fails the walk); a kind with a native form is drawn by
USD, whose definition the stage tests check.

The controls: removing a case that is the only one to use an entry is caught, naming that entry; so is an entry
that no case uses. Each case's parameter leaves also reach its compiled step without a recompilation.
"""

import dataclasses
import importlib.util
import itertools
import math
import os

import numpy as np
import pytest
from cases import CASES, Case

from mime.core import jnp
from mime.docs import load_namespaces
from mime.registry import Drawing, EntryKind, Samples, implements
from mime.registry import registered as registered_entries
from mime.scene import DRAWING_SEGMENTS, DeJonghHelix, SolidBox, SolidEllipsoid, Tube
from mime.scene._helix import STARTS
from mime.testing import (
    KIND_ROUNDING_ULPS,
    covered,
    drawing_checks,
    drawing_kinds,
    drawing_problems,
    installed_entries,
    kind_problems,
    param_reach_failures,
    refused_round_trip_failures,
    round_trip_failures,
    sampling_kinds,
)
from mime.testing._kind_conformance import _SURFACES
from mime.things import ThingKind
from mime.units import dimensionless, m

LAB_FIXTURE = "mime_lab_fixture"


def _needs_lab_fixture(case: Case) -> None:
    if case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None:
        if os.environ.get("CI"):
            pytest.fail("the lab fixture package is not installed; every CI lane installs it from ./lab_fixture")
        pytest.skip("the lab fixture package is not installed: uv pip install --no-deps ./lab_fixture")


def uncovered(cases: tuple[Case, ...], registered: list[tuple[str, str, str]]) -> list[str]:
    entries: set[tuple[str, str, str]] = set()
    kinds: set[str] = set()
    for case in cases:
        if case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None:
            continue
        found_entries, found_kinds = covered(case.make().document())
        entries |= found_entries
        kinds |= found_kinds
    missing = [f"{kind} {name} {version}" for kind, name, version in registered if (kind, name, version) not in entries]
    missing += [f"a thing of kind {kind.value}" for kind in ThingKind if kind.value not in kinds]
    return missing


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("interaction", "toy.spring", "0.1.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_registered_entry_and_every_kind_of_thing_has_a_round_trip_case(registered):
    missing = uncovered(CASES, registered)
    assert not missing, "no round-trip case uses: " + "; ".join(missing) + ". Add one to tests/saved/cases.py."


def test_removing_the_only_case_for_an_entry_fails_the_walk(registered):
    """The control that the walk can fail: each case that alone uses some entry or kind of thing is removed in turn,
    and the walk names what it leaves without a case."""
    load_bearing = 0
    for case in CASES:
        rest = tuple(c for c in CASES if c is not case)
        missing = uncovered(rest, registered)
        if missing:
            load_bearing += 1
            print(f"without {case.name}: no round-trip case uses {'; '.join(missing)}")
    assert load_bearing >= 3


def test_an_entry_registered_without_a_case_fails_the_walk(registered):
    assert uncovered(CASES, [*registered, ("interaction", "toy.uncased", "0.1.0")]) == [
        "interaction toy.uncased 0.1.0"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_case_round_trips_through_a_fresh_process(case, tmp_path):
    _needs_lab_fixture(case)
    if case.refused:
        assert refused_round_trip_failures(case.make(), tmp_path, case.refused) == []
    else:
        assert round_trip_failures(case.make(), tmp_path) == []


@pytest.mark.parametrize("case", [c for c in CASES if not c.refused], ids=lambda case: case.name)
def test_every_cases_parameter_leaves_reach_its_compiled_step(case):
    _needs_lab_fixture(case)
    assert param_reach_failures(case.make().build()) == []


# -- the kind-conformance harness ----------------------------------------------------------------------------------

def kinds_in_cases() -> dict[type, list[tuple[object, dict]]]:
    """Every geometry kind the round-trip cases declare, a structure's or a body's part's, with its settings in SI as
    the plan resolved them."""
    found: dict[type, list[tuple[object, dict]]] = {}
    for case in CASES:
        if case.refused or (case.name == "nudged_toy" and importlib.util.find_spec(LAB_FIXTURE) is None):
            continue
        plans = case.make().plan()
        held = [(g.geometry, dict(g.settings)) for g in plans.geometries.values()]
        held += [(line.solid, dict(line.settings)) for line in plans.materials.parts]
        for kind, settings in held:
            found.setdefault(type(kind), []).append((kind, settings))
    return found


def spacing_of(kind, settings) -> float:
    """A fortieth of the kind's reach from its axis: about 250 samples round it, where the helix's lateral area, a
    periodic integral the trapezoid rule converges on exponentially, is exact to rounding (at a twelfth it is 8e-12
    off)."""
    return kind.radial_reach(settings) / 40.0 if implements(kind, "radial_reach") else 1e-4


@pytest.fixture(scope="module")
def declared_kinds():
    return kinds_in_cases()


def test_every_kind_that_samples_its_surface_passes_the_kind_conformance_harness(declared_kinds):
    load_namespaces()
    sampling = sampling_kinds([record.cls for record in registered_entries(EntryKind.GEOMETRY)])
    assert sampling, "the walk would pass over no kind"
    unchecked = [cls.label() for cls in sampling if cls not in declared_kinds]
    assert unchecked == [], f"no round-trip case declares {unchecked}, so the harness checks nothing of them"
    for cls in sampling:
        for kind, settings in declared_kinds[cls]:
            assert kind_problems(kind, settings, spacing_of(kind, settings)) == []
    print(f"\nchecked: {', '.join(sorted(cls.label() for cls in sampling))}")


@pytest.mark.parametrize("planted", ["volume", "inertia", "area", "normals", "points"])
def test_control_a_kind_whose_closed_form_or_samples_are_planted_wrong_fails_the_harness(declared_kinds, planted):
    """Each check can fail: the helix's volume, inertia or area off by one part in 10^9, its samples' normals turned or
    its points moved out by that much, each fails, on every run (nothing here is drawn)."""
    (kind, settings), *_ = declared_kinds[DeJonghHelix]
    wrong = 1.0 + 1e-9

    class Planted(DeJonghHelix):
        def volume(self, s):
            return super().volume(s) * (wrong if planted == "volume" else 1.0)

        def inertia_per_mass(self, s):
            held = super().inertia_per_mass(s)
            return tuple(tuple(c * (wrong if planted == "inertia" and i == j == 0 else 1.0) for j, c in enumerate(row))
                         for i, row in enumerate(held))

        def area(self, s, patch):
            return super().area(s, patch) * (wrong if planted == "area" else 1.0)

        def samples(self, parameters, patch, counts):
            held = super().samples(parameters, patch, counts)
            if planted == "normals":
                turned = held.normals.at[:, 0].add(1e-9)
                return Samples(held.points, turned / jnp.linalg.norm(turned, axis=1, keepdims=True), held.weights)
            if planted == "points":
                return Samples(held.points * wrong, held.normals, held.weights)
            return held

    planted_kind = Planted(**{name: getattr(kind, name) for name in ("radius", "length", "wavenumber", "modulation",
                                                                      "starts")})
    problems = kind_problems(planted_kind, settings, spacing_of(kind, settings))
    print(f"\n{planted}: {problems}")
    assert problems != []


# -- the helix at every setting it accepts --------------------------------------------------------------------------

#: The helix's settings beside FL-9's case: every count of starts the kind accepts, at the least and the most modulation
#: it accepts and FL-9's, and at no twist, FL-9's, the most of the left hand and, of the right, where its normals were
#: found furthest from the screw's (3.75): (starts, modulation, nu).
HELIX_SETTINGS = list(itertools.product(STARTS, (0.02, 0.33, 0.5), (-4.0, 0.0, 2.33, 3.75)))
#: FL-9's radius and length.
FL9 = {"radius": 1.56e-3, "length": 7.47e-3}


def helix(starts, modulation, nu, cls=DeJonghHelix):
    settings = {**FL9, "wavenumber": nu, "modulation": modulation}
    kind = cls(radius=m(FL9["radius"]), length=m(FL9["length"]), wavenumber=dimensionless(nu),
               modulation=dimensionless(modulation), starts=starts)
    return kind, settings


def chosen_counts(kind, settings) -> dict[str, tuple[int, ...]]:
    """The samples' counts for this setting: a tenth of the reach apart along the axis and out across the caps, where
    the Gauss-Legendre rules of the divergence theorem's volume and inertia converge to rounding at every setting the
    kind accepts (a polynomial and trigonometric integrand in the axial position and in the fraction out); and round
    the lateral patch's turn, where the surface element is a periodic integral whose trapezoid sum converges at a rate
    the setting sets (four starts at modulation 0.5 and nu 4 need about 900 points, FL-9 about 100), the spacing's
    count doubled until the weights, at one point along the axis (along which the element does not change), sum to
    the area within the harness's rounding, thirty-two times the spacing's at most."""
    spacing = kind.radial_reach(settings) / 10.0
    around, along = kind.sample_counts(settings, spacing, "lateral")
    area = kind.area(settings, "lateral")
    for _ in range(6):
        found = float(np.sum(np.asarray(kind.samples(settings, "lateral", (around, 1)).weights)))
        if abs(found - area) <= KIND_ROUNDING_ULPS * np.finfo(np.float64).eps * math.sqrt(around * along) * area:
            break
        around *= 2
    return {"lateral": (around, along), "caps": kind.sample_counts(settings, spacing, "caps")}


@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS)
def test_the_helix_passes_the_kind_conformance_harness_at_every_setting_it_accepts(starts, modulation, nu):
    kind, settings = helix(starts, modulation, nu)
    assert kind.setting_problems(settings) == []
    counts = chosen_counts(kind, settings)
    print(f"\ncounts {counts}")
    assert kind_problems(kind, settings, kind.radial_reach(settings) / 10.0, counts) == []


class DroppedTwist(DeJonghHelix):
    """The planted defect: the cap at +L/2 sampled without its twist phase, ``k L``, so that its points are turned
    back by it about the axis (the cap at -L/2 has none)."""

    def samples(self, parameters, patch, counts):
        held = super().samples(parameters, patch, counts)
        if patch != "caps":
            return held
        half = held.points.shape[0] // 2  # the cap at -L/2 first
        phase = -parameters["wavenumber"] / parameters["radius"] * parameters["length"]
        c, s = math.cos(phase), math.sin(phase)
        plus = held.points[half:]
        turned = jnp.stack([plus[:, 0], c * plus[:, 1] - s * plus[:, 2], s * plus[:, 1] + c * plus[:, 2]], axis=-1)
        return Samples(jnp.concatenate([held.points[:half], turned]), held.normals, held.weights)


@pytest.mark.parametrize("starts", STARTS)
def test_control_the_cap_sampled_without_its_twist_fails_at_every_count_of_starts(starts):
    """At three and four starts the section's second moments are isotropic, so the inertia cannot show the turned
    cap: the check that a cap lies within the screw's cross-section does, on every run (nothing is drawn)."""
    kind, settings = helix(starts, 0.33, 2.33, DroppedTwist)
    problems = kind_problems(kind, settings, kind.radial_reach(settings) / 10.0, chosen_counts(kind, settings))
    print(f"\n{problems}")
    assert any("patch 'caps'" in problem and "outside its region 'screw'" in problem for problem in problems)



# -- the drawing check -----------------------------------------------------------------------------------------------

def test_every_kind_that_draws_itself_passes_the_drawing_check_at_the_writers_resolution(declared_kinds):
    load_namespaces()
    drawing = drawing_kinds([record.cls for record in registered_entries(EntryKind.GEOMETRY)])
    assert {Tube, DeJonghHelix, SolidBox, SolidEllipsoid} <= set(drawing), "the walk would pass over a kind"
    unchecked = [cls.label() for cls in drawing if cls not in declared_kinds]
    assert unchecked == [], f"no round-trip case declares {unchecked}, so the drawing check checks nothing of them"
    for cls in drawing:
        declared = {(repr(kind), repr(sorted(settings.items()))): (kind, settings) for kind, settings in
                    declared_kinds[cls]}  # the cases repeat a declaration: the check is the same
        for kind, settings in declared.values():
            assert drawing_problems(kind, settings, DRAWING_SEGMENTS) == []
    print("\n" + "\n".join(f"{cls.label()}: {', '.join(drawing_checks(declared_kinds[cls][0][0]))}"
                           for cls in sorted(drawing, key=lambda c: c.name)))


def test_every_kind_that_draws_itself_is_held_to_its_surface_its_volume_and_its_area(declared_kinds):
    """The drawing check says what it checks of each kind, and a kind drawn with nothing to hold its vertices, volume
    and area to is a kind the check passes without seeing: it fails here, until the harness has its surface."""
    load_namespaces()
    drawing = drawing_kinds([record.cls for record in registered_entries(EntryKind.GEOMETRY)])
    short = {cls.label(): set(FULL_CHECK) - set(drawing_checks(declared_kinds[cls][0][0])) for cls in drawing}
    assert {label: left for label, left in short.items() if left} == {}


@pytest.mark.parametrize(("starts", "modulation", "nu"), HELIX_SETTINGS)
def test_the_helix_passes_the_drawing_check_at_every_setting_it_accepts(starts, modulation, nu):
    kind, settings = helix(starts, modulation, nu)
    assert drawing_problems(kind, settings, DRAWING_SEGMENTS) == []


FULL_CHECK = ("closed", "on the surface", "volume", "area")

#: Settings of the kinds with no two equal edges, semi-axes or lengths in them, as the declared ones have (a cube, a
#: sphere): a planted permutation of the axes is a different drawing only where they differ.
EDGES = (0.8e-3, 1.3e-3, 2.1e-3)
SOLID_DRAWN = {SolidBox: (SolidBox(size=m(EDGES)), {"size": EDGES}),
               SolidEllipsoid: (SolidEllipsoid(semi_axes=m(EDGES)), {"semi_axes": EDGES})}


def drawn_kind(declared_kinds, cls):
    """A kind of ``cls`` and its settings in SI, with distinct edges where it has them."""
    return SOLID_DRAWN[cls] if cls in SOLID_DRAWN else declared_kinds[cls][0]


@pytest.mark.parametrize("cls", [Tube, DeJonghHelix, SolidBox, SolidEllipsoid])
def test_the_kinds_at_the_settings_the_controls_plant_into_pass_the_drawing_check(declared_kinds, cls):
    """The unplanted arm of the controls below: a plant is caught by what it plants, not by the settings."""
    kind, settings = drawn_kind(declared_kinds, cls)
    assert drawing_problems(kind, settings, DRAWING_SEGMENTS) == []


def moved(drawing: Drawing, points) -> Drawing:
    return Drawing(tuple((float(p[0]), float(p[1]), float(p[2])) for p in points), drawing.face_vertex_counts,
                   drawing.face_vertex_indices, drawing.representation)


def planted(drawing: Drawing, defect: str, scale: float) -> Drawing:
    points = np.asarray(drawing.points)
    counts, indices = drawing.face_vertex_counts, list(drawing.face_vertex_indices)
    if defect == "scaled":
        return moved(drawing, points * (1.0 + scale))
    if defect == "grown":
        return moved(drawing, points * 1.5)
    if defect == "permuted":
        return moved(drawing, points[:, [1, 2, 0]])
    if defect == "stretched":
        return moved(drawing, points * np.array([1.0, 1.0, 1.2]))
    if defect == "thick_wall":
        radius = np.hypot(points[:, 0], points[:, 1])
        return moved(drawing, np.where((radius > 0.5 * (radius.min() + radius.max()))[:, None],
                                       points * np.array([2.0, 2.0, 1.0]), points))
    if defect == "turned_face":
        indices[:counts[0]] = indices[:counts[0]][::-1]
    elif defect == "missing_face":
        counts, indices = counts[1:], indices[counts[0]:]
    return Drawing(drawing.points, tuple(counts), tuple(indices), drawing.representation)


class Untwisted(DeJonghHelix):
    """The planted defect: drawn without its twist, every section at the first's angle."""

    def drawing(self, settings, segments):
        return super().drawing({**settings, "wavenumber": 0.0}, segments)


def _as(cls, kind):
    """``kind`` re-made as an instance of the planted subclass ``cls``, with its own settings."""
    return cls(**{f.name: getattr(kind, f.name) for f in dataclasses.fields(kind)})


def planted_problems(declared_kinds, cls, defect, scale=1e-4, segments=DRAWING_SEGMENTS):
    kind, settings = drawn_kind(declared_kinds, cls)
    held = _as(Untwisted, kind) if defect == "untwisted" else kind
    drawn_by = type(held).drawing

    class Planted(type(held)):
        def drawing(self, s, at):
            drawn = drawn_by(self, s, at)
            return drawn if defect == "untwisted" else planted(drawn, defect, scale)

    return drawing_problems(_as(Planted, held), settings, segments)


#: What each kind is planted with: every defect of the picture, on the four kinds that draw themselves (the tube's
#: own, a wall too thick or too long, besides). ``scaled`` is a part in 10^4: the ellipsoid's was beneath the drawing's
#: own error at the volume before, and a tube's was never planted.
PLANTS = ([(cls, defect) for cls in (Tube, DeJonghHelix, SolidBox, SolidEllipsoid)
           for defect in ("scaled", "permuted", "turned_face", "missing_face")]
          + [(Tube, "grown"), (Tube, "stretched"), (Tube, "thick_wall"), (DeJonghHelix, "untwisted")])


@pytest.mark.parametrize(("cls", "defect"), PLANTS, ids=lambda held: held if isinstance(held, str) else held.__name__)
def test_control_a_drawing_planted_wrong_fails_the_drawing_check_for_every_kind(declared_kinds, cls, defect):
    """Each planted defect fails the check, on every run (nothing is drawn, so nothing varies): every kind's drawing
    scaled by a part in 10^4 or with its axes permuted, the tube's grown by half, a face turned over, a face missing,
    the tube's wall thick or its length 20% too much, and the helix drawn untwisted."""
    problems = planted_problems(declared_kinds, cls, defect)
    print(f"\n{cls.label()} {defect}: {problems}")
    assert problems


def test_control_a_scale_beneath_the_vertices_check_is_still_caught_by_the_volume_and_area(declared_kinds):
    """The vertices lie on the surface to rounding, and the volume and area converge on the closed forms to within the
    fourth order's term: a drawing scaled by a part in 10^4 fails each, and each is named."""
    for cls in (Tube, DeJonghHelix, SolidBox, SolidEllipsoid):
        problems = planted_problems(declared_kinds, cls, "scaled")
        text = "\n".join(problems)
        print(f"\n{cls.label()}:\n{text}")
        assert all(part in text for part in ("off its surface", "drawing's volume", "drawing's area"))


@pytest.mark.parametrize("closed", ["volume", "area"])
@pytest.mark.parametrize("cls", [Tube, SolidBox, SolidEllipsoid])
def test_control_a_surface_whose_closed_form_is_off_fails_the_drawing_check(monkeypatch, declared_kinds, cls, closed):
    """The volume and the area are each held to a closed form on their own: the harness's own closed form off by a
    part in 10^3 fails the unplanted drawing, naming it: above the fourth-order term the comparison resolves, which
    is 10^-6 to 10^-4 at the writer's resolution, so a part in 10^5 passes for some kinds, and is not planted."""
    kind, settings = drawn_kind(declared_kinds, cls)
    surface = _SURFACES[kind.name]

    def wrong(held):
        right = surface(held)
        return dataclasses.replace(right, **{closed: getattr(right, closed) * 1.001})

    monkeypatch.setitem(_SURFACES, kind.name, wrong)
    problems = drawing_problems(kind, settings, DRAWING_SEGMENTS)
    print(f"\n{kind.label()} {closed}: {problems}")
    assert len(problems) == 1 and f"drawing's {closed}" in problems[0]


@pytest.mark.parametrize("closed", ["volume", "area"])
def test_control_the_helix_whose_closed_form_is_off_fails_the_drawing_check(declared_kinds, closed):
    """The same for the helix, whose closed forms are its own methods: off by a part in 10^3."""
    kind, settings = declared_kinds[DeJonghHelix][0]

    class Wrong(DeJonghHelix):
        def volume(self, s):
            return super().volume(s) * (1.001 if closed == "volume" else 1.0)

        def area(self, s, patch):
            return super().area(s, patch) * (1.001 if closed == "area" else 1.0)

    problems = drawing_problems(_as(Wrong, kind), settings, DRAWING_SEGMENTS)
    print(f"\n{closed}: {problems}")
    assert len(problems) == 1 and f"drawing's {closed}" in problems[0]
