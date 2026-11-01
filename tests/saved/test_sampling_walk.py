"""The sampling conformance suite (``mime.testing``'s ``sampling_problems``, ``conditioning_problems``,
``reach_problems`` and ``distance_problems``) on every registered kind, MIME's and every installed namespace's, at
each declaration a round-trip case makes of it: its samples spread evenly (nearest pairs, coverage by patch, seam and
the screw's crests and troughs) at a few hundred samples and at a couple of thousand, the MRS matrix on them
conditioned at today's blob and at the larger fractions a smoother blob needs, its hull query holding its body and no
more; and every wall kind's distance and normal.

The controls, each failing on every run (nothing here is drawn): every current kind's own quadrature, the grid its
sampling chooses from, fails the spacing check (it is what the MRS rung took before 7m-a); each kept version fails it
too, and only those; a sampling that leaves out the screw's caps fails the coverage check while its spacing passes;
the screw's quadrature fails the crests' and caps' coverage; the sphere's quadrature fails the conditioning bound; a
reach planted short, long, or without a capsule's caps fails; a wall distance planted off, or with its normal
reversed, fails.

**What was measured** (float64, on the declared settings, at ``SUITE_SAMPLES`` and ``FINER_SAMPLES``; recorded so that
a threshold's margin is known): the quasi-uniform samplings' 5th percentile of nearest neighbours over the mean spacing
0.76 to 0.81, their least 0.61 to 0.73, every region's coverage 0.96 to 1.06 (the seams' over two spacings); the
quadratures' 0.09 to 0.36 and 0.06 to 0.36, the screw's caps 2.4 to 2.9 times their share. The conditioning, at
``SUITE_SAMPLES``, in ``CONDITION_BOUNDS``'s docstring."""

import dataclasses
import importlib.util

import jax.numpy as jnp
import numpy as np
import pytest
from cases import CASES

from mime.docs import load_namespaces
from mime.registry import EntryKind, implements, lookup
from mime.registry import registered as registered_entries
from mime.scene import KEPT, Box, Capsule, Screw, Sphere, quadrature
from mime.testing import (
    CONDITION_BOUNDS,
    FINER_SAMPLES,
    SUITE_SAMPLES,
    condition_number,
    conditioning_problems,
    distance_problems,
    installed_entries,
    reach_problems,
    sampling_problems,
    sampling_quality,
    suite_spacing,
)
from mime.units import m

LAB_FIXTURE = "mime_lab_fixture"


def declared_kinds() -> dict[type, list[tuple[object, dict]]]:
    """Every geometry kind the round-trip cases declare, a structure's or a body's part's, with its settings in SI as
    the plan resolved them, each declaration once."""
    found: dict[type, dict[str, tuple[object, dict]]] = {}
    for case in CASES:
        if case.refused or (case.lab and importlib.util.find_spec(LAB_FIXTURE) is None):
            continue
        plans = case.make().plan()
        held = [(g.geometry, dict(g.settings)) for g in plans.geometries.values()]
        held += [(line.solid, dict(line.settings)) for line in plans.materials.parts]
        for kind, settings in held:
            found.setdefault(type(kind), {})[repr((kind, sorted(settings.items())))] = (kind, settings)
    return {cls: list(held.values()) for cls, held in found.items()}


@pytest.fixture(scope="module")
def declared():
    return declared_kinds()


@pytest.fixture(scope="module")
def sampling_classes():
    """Every installed geometry kind that samples its surface: one a test registers on its own is no installed entry,
    as the registry walk reads them."""
    load_namespaces()
    installed = {(kind, name, version) for kind, name, version in installed_entries()}
    found = [record.cls for record in registered_entries(EntryKind.GEOMETRY)
             if (record.kind.value, record.name, record.version) in installed and implements(record.cls, "sampling")]
    assert {Sphere, Capsule, Screw, Box} <= set(found), "the walk would pass over a kind"
    return found


def kept(cls) -> bool:
    return (cls.name, cls.version) in KEPT


def screw_bands(kind, settings):
    """The screw's crests and troughs, its lateral samples whose distance from the axis is beyond 0.6 of its
    modulation above and below its radius (the helix score's bands)."""
    radius, modulation = settings["radius"], settings["modulation"]
    lateral = list(kind.patches).index("lateral")

    def labeller(points, labels):
        reach = (np.hypot(points[:, 1], points[:, 2]) / radius - 1.0) / modulation
        return {"crests": (labels == lateral) & (reach > 0.6), "troughs": (labels == lateral) & (reach < -0.6)}

    return labeller


def labeller_of(kind, settings):
    return screw_bands(kind, settings) if isinstance(kind, Screw) else None


#: The kinds the suite runs on, by name: every installed kind that samples its surface, less the kept versions (held to
#: their failure below). A kind registered without a line here fails the first test.
WALKED = ("sphere", "capsule", "screw", "cylinder", "box", "ellipsoid", "labx.bead")


def test_the_suite_walks_every_installed_kind_that_samples_and_each_has_a_declaration(declared, sampling_classes):
    current = {cls.name for cls in sampling_classes if not kept(cls)}
    assert current == set(WALKED), f"walk {sorted(current - set(WALKED))}; drop {sorted(set(WALKED) - current)}"
    unchecked = [cls.label() for cls in sampling_classes if cls not in declared]
    assert unchecked == [], f"no round-trip case declares {unchecked}, so the suite checks nothing of them"


def declarations(declared, sampling_classes, name):
    (cls,) = [cls for cls in sampling_classes if cls.name == name and not kept(cls)]
    return declared[cls]


@pytest.mark.parametrize("samples", [SUITE_SAMPLES, FINER_SAMPLES])
@pytest.mark.parametrize("name", WALKED)
def test_every_kind_that_samples_spreads_its_samples_evenly(declared, sampling_classes, name, samples):
    found: dict[str, list[str]] = {}
    for kind, settings in declarations(declared, sampling_classes, name):
        spacing = suite_spacing(kind, settings, samples)
        problems = sampling_problems(kind, settings, spacing, labeller=labeller_of(kind, settings))
        if problems:
            found[repr(sorted(settings.items()))] = problems
    assert found == {}


def test_control_every_kinds_own_quadrature_fails_the_spacing_check(declared, sampling_classes):
    """The grid each kind's sampling chooses from, which the MRS rung took before 7m-a, crowds: the check fails on
    each, at the suite's spacing."""
    passed = []
    for cls in sampling_classes:
        kind, settings = declared[cls][0]
        spacing = suite_spacing(kind, settings)
        grid = quadrature(kind, settings, spacing, tuple(kind.patches))
        problems = sampling_problems(kind, settings, spacing, sampling=grid)
        if not any("5th percentile" in p for p in problems):
            passed.append(kind.label())
    assert passed == []


def test_control_each_kept_version_fails_the_spacing_check_and_no_current_one_does(declared, sampling_classes):
    """A kept version's sampling is its quadrature: the suite fails on it, on a real kind, as a run of the MRS rung
    before 7m-a took it; the current version of the same kind passes at the same settings."""
    load_namespaces()
    assert {cls for cls in sampling_classes if kept(cls)} == {held for held, _ in KEPT.values()}
    for (name, version), (held, current) in KEPT.items():
        successor = lookup(EntryKind.GEOMETRY, name, current)
        assert successor is not None
        kind, settings = declared[held][0]
        now = successor.cls(**{field.name: getattr(kind, field.name) for field in dataclasses.fields(kind)})
        spacing = suite_spacing(kind, settings)
        assert any("5th percentile" in p for p in sampling_problems(kind, settings, spacing)), kind.label()
        assert sampling_problems(now, settings, spacing, labeller=labeller_of(now, settings)) == [], now.label()


def test_control_a_sampling_that_leaves_out_a_patch_fails_the_coverage_check_and_passes_the_spacing(declared):
    """Coverage catches what spacing cannot: the screw sampled quasi-uniformly on its thread alone is evenly spread,
    and its caps hold none of their share."""
    kind, settings = declared[Screw][0]
    spacing = suite_spacing(kind, settings, FINER_SAMPLES)
    planted = kind.sampling(settings, spacing, ("lateral",))
    problems = sampling_problems(kind, settings, spacing, sampling=planted)
    assert any("patch caps hold 0.000 times" in p for p in problems), problems
    assert not any("5th percentile" in p or "closest pair" in p or "weights sum" in p for p in problems), problems


def test_the_screws_crests_troughs_and_caps_hold_their_share_and_its_quadratures_do_not(declared):
    """The screw's regions the helix score measured: its quadrature gives its caps 2.4 to 2.9 times their share and its
    crests 0.7, and the check fails it there; its sampling holds each within the band (the walk above)."""
    kind, settings = declared[Screw][0]
    spacing = suite_spacing(kind, settings, FINER_SAMPLES)
    grid = quadrature(kind, settings, spacing, tuple(kind.patches))
    quality = sampling_quality(kind, settings, spacing, sampling=grid, labeller=screw_bands(kind, settings))
    assert quality.coverage["patch caps"] > 2.0 and quality.coverage["crests"] < 0.85
    problems = sampling_problems(kind, settings, spacing, sampling=grid, labeller=screw_bands(kind, settings))
    assert any("patch caps hold" in p for p in problems) and any("crests hold" in p for p in problems)


@pytest.mark.parametrize("name", WALKED)
def test_every_kind_that_samples_keeps_the_mrs_matrix_conditioned(declared, sampling_classes, name):
    found = {}
    for kind, settings in declarations(declared, sampling_classes, name):
        problems = conditioning_problems(kind, settings, suite_spacing(kind, settings))
        if problems:
            found[repr(sorted(settings.items()))] = problems
    assert found == {}


def test_control_the_spheres_quadrature_fails_the_conditioning_bound_at_todays_blob(declared):
    kind, settings = declared[Sphere][0]
    spacing = suite_spacing(kind, settings)
    grid = quadrature(kind, settings, spacing, tuple(kind.patches))
    assert condition_number(kind, settings, spacing, 0.5, sampling=grid) > CONDITION_BOUNDS[0.5]
    assert condition_number(kind, settings, spacing, 0.5) <= CONDITION_BOUNDS[0.5]


def test_every_geometry_kind_answers_the_hull_query():
    load_namespaces()
    installed = {(kind, name, version) for kind, name, version in installed_entries()}
    missing = [record.cls.label() for record in registered_entries(EntryKind.GEOMETRY)
               if (record.kind.value, record.name, record.version) in installed and not implements(record.cls, "reach")]
    assert missing == [], f"every geometry kind answers the hull query; {missing} do not"


@pytest.mark.parametrize("name", WALKED)
def test_every_kind_reaches_as_far_as_its_body_and_no_further(declared, sampling_classes, name):
    found = {}
    for kind, settings in declarations(declared, sampling_classes, name):
        problems = reach_problems(kind, settings)
        if problems:
            found[repr(sorted(settings.items()))] = problems[:3]
    assert found == {}


@pytest.mark.parametrize("plant", ["short", "long", "no_caps"])
def test_control_a_reach_planted_wrong_fails(plant):
    capsule = Capsule(radius=m(1.0e-3), height=m(2.0e-3), axis=(0.0, 0.6, 0.8))
    settings = {"radius": 1.0e-3, "height": 2.0e-3, "axis": (0.0, 0.6, 0.8)}

    def planted(parameters, direction):
        if plant == "short":
            return capsule.reach(parameters, direction) * (1.0 - 1e-3)
        if plant == "long":
            return capsule.reach(parameters, direction) * (1.0 + 1e-3)
        return parameters["radius"] * jnp.sqrt(jnp.sum(jnp.asarray(direction) ** 2))

    assert reach_problems(capsule, settings) == []
    assert reach_problems(capsule, settings, reach=planted) != []


def tube_closed_form(centre, axis, radius):
    centre, axis = np.asarray(centre), np.asarray(axis) / np.linalg.norm(axis)

    def closed(points):
        offset = points - centre
        across = offset - np.outer(offset @ axis, axis)
        length = np.linalg.norm(across, axis=1)
        return radius - length, -across / length[:, None]

    return closed


TUBE = {"centre": (1.0e-3, -2.0e-3, 0.5e-3), "axis": (0.0, 0.6, 0.8), "radius": 3.175e-3}


def tube_points():
    """Points of the lumen, off its axis: a fixed spiral from near the axis to near the wall."""
    axis = np.asarray(TUBE["axis"])
    first = np.cross(axis, (1.0, 0.0, 0.0))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    t = np.linspace(0.05, 0.98, 40)
    return (np.asarray(TUBE["centre"]) + np.outer(t * TUBE["radius"] * np.cos(7.0 * t), first)
            + np.outer(t * TUBE["radius"] * np.sin(7.0 * t), second) + np.outer((t - 0.5) * 1e-2, axis))


def test_every_wall_kind_gives_its_distance_and_free_space_none():
    """Every installed wall kind: one a test registers on its own (the tube oracle's series) is no installed entry."""
    load_namespaces()
    installed = {(kind, name, version) for kind, name, version in installed_entries()}
    walls = {record.name: record.cls for record in registered_entries(EntryKind.WALL_KIND)
             if (record.kind.value, record.name, record.version) in installed}
    assert {"free_space", "tube"} <= set(walls)
    closed = {"tube": tube_closed_form(TUBE["centre"], TUBE["axis"], TUBE["radius"])}
    unchecked = [name for name, cls in walls.items() if cls.matches is not None and name not in closed]
    assert unchecked == [], f"no closed form here holds the distance of {unchecked}"
    for name, cls in walls.items():
        parameters = TUBE if cls.matches == "cylinder" else {}
        assert distance_problems(cls(), parameters, tube_points(), closed.get(name)) == [], name


@pytest.mark.parametrize("plant", ["off", "reversed"])
def test_control_a_wall_distance_planted_wrong_fails(plant):
    """A tube's distance planted a micrometre off, or its normal reversed."""
    load_namespaces()
    tube = lookup(EntryKind.WALL_KIND, "tube", "0.1.0")
    assert tube is not None
    held = tube.cls()

    class Planted:
        matches = held.matches

        def label(self):
            return f"planted {plant}"

        def distance(self, parameters, point):
            found = held.distance(parameters, point)
            if plant == "off":
                return type(found)(found.distance + 1e-6, found.normal)
            return type(found)(found.distance, -found.normal)

    closed = tube_closed_form(TUBE["centre"], TUBE["axis"], TUBE["radius"])
    assert distance_problems(held, TUBE, tube_points(), closed) == []
    assert distance_problems(Planted(), TUBE, tube_points(), closed) != []
