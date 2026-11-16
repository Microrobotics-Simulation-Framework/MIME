"""Crease measures on the sampled route (camera PR 5b): a creased body's caps as exact edge functions, composed with
the march's branches and their regions, each corner cell the exact share of that composition.

- **The composition's share** (``coverage.composed_covered``): a union of intersections of half-planes in a cell, by
  inclusion and exclusion of each subset's convex polygon's boundary integral, against the same polygons clipped
  exactly (Sutherland-Hodgman) on drawn lines, coinciding lines and lines along a cell's side included. **Control**
  (``CAMERA_PLANT=composed-sign``): one subset's sign flipped fails, every run. A known gap of camera PR 5a's
  ``union_covered`` (a line repeated along a cell's side) is recorded, not fixed.
- **G17, the crease measure's conformance** (``mime.testing.crease_measure_problems``): the cylinder's caps on drawn
  rays have the body's sign at their planes' crossings and vanish on its drawing's rims. **Control**
  (``CAMERA_PLANT=wide-cap``): the measure's caps one per cent wide are caught on every run.
- **G18, the cylinder on the sampled route** (``route="sampled"``; its default stays the closed form): every pixel
  against the exact image (the hull of the rims' images, ``hull_oracle``) within ``PIXEL_LINE``, and the silhouette's
  motion (the secant over one pixel of the body's motion along x and z of the darkness's equivalent radius and
  centroid) within ``MOTION_LINE`` of the exact image's, rods turned 0 to 30 degrees towards the line of sight, rolled
  and moved by a fraction of a pixel, in both dtypes; caps narrower than two sub-pixels raise ``thin_cap`` and the
  pixels about them are judged by the flag. Both lines are ``scripts/camera_route2/crease_line.py``'s, fixed before any
  of these draws (``predictions_camera_pr5b.json``): the scan's largest error at its 360 poses, by an exact numpy
  re-implementation of the scheme, times 1.25, a margin declared, not derived, for the poses between the scan's points,
  which the drawn views test. Against route 1 the swap holds each route to its line against the same exact image.
  **Controls**, on the pinned views (turns 0, 2 and 4, edge-on to nearly so, where the side meets a cap at about a
  right angle): ``CAMERA_PLANT=single-edge-crease`` (each pixel one straight edge, the composed line's, no sub-samples:
  cycle 4's corner error) and ``CAMERA_PLANT=no-crease-measure`` (the body's first-order distance, the greatest of its
  regions', marched alone, the crease without its measure: camera truth 1's defect) fail the pixel line, every run;
  ``CAMERA_PLANT=crease-lag`` (the caps placed ``LAG`` of the body's step behind it in each moved image, the view's own
  image unchanged) fails the motion line, every run, the pixel line passing.
- **G17, the derivative**: the jvp of the darkness's equivalent radius and centroid in the body's x and z against the
  central difference of the same renderer, judged on the pixels about the caps' rims and on the rest apart; the line is
  the central difference's own error at its last step, stated by the step sweep (``derivative_line``). The held
  march's term is zero: each read point carries the derivative of the image's own held point. The rule grew after its
  first failure (the jvp 2.56 and 3.44 times the difference's own error at turns 4 and 15, the held march's term) and
  was settled by fixing the derivative, not by widening the line; the views it is confirmed on were drawn after.
  **Controls** (``CAMERA_PLANT=held-caps``, the caps' derivative in the body's place gone; ``CAMERA_PLANT=near-axis``, the
  read points of sub-rays passing the side's axis moved in the derivative only).
- **The image's own derivative error, pinned by band** (the maintainer's decision, 2026-11-16): the rendered image's
  derivative (the jvp, which G17 holds to it) against the exact image's, a pixel at a time, in two classes with a line
  for each in each band of the turn (``IMAGE_BANDS``, ``scripts/camera_route2/crease_image_line.py --bands``: 1.25 times
  the 360-pose scan's largest in the band, a margin declared, not derived): **the sliver** (a pixel one of whose
  sub-rays passes within ``NEAR_AXIS`` of the side's axis, at the rod's end with the cap nearly edge-on: 14.9 px² a px
  at worst, where the exact image's is 0) and **the ripple** (every other pixel about the outline: up to 0.88 at caps
  turned under about 8 degrees, under 0.16 from 15 up). Thin caps are judged by their flag wherever the axis passes
  (thin first, the maintainer's word after a fresh view's thin cap's pixel crossed the sliver's line), and under 8
  degrees the sliver has one line for every band (the scan's largest there, at turn 6, times 1.25): one pose's sliver
  moves by about a quarter between machines. A known error of the image, carried and bounded so neither class can grow
  unseen, not a bound on the image's accuracy; the follow-up "creased image accuracy" (the route to exact) removes the
  sliver, after which these lines tighten. Checked at G17's views and at each band's tails (the scan's poses that set
  its lines). **Controls** (``CAMERA_PLANT=grown-ripple``, ``grown-sliver``: the error grown by ``PLANT_SIZE`` px² a px
  in the image itself, which the jvp gate cannot see; ``single-edge-crease``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import _cache_ceiling
import hull_oracle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import PLANT_SIZE, planted
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import mime.sensing._coverage as coverage
import mime.sensing._sampled as sampled_route
from mime.registry import CreaseMeasure, lookup
from mime.registry import EntryKind
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render
from mime.testing import crease_measure_problems

PREDICTIONS = json.loads((Path(__file__).parent / "predictions_camera_pr5b.json").read_text())

#: How far above where a worker stands after giving its compiled programs back this file lets its resident memory grow
#: before it gives them back again (tests/_cache_ceiling.py). Each view compiles its renders afresh (the rod's axis is
#: a static setting) and none is called again, so a worker kept them all: one process running G17's 26 ids held 5.5
#: GiB at its end, G18's 80 drawn views 10.5 GiB; four such workers is more than CI's runner holds (16 GB), and pull
#: request 358's first run lost the runners of this file's shard and stress jobs with no test failing.
HEADROOM_MIB = 1024
LIMIT_MIB: list[float] = []  # one worker's limit: where it stood after the last give-back, plus the headroom


@pytest.fixture(autouse=True)
def programs_given_back():
    """As ``test_lubrication.py``'s: give JAX's compiled programs back once a worker grows past its headroom."""
    yield
    now = _cache_ceiling.resident_mib()
    if now is None:
        return
    if not LIMIT_MIB or now > LIMIT_MIB[0]:
        _cache_ceiling.give_back_programs()
        LIMIT_MIB[:] = [(_cache_ceiling.resident_mib() or now) + HEADROOM_MIB]
#: The lines ``scripts/camera_route2/crease_line.py`` derived (its ``pixel_line`` and ``motion_line``), held in the
#: predictions file before any draw of these gates ran; never fitted.
PIXEL_LINE = PREDICTIONS["lines"]["pixel_line"]
MOTION_LINE = PREDICTIONS["lines"]["motion_line"]

WIDTH, HEIGHT, FOCAL, DISTANCE = 96, 64, 900.0, 30e-3
RADIUS, LENGTH = 0.31e-3, 2.07e-3
CENTRE = np.array((0.113e-3, 0.0, 0.071e-3))
PIXEL = DISTANCE / FOCAL
IDENTITY = (1.0, 0.0, 0.0, 0.0)
CAMERA = Pinhole(WIDTH, HEIGHT, FOCAL, FOCAL, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
POSE = CameraPose(jnp.asarray([0.0, -DISTANCE, 0.0]), jnp.asarray([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]))
#: The views the controls are pinned on: edge-on and nearly so, where the side meets a cap at about a right angle.
PINNED = (0.0, 2.0, 4.0)


# -- the composition's share ---------------------------------------------------------------------------------------------

def _clip(polygon: np.ndarray, nu: float, nv: float, c: float) -> np.ndarray:
    """A convex polygon clipped to ``nu x + nv y <= c``."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        fa, fb = nu * a[0] + nv * a[1] - c, nu * b[0] + nv * b[1] - c
        if fa <= 0:
            out.append(a)
        if (fa <= 0) != (fb <= 0):
            out.append(a + fa / (fa - fb) * (b - a))
    return np.asarray(out)


def exact_share(terms, lines) -> float:
    """The share of the unit cell inside the union of the terms' intersections, by clipping each subset's polygon."""
    import itertools

    total = 0.0
    for count in range(1, len(terms) + 1):
        for subset in itertools.combinations(terms, count):
            polygon = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
            for index in sorted({i for term in subset for i in term}):
                distance, nu, nv = lines[index]
                polygon = _clip(polygon, nu, nv, -distance) if len(polygon) else polygon
            area = 0.0 if len(polygon) < 3 else 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                                                           - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
            total += area if count % 2 else -area
    return total


def allowance(terms, lines) -> float:
    """What the composition's own floor lets its share differ from the exact area by, on these lines: where two lines
    are parallel to within ``UNION_FLOOR`` (or antiparallel) and lie within it of each other, its divisions take their
    slope as the floor, so a share's rounding grows by ``1 / UNION_FLOOR`` and the rule that coinciding lines keep half of
    a chord each misplaces at most the sliver between them (its width, their separation and the angle between them, a
    chord's length long), once in each subset of the terms that holds both. Zero on every other draw, so that a draw
    with no such pair is held to the clipped area's own rounding."""
    import itertools

    total = 0.0
    eps = float(np.finfo(np.float64).eps)
    for (i, (di, ui, vi)), (j, (dj, uj, vj)) in itertools.combinations(enumerate(lines), 2):
        cross = abs(ui * vj - vi * uj)
        separation = abs(di - dj) if ui * uj + vi * vj > 0 else abs(di + dj)
        if cross <= coverage.UNION_FLOOR * math.sqrt(2.0) and separation < coverage.UNION_FLOOR:
            holding = sum(1 for count in range(1, len(terms) + 1) for subset in itertools.combinations(terms, count)
                          if {i, j} <= {k for term in subset for k in term})
            total += holding * (math.sqrt(2.0) * (separation + cross) + 64 * eps / coverage.UNION_FLOOR)
    return total


def capped_terms():
    """A creased body's composition (two branches, each with its span; two caps, each with its ball)."""
    from mime.scene._creases import capped

    def compose(parts, lesser, greater):
        return capped([(parts[0], parts[2]), (parts[1], parts[3])], parts[4:], lesser, greater)

    return coverage.terms_of(compose, 8)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(distances=st.lists(st.floats(-1.2, 1.2), min_size=8, max_size=8),
       angles=st.lists(st.floats(0.0, 2.0 * math.pi), min_size=8, max_size=8))
@example(distances=[0.0, 0.0, 0.0, 0.0, 0.0, 2e-09, 0.0, 0.0], angles=[0.0] * 8)
def test_the_composed_share_is_the_exact_area_of_its_polygons(float64, distances, angles):
    terms = capped_terms()
    lines = [(d, math.cos(a), math.sin(a)) for d, a in zip(distances, angles, strict=True)]
    found = float(coverage.composed_covered(terms, [coverage.Line(*map(jnp.asarray, line)) for line in lines]))
    expected = exact_share(terms, lines)
    if planted("composed-sign"):
        expected = expected + 2.0 * exact_share(terms[:1], lines)
    assert abs(found - expected) <= 1e-12 + allowance(terms, lines), (found, expected)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(distances=st.lists(st.floats(-1.2, 1.2), min_size=2, max_size=2),
       angles=st.lists(st.floats(0.0, 2.0 * math.pi), min_size=2, max_size=2))
@example(distances=[0.0, 0.5], angles=[1.0, 0.0])
@example(distances=[0.5, 0.5], angles=[0.0, 0.0])
@example(distances=[-0.5, 0.3], angles=[math.pi / 2, 2.0])
@example(distances=[0.5, 0.5], angles=[0.5, 0.5])
def test_a_union_of_two_lines_is_exact_where_they_coincide_too(float64, distances, angles):
    """Two one-line terms: the exact share, also where the lines coincide (each keeps half of their one chord) and where
    one lies along the cell's side (``union_covered`` reads an eighth there for a line repeated along a side). The
    examples are the draws that found the side's tie (a line on a side beside one across it, the first found by
    Hypothesis on this file's first run after the gate's draws)."""
    lines = [(d, math.cos(a), math.sin(a)) for d, a in zip(distances, angles, strict=True)]
    found = float(coverage.composed_covered(((0,), (1,)), [coverage.Line(*map(jnp.asarray, line)) for line in lines]))
    assert abs(found - exact_share(((0,), (1,)), lines)) <= 1e-12 + allowance(((0,), (1,)), lines)


@pytest.mark.parametrize(("distance", "exact"), [(0.5, 0.0), (-0.5, 1.0)])
def test_known_gap_union_covered_of_a_line_repeated_along_a_cells_side(float64, distance, exact):
    """**A known gap in camera PR 5a's ``union_covered``, recorded here, not fixed** (a fix would move 5a's compiled
    programs; it is on the maintainer's list). The case: one line along a cell's side (its normal along u, its distance
    from the centre half the cell's side), given as both of the union's lines. A half-plane's union with itself is that
    half-plane: the cell's share is exactly 0 where the body lies beyond the side and 1 where the cell is inside it.
    ``union_covered`` reads 0.125 and 1.125, because the side the line lies along reads half its length inside; the case
    has measure zero among drawn lines. ``composed_covered``, which creased bodies use, is exact there. This test fails
    when the gap is closed, and then asserts the exact share instead."""
    line = coverage.Line(jnp.asarray(distance), jnp.asarray(1.0), jnp.asarray(0.0))
    assert abs(float(coverage.composed_covered(((0,), (1,)), [line, line])) - exact) <= 1e-12
    assert abs(float(coverage.union_covered(line, line)) - (exact + 0.125)) <= 1e-12


# -- G17, the measure's conformance -------------------------------------------------------------------------------------

def cylinder_kind():
    return lookup(EntryKind.GEOMETRY, "cylinder", "0.1.0").cls


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(radius=st.floats(0.1e-3, 1e-3), length=st.floats(0.2e-3, 4e-3),
       axis=st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(0.2, 1)), seed=st.integers(0, 2 ** 31 - 1))
def test_g17_the_cylinders_crease_measure_conforms(float64, radius, length, axis, seed, monkeypatch):
    from mime.units import Quantity

    kind_cls = cylinder_kind()
    settings_si = {"radius": radius, "length": length, "axis": axis}
    if planted("wide-cap"):
        found_measure = kind_cls.crease_measure

        def wide(self, parameters, o, d):
            held = found_measure(self, {**parameters, "radius": 1.01 * parameters["radius"]}, o, d)
            return CreaseMeasure(held.pieces, held.compose, held.faces)

        monkeypatch.setattr(kind_cls, "crease_measure", wide)
    from mime.units import Length

    kind = kind_cls(radius=Quantity(radius, Length), length=Quantity(length, Length), axis=axis)
    rng = np.random.default_rng(seed)
    size = math.hypot(radius, length / 2)
    origins = rng.normal(size=(400, 3))
    origins = 30.0 * size * origins / np.linalg.norm(origins, axis=1, keepdims=True)
    targets = rng.uniform(-1.5 * size, 1.5 * size, size=(400, 3))
    problems = crease_measure_problems(kind, settings_si, origins, targets)
    assert not problems, problems


# -- G18, the cylinder on the sampled route ------------------------------------------------------------------------------

def axis_of(turn: float, roll: float) -> np.ndarray:
    a, r = math.radians(turn), math.radians(roll)
    return np.array([math.cos(a) * math.cos(r), math.sin(a), math.cos(a) * math.sin(r)])


#: The crease's lag behind the body's motion under ``CAMERA_PLANT=crease-lag``, as a share of the motion
#: (``CAMERA_PLANT_SIZE`` sets it): of a one-pixel step, this much of a pixel.
LAG = PLANT_SIZE or 0.05
#: The body's motion from the view's own centre in the image being made (``imaged``), in metres: the lag's reference.
_moved = [np.zeros(3)]


def plant_scheme(monkeypatch) -> None:
    """The gate's controls: the composed line's single edge a pixel, the crease without its measure, or the crease
    lagging the body's motion (its caps placed ``LAG`` of the body's step behind it in each moved image; the view's own
    image is unchanged, so only the motion can see it)."""
    import mime.sensing._render as render_module

    if planted("crease-lag"):
        kind_cls = cylinder_kind()
        found_measure = kind_cls.crease_measure

        def lagging(self, parameters, o, d):
            shift = LAG * _moved[0]
            return found_measure(self, parameters, tuple(o[i] + shift[i] for i in range(3)), d)

        monkeypatch.setattr(kind_cls, "crease_measure", lagging)

    if planted("single-edge-crease"):
        found_box = render_module._box_coverage

        def single(subject, p, o, d, du, dv, corner_samples, form=None, unverified=None):
            return found_box(subject, p, o, d, du, dv, 1, form, unverified)

        monkeypatch.setattr(render_module, "_box_coverage", single)
    if planted("no-crease-measure"):
        import dataclasses

        found_place, found_evaluate = Subject._place_sampled, sampled_route.evaluate

        def unmeasured(self):
            found_place(self)
            marched = self.placed.sampled
            body = dataclasses.replace(marched, regions=(), creases=0, compose=None)

            def evaluate(sampled, p, x, y, z):
                value = found_evaluate(marched, p, x, y, z)
                for region in marched.regions:
                    value = jnp.maximum(value, sampled_route._region(marched, p, region, x, y, z))
                return value

            monkeypatch.setattr(sampled_route, "evaluate", evaluate)
            placed = dataclasses.replace(self.placed, sampled=body, form=sampled_route.form(body))
            object.__setattr__(self, "placed", placed)

        monkeypatch.setattr(Subject, "_place_sampled", unmeasured)
    if planted("held-caps"):
        # The crease measure read from a held pinhole: the caps' values and their lines across the pixel kept, their
        # derivative in the body's place gone (the caps' outlines standing still in the jvp while the body moves).
        kind_cls = cylinder_kind()
        measured = kind_cls.crease_measure

        def held(self, parameters, o, d):
            return measured(self, parameters, tuple(jax.lax.stop_gradient(c) for c in o), d)

        monkeypatch.setattr(kind_cls, "crease_measure", held)
    if planted("near-axis"):
        # The read points of sub-rays within NEAR_AXIS px of the side's axis (where the singular poses' sub-rays pass
        # it) moved an extra PLANT_SIZE (default one) of the body's motion along x and z, in the derivative only.
        found_place = sampled_route._place

        def near_axis(sampled, p, o, d, held, read, made):
            place = found_place(sampled, p, o, d, held, read, made)
            a = jax.lax.stop_gradient(jnp.asarray(p["axis"]))
            a = a / jnp.linalg.norm(a)
            h_o, h_d = jax.lax.stop_gradient(o), jax.lax.stop_gradient(d)
            cross = (h_d[1] * a[2] - h_d[2] * a[1], h_d[2] * a[0] - h_d[0] * a[2], h_d[0] * a[1] - h_d[1] * a[0])
            apart = jnp.abs(sum(h_o[i] * cross[i] for i in range(3))) / jnp.sqrt(sum(c * c for c in cross))
            motion = -((o[0] - h_o[0]) + (o[2] - h_o[2]))
            return place + jnp.where(apart < NEAR_AXIS * PIXEL, (PLANT_SIZE or 1.0) * motion, 0.0)

        monkeypatch.setattr(sampled_route, "_place", near_axis)
    if planted("grown-ripple") or planted("grown-sliver") or planted("confined-sliver"):
        # The image's own derivative error grown: each pixel's share moved GROWTH px^2 a px of the body's motion from
        # the view's own place (along x and z), times its weight, as a ripple of period RIPPLE px (its value within
        # GROWTH x RIPPLE / 2 pi of the scheme's). Value and derivative alike, so the jvp and the central difference
        # both see it and the jvp gate cannot: only the image check can. grown-ripple weighs each partial pixel by
        # 4c(1 - c); grown-sliver weighs by 1 each pixel a sub-ray of which passes within WIDENED px of the side's axis
        # (the sliver's class, four times as wide: in it and beside it).
        found_box = render_module._box_coverage
        sliver = planted("grown-sliver") or planted("confined-sliver")
        # grown-sliver reaches WIDENED times the class's own reach, so its catches under 8 degrees come through the ripple
        # class too; confined-sliver reaches no further than the class does (a refuting review's finding).
        reach = 1.0 if planted("confined-sliver") else WIDENED
        growth = PLANT_SIZE or 1.0

        def grown(subject, p, o, d, du, dv, corner_samples, form=None, unverified=None):
            share = found_box(subject, p, o, d, du, dv, corner_samples, form, unverified)
            home = np.array([0.0, -DISTANCE, 0.0]) - _view_centre[0]
            place = ((home[0] - o[0]) + (home[2] - o[2])) / PIXEL
            ripple = growth * RIPPLE / (2 * math.pi) * jnp.sin(2 * math.pi * place / RIPPLE)
            if not sliver:
                return share + ripple * 4.0 * share * (1.0 - share)
            a = jax.lax.stop_gradient(jnp.asarray(p["axis"]))
            a = a / jnp.linalg.norm(a)
            h_o = [jax.lax.stop_gradient(c) for c in o]
            least = jnp.full_like(share, jnp.inf)
            for su, sv in np.asarray(coverage.sub_offsets(corner_samples), dtype=float):
                ray = [jax.lax.stop_gradient(d[i] + su * du[i] + sv * dv[i]) for i in range(3)]
                cross = (ray[1] * a[2] - ray[2] * a[1], ray[2] * a[0] - ray[0] * a[2], ray[0] * a[1] - ray[1] * a[0])
                apart = jnp.abs(sum(h_o[i] * cross[i] for i in range(3))) / jnp.sqrt(sum(c * c for c in cross))
                least = jnp.minimum(least, apart)
            return share + jnp.where(least < reach * NEAR_AXIS * PIXEL, ripple, 0.0)

        monkeypatch.setattr(render_module, "_box_coverage", grown)


#: The near-axis plant's reach from the side's axis, in pixels; the image's sliver class.
NEAR_AXIS = 0.01
#: The grown-sliver plant's reach, as a share of ``NEAR_AXIS``, and the growth plants' ripple's period, in pixels of
#: the body's motion (about the image's own ripple's, 0.05 to 0.1).
WIDENED, RIPPLE = 4.0, 0.1
#: The view's own centre while it is imaged (the growth plants' reference place).
_view_centre = [CENTRE]


def rod(axis, route: str | None = "sampled") -> Subject:
    return Subject("rod", "cylinder", {"radius": RADIUS, "length": LENGTH, "axis": tuple(float(a) for a in axis)},
                   appearance=Appearance.opaque(), moving=True, route=route)


def imaged(body: Subject, centre: np.ndarray, moved: np.ndarray | None = None):
    """The body's image at ``centre``; ``moved``, the step from its view's own centre (the lag plant's reference)."""
    _moved[0] = np.zeros(3) if moved is None else moved
    found = render([body], CAMERA, POSE, {"rod": (jnp.asarray(centre), jnp.asarray(IDENTITY))})
    return 1.0 - np.asarray(found.transmission, dtype=np.float64), found


def judged_pixels(centre: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, bool]:
    """The pixels the line judges (not about a cap narrower than two sub-pixels) and whether any cap is that thin."""
    judged = np.ones((HEIGHT, WIDTH), dtype=bool)
    thin = False
    rims = hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH)
    for end, rim in zip((-LENGTH / 2, LENGTH / 2), rims, strict=True):
        if hull_oracle.minor_axis(centre + end * axis, axis, RADIUS, FOCAL, DISTANCE) < 2.0 / coverage.CORNER_SAMPLES:
            thin = True
            projected = hull_oracle.project(rim, FOCAL, DISTANCE, WIDTH, HEIGHT)
            low, high = np.floor(projected.min(axis=0)).astype(int) - 2, np.ceil(projected.max(axis=0)).astype(int) + 2
            judged[max(0, low[1]):high[1] + 1, max(0, low[0]):high[0] + 1] = False
    return judged, thin


def aggregates(covered: np.ndarray) -> np.ndarray:
    v, u = np.indices(covered.shape)
    total = covered.sum()
    return np.array([math.sqrt(total / math.pi), (covered * u).sum() / total, (covered * v).sum() / total])


def judged_view(turn: float, roll: float, offset: tuple[float, float], dtype: str, monkeypatch) -> dict:
    plant_scheme(monkeypatch)
    axis = axis_of(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * PIXEL
    covered, frame = imaged(rod(axis), centre)
    exact = hull_oracle.shares(hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH), FOCAL, DISTANCE, WIDTH,
                               HEIGHT)
    judged, thin = judged_pixels(centre, axis)
    pixel = float(np.max(np.abs(covered - exact)[judged]))
    motion = 0.0
    for step in (np.array([PIXEL, 0.0, 0.0]), np.array([0.0, 0.0, PIXEL])):
        moved, _ = imaged(rod(axis), centre + step, step)
        moved_exact = hull_oracle.shares(hull_oracle.cylinder_outline(centre + step, axis, RADIUS, LENGTH), FOCAL,
                                         DISTANCE, WIDTH, HEIGHT)
        scheme = aggregates(moved) - aggregates(covered)
        truth = aggregates(moved_exact) - aggregates(exact)
        motion = max(motion, float(np.linalg.norm(scheme - truth) / np.linalg.norm(truth)))
    return {"pixel": pixel, "motion": motion, "thin": thin, "flag": bool(frame.flags.get("thin_cap:rod", False)),
            "line": PIXEL_LINE[dtype]}


def failures(found: dict) -> list[str]:
    problems = []
    if not found["pixel"] <= found["line"]:
        problems.append(f"a pixel {found['pixel']:.4f} from the exact image, past the line {found['line']:.4f}")
    if not found["motion"] <= MOTION_LINE:
        problems.append(f"the motion {found['motion']:.4g} from the exact image's, past the line {MOTION_LINE:.4g}")
    if found["flag"] != found["thin"]:
        problems.append(f"thin_cap {found['flag']} where a cap is {'' if found['thin'] else 'not '}that thin")
    return problems


@pytest.mark.parametrize("turn", PINNED)
def test_g18_the_pinned_views_are_within_the_lines(dtype, turn, monkeypatch):
    found = judged_view(turn, 0.0, (0.0, 0.0), dtype, monkeypatch)
    print(f"\nturn {turn} {dtype}: {found}")
    assert not failures(found), failures(found)


#: The drawn views' turns, 0 to 30 degrees, in ``DRAWN_GROUPS`` bands each drawn apart, ``DRAWN_A_GROUP`` views a band
#: (40 a dtype, where one test drew 30 over the whole range): about a minute a test on a runner, every band of the range
#: drawn on every run.
DRAWN_GROUPS = 10
DRAWN_TURNS = 30.0
DRAWN_A_GROUP = 4


@pytest.mark.parametrize("group", range(DRAWN_GROUPS))
@settings(max_examples=DRAWN_A_GROUP, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(within=st.floats(0.0, 1.0), roll=st.floats(0.0, 90.0), du=st.floats(0.0, 1.0), dv=st.floats(0.0, 1.0))
def test_g18_drawn_views_are_within_the_lines(dtype, group, within, roll, du, dv, monkeypatch):
    turn = (group + within) * DRAWN_TURNS / DRAWN_GROUPS
    found = judged_view(turn, roll, (du, dv), dtype, monkeypatch)
    assert not failures(found), failures(found)


#: The drawn views' tails, pinned after the draws (the seventh probe's 200 fresh views a dtype, run 43814057533;
#: ``predictions_camera_pr5b.json``, ``outcomes.g18_drawn``): in each dtype the two views nearest the pixel line, then
#: the two nearest the motion line, as (turn, roll, offset in pixels). The gate's three controls fail on each, every
#: run, but where stated (``outcomes.tails``).
TAILS: dict[str, tuple[tuple[float, float, tuple[float, float]], ...]] = {
    "float32": (
        (0.08318973443307898, 7.789569005130011, (0.6963921755975593, 0.09691402065594068)),
        (4.167819566112463, 50.84929041749485, (0.343092926225011, 0.89152017722661)),
        (2.0018660999306137, 87.20270246113799, (0.8755239964813695, 0.1304005230546309)),
        (1.691199284926428, 52.80791933064757, (0.2071278397381736, 0.24822654209717077)),
    ),
    "float64": (
        (0.08969212078949074, 40.233265026838254, (0.2574786643014014, 0.954961322547809)),
        (3.638260898078464, 50.21308799258834, (0.765496009300265, 0.12864774877105367)),
        (1.38492085197503, 82.77012271551312, (0.8824135620510877, 0.5311001664347395)),
        (1.9035429443756058, 29.755977849879795, (0.27483079357254747, 0.47060691278067646)),
    ),
}


@pytest.mark.parametrize("tail", range(4))
def test_g18_the_drawn_tails_are_within_the_lines(dtype, tail, monkeypatch):
    turn, roll, offset = TAILS[dtype][tail]
    found = judged_view(turn, roll, offset, dtype, monkeypatch)
    print(f"\ntail {tail} {dtype} (turn {turn}, roll {roll}, offset {offset}): {found}")
    assert not failures(found), failures(found)


def test_g18_route_one_and_route_two_each_hold_to_the_exact_image(dtype):
    """The swap: at one view the closed form (its gate's line, ``test_corners.LINE``) and the sampled route (this gate's)
    each within its line of the same exact image, and their motions within the sum of their motion's distances."""
    axis = axis_of(7.0, 33.0)
    exact = hull_oracle.shares(hull_oracle.cylinder_outline(CENTRE, axis, RADIUS, LENGTH), FOCAL, DISTANCE, WIDTH,
                               HEIGHT)
    one, _ = imaged(rod(axis, None), CENTRE)
    two, _ = imaged(rod(axis), CENTRE)
    assert float(np.max(np.abs(one - exact))) <= 1.0 / coverage.CORNER_SAMPLES ** 2
    assert float(np.max(np.abs(two - exact))) <= PIXEL_LINE[dtype]


# -- G17, the derivative ------------------------------------------------------------------------------------------------

#: The sweep's steps, in pixels of the body's motion: a quarter apart, from 1/64 to 1/16384 of a pixel. Near a cap seen
#: almost edge-on the image's derivative turns within a few hundredths of a pixel of motion (the cap's outline curves
#: within a sub-pixel), so the central difference reaches its own small-step behaviour only below about 1/1000 of a
#: pixel; float64's rounding is still far below it at the last step.
STEPS = tuple(4.0 ** -k for k in range(3, 8))
#: The exact image's step for the image check (``hull_oracle``'s central difference, settled at it in the singular
#: check), in pixels.
EXACT_STEP = 1e-4
#: The derivative's views (turn, roll, offset in pixels). ``DERIVATIVE_SEEN``: the first forms' views, measured before
#: the held march's term was found (their failure at turns 4 and 15 is why it was looked for); ``DERIVATIVE_SINGULAR``:
#: the three poses where a sub-ray passes the side's axis at the rod's end (``crease_singular_check.py``), measured in
#: ``crease_held_point_check.py``. Both kept, neither evidence. ``DERIVATIVE_FIRST_FRESH``: the first fresh draw, run
#: under the first class order (the sliver's before the thin's), where one view's thin cap's pixel crossed the sliver's
#: line; the order then moved on the maintainer's word, so they are seen views, pinned. ``DERIVATIVE_FRESH``: drawn from
#: fresh entropy after the class order and the lines were committed, their predictions committed before they ran
#: (``predictions_camera_pr5b.json``, ``image_guard`` and ``image_guard_thin_first``).
DERIVATIVE_SEEN = tuple((turn, 20.0, (0.0, 0.0)) for turn in (0.0, 2.0, 4.0, 15.0))
DERIVATIVE_SINGULAR = ((6.0, 75.0, (0.3, 0.8)), (3.0, 0.0, (0.0, 0.0)), (2.0, 0.0, (0.0, 0.0)))
#: Drawn 2026-11-16 by ``numpy.random.default_rng(10666201688275216299)`` (the seed from fresh entropy): turn U(0, 30),
#: roll U(0, 90), offsets U(0, 1).
DERIVATIVE_FIRST_FRESH: tuple[tuple[float, float, tuple[float, float]], ...] = (
    (1.3482647412161108, 24.808128585345663, (0.7863700316759962, 0.129462421621917)),
    (1.8428440541539104, 19.019379682163457, (0.8730422629648931, 0.21199331858582915)),
    (29.36202677012414, 87.47250430703737, (0.7066187859364247, 0.34617106753450977)),
    (2.5482240708320045, 52.449788818546224, (0.13916151897523987, 0.4291590510679173)),
    (11.932991353146052, 18.46225045457095, (0.6634188681067769, 0.3227586724944974)),
    (21.635710677105802, 66.65891259309886, (0.1369225864345428, 0.20859012790484321)),
)
#: Drawn 2026-11-16 by ``numpy.random.default_rng(12623130533721122703)`` (the seed from fresh entropy), after the class
#: order and the lines were committed: turn U(0, 30), roll U(0, 90), offsets U(0, 1).
DERIVATIVE_FRESH: tuple[tuple[float, float, tuple[float, float]], ...] = (
    (4.955053700369372, 55.12693375836997, (0.9527046003383626, 0.9431572270694022)),
    (11.260245957246276, 11.723617036975805, (0.5122441478803482, 0.8746822271780829)),
    (5.639246213494711, 60.89595119835083, (0.5880991766778857, 0.041059588390126245)),
    (7.6736007349550945, 46.778119015871994, (0.7861468838228813, 0.25967823307132387)),
    (12.38555855493949, 51.82453720642759, (0.39766334656559943, 0.26379711316504206)),
    (16.62318344638479, 51.742260318465, (0.01654780985922355, 0.8605970733429448)),
)


def derivative_line(differences: list[np.ndarray], floor: np.ndarray) -> float:
    """What the jvp's motion may lie from the central difference's at the sweep's last step, in the norm of the motion
    (the equivalent radius and the centroid, in pixels a pixel): the difference's own error, twice the change between
    its last two steps (a quarter apart, so where its error is second order in the step, thirty times its error at the
    last), and its rounding there. The held march's own term is zero: each read point carries the derivative of the
    image's own held point (``_sampled._place``), and the implicit step's slope and curvature are read at it as it moves
    (``_sampled._moving``), so the jvp is the rendered image's derivative on its branch in the body's place; the term
    ``scripts/camera_route2/crease_march_line.py`` derived (the jvp's held points against the image's, at most 9.8e-5
    px a px at 357 of its 360 poses and 0.50 at one pixel of three) is the one the read points no longer leave out."""
    return float(np.linalg.norm(2.0 * (differences[-1] - differences[-2])) + np.linalg.norm(floor))


def crease_mask(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The pixels within two of the caps' rims' images."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for rim in hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH):
        projected = hull_oracle.project(rim, FOCAL, DISTANCE, WIDTH, HEIGHT)
        for u, v in np.round(projected).astype(int)[::20]:
            mask[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = True
    return mask


def _view_id(view) -> str:
    turn, roll, (du, dv) = view
    return f"t{turn:g}-r{roll:g}-u{du:g}-v{dv:g}"


# -- the image's own derivative error, pinned by band ---------------------------------------------------------------------

#: The image check's lines by band of the turn (``scripts/camera_route2/crease_image_line.py --bands``, held in the
#: predictions file before any fresh view): in each band, for the ripple's class (``rest``) and the sliver's
#: (``singular``), 1.25 times the 360-pose scan's largest error over its poses in the closed band, in pixels of share a
#: pixel of the body's motion, with the pose that set it (its tail); under 8 degrees the sliver's line is one for every
#: band, over all the poses there (the bands' own beside it, ``singular_band``). Never fitted; moved once, on the
#: maintainer's word (thin first, and the sliver's one line), the scan re-run before any view of the second draw.
IMAGE_BANDS = PREDICTIONS["lines"]["image_bands"]["bands"]
#: Each band's tails, the scan's poses that set its lines, once each: pinned, so a growth of either class past the
#: declared margin at the pose where it was largest fails, whatever the drawn views find.
IMAGE_TAILS = tuple(sorted({(tail["turn"], tail["roll"], tuple(tail["offset"]))
                            for band in IMAGE_BANDS for tail in (band["rest_tail"], band["singular_tail"])}))


def image_lines(turn: float) -> dict[str, float]:
    """Each judged class's line at a view's turn: the strictest of the bands whose closed range holds it (both, on an
    edge, as the scan's poses there set both)."""
    holding = [band for band in IMAGE_BANDS if band["turns"][0] <= turn <= band["turns"][1]]
    assert holding, f"turn {turn} outside the scanned bands"
    return {name: min(band[name] for band in holding) for name in ("rest", "singular")}


def axis_distance(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Each pixel's least distance of its sub-rays from the side's axis line, in pixels at the body (as
    ``crease_image_line.axis_distance``)."""
    v, u = (x.ravel().astype(float) for x in np.mgrid[0:HEIGHT, 0:WIDTH])
    raw = np.stack([(u - (WIDTH - 1) / 2) / FOCAL, np.ones_like(u), -(v - (HEIGHT - 1) / 2) / FOCAL], -1)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    d = raw / norm
    e_u, e_v = np.array([1.0 / FOCAL, 0.0, 0.0]), np.array([0.0, 0.0, -1.0 / FOCAL])
    du = e_u / norm - d * (d @ e_u)[..., None] / norm
    dv = e_v / norm - d * (d @ e_v)[..., None] / norm
    o = np.array([0.0, -DISTANCE, 0.0]) - centre
    least = np.full(len(u), np.inf)
    for su, sv in coverage.sub_offsets(coverage.CORNER_SAMPLES):
        cross = np.cross(d + su * du + sv * dv, axis)
        least = np.minimum(least, np.abs(cross @ o) / np.linalg.norm(cross, axis=-1))
    return (least / PIXEL).reshape(HEIGHT, WIDTH)


def exact_shares(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return hull_oracle.shares(hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH), FOCAL, DISTANCE, WIDTH, HEIGHT)


def image_classes(centre: np.ndarray, axis: np.ndarray) -> dict[str, np.ndarray]:
    """The band's pixels (within two of a partial pixel of the exact image) by class, as the scan set them apart, thin
    first (the maintainer's word, F0 record item 103): ``thin`` (about a cap narrower than two sub-pixels, judged by its
    flag wherever the axis passes), ``singular`` (a sub-ray within ``NEAR_AXIS`` of the side's axis, outside thin boxes)
    and ``rest``."""
    exact = exact_shares(centre, axis)
    partial = (exact > 0) & (exact < 1)
    band = partial.copy()
    for dj in range(-2, 3):
        for di in range(-2, 3):
            band |= np.roll(np.roll(partial, dj, 0), di, 1)
    judged, _ = judged_pixels(centre, axis)
    singular = band & judged & (axis_distance(centre, axis) < NEAR_AXIS)
    return {"rest": band & judged & ~singular, "singular": singular, "thin": band & ~judged}


def image_check(turn: float, centre: np.ndarray, axis: np.ndarray, moved: dict[int, np.ndarray], flag: bool
                ) -> tuple[list[str], float]:
    """Each band pixel's image error, the rendered image's derivative along x and z (``moved``, by the body's axis
    index, in px^2 a px) less the exact image's (``hull_oracle``'s central difference at ``EXACT_STEP``), within its
    class's line at the view's turn; thin caps by their flag. The problems, and the largest error as a share of its
    line."""
    lines = image_lines(turn)
    classes = image_classes(centre, axis)
    problems, worst = [], 0.0
    for index, tangent in moved.items():
        e = np.zeros(3)
        e[index] = PIXEL * EXACT_STEP
        exact = (exact_shares(centre + e, axis) - exact_shares(centre - e, axis)) / (2 * EXACT_STEP)
        error = np.abs(tangent - exact)
        for name, mask in classes.items():
            if not mask.any():
                continue
            at = np.unravel_index(int(np.argmax(np.where(mask, error, -1.0))), error.shape)
            line = lines.get(name)
            judged = f"{error[at] / line:.3g} of its line {line:.3g}" if line else "judged by the flag"
            print(f"  the image's error along {'xyz'[index]}, {name} ({int(mask.sum())} pixels): largest "
                  f"{error[at]:.4g} at v{at[0]} u{at[1]} ({judged})")
            if line:
                worst = max(worst, float(error[at]) / line)
                if not error[at] <= line:
                    problems.append(f"the {name} class's image error {error[at]:.4g} along {'xyz'[index]} at "
                                    f"v{at[0]} u{at[1]}, past its band's line {line:.4g}")
    _, thin = judged_pixels(centre, axis)
    if flag != thin:
        problems.append(f"thin_cap {flag} where a cap is {'' if thin else 'not '}that thin")
    return problems, worst


@pytest.mark.parametrize("view", DERIVATIVE_SEEN + DERIVATIVE_SINGULAR + DERIVATIVE_FIRST_FRESH + DERIVATIVE_FRESH,
                         ids=_view_id)
def test_g17_the_creased_derivative_is_the_images(float64, view, monkeypatch):
    """The jvp of the darkness's equivalent radius and centroid, on the pixels about the caps' rims and on the rest
    apart, in the body's x and z, against the central difference of the same renderer at the sweep's last step, within
    ``derivative_line``; and the image check (``image_check``) on the jvp's image. **Controls** (``CAMERA_PLANT``):
    ``held-caps`` (the crease measure read from a held pinhole, the caps' derivative in the body's place gone) and
    ``near-axis`` (the read points of sub-rays within a hundredth of a pixel of the side's axis moved an extra one of the
    body's motion, in the derivative only: at the singular poses) fail the jvp gate; ``grown-ripple``, ``grown-sliver``
    and ``single-edge-crease`` change the image, which the jvp follows, and fail only the image check."""
    plant_scheme(monkeypatch)
    turn, roll, offset = view
    axis = axis_of(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * PIXEL
    _view_centre[0] = centre
    body = rod(axis)
    crease = crease_mask(centre, axis)
    masks = (crease, ~crease)
    v, u = np.indices((HEIGHT, WIDTH))

    def dark(at):
        found = render([body], CAMERA, POSE, {"rod": (at, jnp.asarray(IDENTITY))})
        return 1.0 - found.transmission, jnp.asarray(found.flags.get("thin_cap:rod", False), jnp.float64)

    def aggregates_of(image):
        return np.stack([aggregates(np.asarray(image, np.float64) * mask) for mask in masks])

    def aggregates_moved(image, tangent):
        # The aggregates' derivative from the image's (they are made from it): the radius's, then the centroid's.
        rows = []
        for mask in masks:
            covered, moved = image * mask, tangent * mask
            radius, cu, cv = aggregates(covered)
            total = covered.sum()
            rows.append([moved.sum() / (2 * math.pi * radius), ((u - cu) * moved).sum() / total,
                         ((v - cv) * moved).sum() / total])
        return np.asarray(rows)

    values = jax.jit(dark)
    tangents = jax.jit(lambda at, e: jax.jvp(dark, (at,), (e,))[1][0])
    image, flag = (np.asarray(x, np.float64) for x in values(jnp.asarray(centre)))
    scale = np.abs(aggregates_of(image))
    floor = 64 * float(np.finfo(np.float64).eps) * scale / (PIXEL * STEPS[-1])
    worst, moved = 0.0, {}
    for axis_index in (0, 2):
        e = np.zeros(3)
        e[axis_index] = 1.0
        moved[axis_index] = np.asarray(tangents(jnp.asarray(centre), jnp.asarray(e)), np.float64) * PIXEL
        tangent = aggregates_moved(image, moved[axis_index])
        differences = [(aggregates_of(values(jnp.asarray(centre + PIXEL * step * e))[0])
                        - aggregates_of(values(jnp.asarray(centre - PIXEL * step * e))[0])) / (2 * step)
                       for step in STEPS]
        for mask, name in enumerate(("crease", "rest")):
            held = [difference[mask] for difference in differences]
            line = derivative_line(held, floor[mask] * PIXEL)
            apart = float(np.linalg.norm(tangent[mask] - held[-1]))
            worst = max(worst, apart / line)
            print(f"\n{_view_id(view)}, axis {axis_index}, {name}: jvp {tangent[mask]} (px a px)\n"
                  f"  differences {np.asarray(held)}\n  apart {apart:.3g}, line {line:.3g} ({apart / line:.3g} of it)")
    problems, image_worst = image_check(turn, centre, axis, moved, bool(flag))
    print(f"\n{_view_id(view)}: the jvp's largest distance from the central difference, as a share of its line "
          f"{worst:.3g}; the image's largest error as a share of its band's line {image_worst:.3g}")
    assert worst <= 1.0 and not problems, (worst, problems)


@pytest.mark.parametrize("view", IMAGE_TAILS, ids=_view_id)
def test_g17_the_image_error_is_within_its_band_at_the_bands_tails(float64, view, monkeypatch):
    """The image check (``image_check``) at the scan's poses that set each band's lines: each class's largest about
    0.8 of its line there (1 over the declared margin), so a growth of more than a quarter where it was largest fails.
    **Controls** (``CAMERA_PLANT``): ``grown-ripple`` and ``grown-sliver`` (the image's own error grown, ``PLANT_SIZE``
    px^2 a px)."""
    plant_scheme(monkeypatch)
    turn, roll, offset = view
    axis = axis_of(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * PIXEL
    _view_centre[0] = centre
    body = rod(axis)

    def dark(at):
        found = render([body], CAMERA, POSE, {"rod": (at, jnp.asarray(IDENTITY))})
        return 1.0 - found.transmission, jnp.asarray(found.flags.get("thin_cap:rod", False), jnp.float64)

    # The flag from the image's program and the tangents from one of the image alone, as G17's views take them: one
    # program of both, primal and tangent, peaked at 5.0 GiB alone in a process, these two at 2.2 to 2.3 (a laptop,
    # /usr/bin/time, the turn-20 and turn-6 tails).
    values = jax.jit(dark)
    tangents = jax.jit(lambda at, e: jax.jvp(dark, (at,), (e,))[1][0])
    flag = bool(np.asarray(values(jnp.asarray(centre))[1]))
    moved = {}
    for axis_index in (0, 2):
        e = np.zeros(3)
        e[axis_index] = 1.0
        moved[axis_index] = np.asarray(tangents(jnp.asarray(centre), jnp.asarray(e)), np.float64) * PIXEL
    print(f"\n{_view_id(view)}:")
    problems, worst = image_check(turn, centre, axis, moved, flag)
    print(f"  the image's largest error as a share of its band's line {worst:.3g}")
    assert not problems, problems
