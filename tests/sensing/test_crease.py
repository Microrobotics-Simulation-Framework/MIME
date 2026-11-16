"""Crease measures on the sampled route (camera PR 5b): a creased body's caps as exact edge functions, composed with
the march's branches and their regions, each corner cell the exact share of that composition.

- **The composition's share** (``coverage.composed_covered``): a union of intersections of half-planes in a cell, by
  inclusion and exclusion of each subset's convex polygon's boundary integral, against the same polygons clipped
  exactly (Sutherland-Hodgman) on drawn lines, coinciding lines and lines along a cell's side included. **Control** (``CAMERA_PLANT=composed-sign``): one subset's sign flipped fails, every run.
- **G17, the crease measure's conformance** (``mime.testing.crease_measure_problems``): the cylinder's caps on drawn
  rays have the body's sign at their planes' crossings and vanish on its drawing's rims. **Control**
  (``CAMERA_PLANT=wide-cap``): the measure's caps one per cent wide are caught on every run.
- **G18, the cylinder on the sampled route** (``route="sampled"``; its default stays the closed form): every pixel
  against the exact image (the hull of the rims' images, ``hull_oracle``) within ``PIXEL_LINE``, and the silhouette's
  motion (the secant over one pixel of the body's motion along x and z of the darkness's total and centroid) within
  ``MOTION_LINE`` of the exact image's, rods turned 0 to 30 degrees towards the line of sight, rolled and moved by a
  fraction of a pixel, in both dtypes; caps narrower than two sub-pixels raise ``thin_cap`` and the pixels about them
  are judged by the flag. Both lines are ``scripts/camera_route2/crease_line.py``'s, derived before any of these draws
  from the scheme's own arithmetic on the exact geometry (``predictions_camera_pr5b.json``). Against route 1 the swap
  holds each route to its line against the same exact image. **Controls**, on the pinned views (turns 0, 2 and 4,
  edge-on to nearly so, where the side meets a cap at about a right angle): ``CAMERA_PLANT=single-edge-crease`` (each
  pixel one straight edge, the composed line's, no sub-samples: cycle 4's corner error) and
  ``CAMERA_PLANT=no-crease-measure`` (the body's first-order distance, the greatest of its regions', marched alone,
  the crease without its measure: camera truth 1's defect) fail, every run.
- **G17, the derivative**: the jvp of the darkness's total and centroid in the body's x and z against the central
  difference of the same renderer, judged on the pixels about the caps' rims and on the rest apart; the line is the
  central difference's own error at its step, stated by the step sweep (``derivative_line``: the difference between
  the central differences at the step and at half of it, twice, and their rounding), never a round number.
  **Control** (``CAMERA_PLANT=no-crease-measure``) fails, every run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import hull_oracle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.sensing._coverage as coverage
import mime.sensing._sampled as sampled_route
from mime.registry import CreaseMeasure, lookup
from mime.registry import EntryKind
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render
from mime.testing import crease_measure_problems

PREDICTIONS = json.loads((Path(__file__).parent / "predictions_camera_pr5b.json").read_text())
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


def capped_terms():
    """A creased body's composition (two branches, each with its span; two caps, each with its ball)."""
    from mime.scene._creases import capped

    def compose(parts, lesser, greater):
        return capped([(parts[0], parts[2]), (parts[1], parts[3])], parts[4:], lesser, greater)

    return coverage.terms_of(compose, 8)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(distances=st.lists(st.floats(-1.2, 1.2), min_size=8, max_size=8),
       angles=st.lists(st.floats(0.0, 2.0 * math.pi), min_size=8, max_size=8))
def test_the_composed_share_is_the_exact_area_of_its_polygons(float64, distances, angles):
    terms = capped_terms()
    lines = [(d, math.cos(a), math.sin(a)) for d, a in zip(distances, angles, strict=True)]
    found = float(coverage.composed_covered(terms, [coverage.Line(*map(jnp.asarray, line)) for line in lines]))
    expected = exact_share(terms, lines)
    if planted("composed-sign"):
        expected = expected + 2.0 * exact_share(terms[:1], lines)
    assert abs(found - expected) <= 1e-12, (found, expected)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(distances=st.lists(st.floats(-1.2, 1.2), min_size=2, max_size=2),
       angles=st.lists(st.floats(0.0, 2.0 * math.pi), min_size=2, max_size=2))
def test_a_union_of_two_lines_is_exact_where_they_coincide_too(float64, distances, angles):
    """Two one-line terms: the exact share, also where the lines coincide (each keeps half of their one chord) and where
    one lies along the cell's side (``union_covered`` reads an eighth there for a line repeated along a side)."""
    lines = [(d, math.cos(a), math.sin(a)) for d, a in zip(distances, angles, strict=True)]
    found = float(coverage.composed_covered(((0,), (1,)), [coverage.Line(*map(jnp.asarray, line)) for line in lines]))
    assert abs(found - exact_share(((0,), (1,)), lines)) <= 1e-12


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


def plant_scheme(monkeypatch) -> None:
    """The gate's controls: the composed line's single edge a pixel, or the crease without its measure."""
    import mime.sensing._render as render_module

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


def rod(axis, route: str | None = "sampled") -> Subject:
    return Subject("rod", "cylinder", {"radius": RADIUS, "length": LENGTH, "axis": tuple(float(a) for a in axis)},
                   appearance=Appearance.opaque(), moving=True, route=route)


def imaged(body: Subject, centre: np.ndarray):
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
        moved, _ = imaged(rod(axis), centre + step)
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


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(turn=st.floats(0.0, 30.0), roll=st.floats(0.0, 90.0), du=st.floats(0.0, 1.0), dv=st.floats(0.0, 1.0))
def test_g18_drawn_views_are_within_the_lines(dtype, turn, roll, du, dv, monkeypatch):
    found = judged_view(turn, roll, (du, dv), dtype, monkeypatch)
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

def derivative_line(differences: list[np.ndarray], floor: np.ndarray) -> np.ndarray:
    """The central difference's own error at its step, stated by the sweep: twice the change between the differences
    at the step and at half of it (the last two of the sweep), and their rounding."""
    return 2.0 * np.abs(differences[-1] - differences[-2]) + floor


def crease_mask(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The pixels within two of the caps' rims' images."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for rim in hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH):
        projected = hull_oracle.project(rim, FOCAL, DISTANCE, WIDTH, HEIGHT)
        for u, v in np.round(projected).astype(int)[::20]:
            mask[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = True
    return mask


@pytest.mark.parametrize("turn", PINNED + (15.0,))
def test_g17_the_creased_derivative_is_the_images(float64, turn, monkeypatch):
    plant_scheme(monkeypatch)
    axis = axis_of(turn, 20.0)
    body = rod(axis)
    crease = crease_mask(CENTRE, axis)
    v, u = (jnp.asarray(x, jnp.float64) for x in np.indices((HEIGHT, WIDTH)))

    def aggregates_on(at, mask):
        dark = (1.0 - render([body], CAMERA, POSE, {"rod": (at, jnp.asarray(IDENTITY))}).transmission) * mask
        total = jnp.sum(dark)
        return jnp.stack([total, jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    found = []
    for mask in (crease, ~crease):
        held = jnp.asarray(mask, jnp.float64)
        values = jax.jit(lambda at: aggregates_on(at, held))
        for axis_index in (0, 2):
            e = np.zeros(3)
            e[axis_index] = 1.0
            tangent = np.asarray(jax.jvp(lambda at: aggregates_on(at, held), (jnp.asarray(CENTRE),),
                                         (jnp.asarray(e),))[1])
            differences = []
            for k in range(1, 5):
                h = PIXEL / 4 ** k
                differences.append((np.asarray(values(jnp.asarray(CENTRE + h * e)))
                                    - np.asarray(values(jnp.asarray(CENTRE - h * e)))) / (2 * h))
            scale = np.abs(np.asarray(values(jnp.asarray(CENTRE))))
            floor = 64 * float(np.finfo(np.float64).eps) * scale / (PIXEL / 4 ** 4)
            line = derivative_line(differences, floor)
            found.append((np.abs(tangent - differences[-1]), line))
    worst = max(float(np.max(apart / np.maximum(line, 1e-300))) for apart, line in found)
    print(f"\nturn {turn}: the jvp's largest distance from the central difference, as a share of its own error {worst:.3g}")
    assert worst <= 1.0
