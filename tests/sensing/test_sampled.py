"""Route 2, the sampled route (``mime.sensing._sampled``), against route 1 and the sign-only truth.

- **G18, the swap test.** A sphere (its box edge) and a capsule imaged by route 2 (``route="sampled"``) against route 1,
  per pixel and in the silhouette's motion, in float64 and float32. Their sign functions are their exact distances, so
  route 2 differs from route 1 only by where the march holds its point (the parabola's error, read at the second order).
  **Control** (``CAMERA_PLANT=sampled-offset``): the march's first-order distance offset by a tenth of a pixel fails it,
  every run. (Its first plant, the distance 10 % short, cancelled: the line is the distance over its gradient, and both
  were scaled; 0 of 30 runs caught it, so it said nothing.)
- **The chord**: an attenuating sphere's transmission by route 2's chord (its samples' sign changes, interpolated)
  against route 1's closed-form chord.
- **The located error's fix** (camera PR 5's first step): on a helix, whose turns cross in its image, the pixels where
  two branches of its outline meet (two near-zero minima along the ray) take route 1's corner sub-samples of the two
  branches' union; against the sign-only truth (``sampled_truth``), every such pixel is within ``CORNER_LINE`` and the
  single edge (the second branch dropped) is not. **Control** (``CAMERA_PLANT=single-edge``, a pose drawn from
  ``DETECTION_SEED``): the single edge fails it, every run; the line is derived before the rate run
  (``predictions_camera_pr5a.json``).
- **G21, the window**: a moving helix rendered in a window equals its whole-frame render where nothing overflows, and
  its overflow flag stays down over poses within the declared motion; **control** (``CAMERA_PLANT=small-window``): the
  window at 0.3 of its size raises the flag and its image differs, every run.
- **The held march's derivative** (the switch scan's ``cleared_by`` for its ``stop_gradient``): route 2's forward-mode
  derivative of the image's darkness and its centroid equals the central difference of the same renderer."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sampled_truth
from camera_cases import SEED, across, planted

import mime.sensing._sampled as sampled_route
from mime.core import quaternion
from mime.sensing import Appearance, Subject, render
from mime.sensing import _coverage as coverage
from mime.sensing import _render as render_module

#: Route 2 against route 1 on a body whose sign function is its exact distance: the largest per-pixel difference in a
#: pixel's share, and the silhouette's motion's relative difference. Declared in predictions_camera_pr5a.json.
SWAP_PIXEL_LINE = 2e-3
SWAP_MOTION_LINE = 2e-3
#: The chord: the relative difference of an attenuating sphere's absorbed light (the darkness summed over the frame)
#: against route 1's. A grazing ray's chord is off by up to a sample's spacing, a share of the frame's darkness.
CHORD_LINE = 1e-3
#: The located error's line: one corner sub-sample's share of a pixel, 1/16 of the 4 x 4 the corner choice takes, the
#: finest a pixel's share is resolved by there. The first line, 0.04 (twice the worst the corner sub-samples left on the
#: screw's located pixels, run 43607018230), was falsified by the first rate run on the coil: 7 of 30 clean draws over it,
#: to 0.0547 (the truth converged: 0.004 against four times its samples). This one is derived from the sub-samples'
#: resolution, not fitted between the arms, though chosen after draws 0 to 59 were seen at 128 pixels (the clean worst
#: 0.0547, the single edge's least 0.0853); it is judged by the rate run that follows, at 128 and at 256 pixels, the second
#: seen at no draw (the record is in ``predictions_camera_pr5a.json``).
CORNER_LINE = 1.0 / 16.0

LENS, POSE = across(128, field=3e-3)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
COIL = {"radius": 0.4e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.08e-3}


def subject(kind: str, settings: dict, route: str | None = None, **written) -> Subject:
    coverage_of = "box_edge" if kind == "sphere" else ""
    return Subject("body", kind, settings, appearance=written.pop("appearance", Appearance.opaque()), moving=True,
                   coverage=coverage_of, route=route, **written)


def frame(body: Subject, position, orientation=IDENTITY, view=(LENS, POSE)) -> np.ndarray:
    """The frame's transmission, in the run's dtype (the ``dtype`` or ``float64`` fixture sets it)."""
    found = render([body], view[0], view[1], {"body": (jnp.asarray(position), jnp.asarray(orientation))})
    return np.asarray(found.transmission, dtype=np.float64)


def motion(body: Subject, position) -> np.ndarray:
    """The darkness's total and centroid (u, v), and their derivatives in the body's x and z: ``(3, 3)``."""
    v, u = np.indices((LENS.height, LENS.width))

    def aggregates(at):
        dark = 1.0 - render([body], LENS, POSE, {"body": (at, jnp.asarray(IDENTITY))}).transmission
        total = jnp.sum(dark)
        return jnp.stack([total, jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    at = jnp.asarray(position)
    jacobian = jax.jacfwd(aggregates)(at)
    return np.asarray(jnp.stack([aggregates(at), jacobian[:, 0], jacobian[:, 2]]), dtype=np.float64)


#: The swap control's offset: a tenth of the pixel's footprint at the bodies' distance (3 mm over 128 pixels), in metres.
OFFSET = 0.1 * 3e-3 / 128


def offset_distance(found_evaluate):
    """The plant: the first-order distance wrong by a constant, its gradient right, so that the line moves by it."""
    def evaluate(sampled, p, x, y, z):
        return found_evaluate(sampled, p, x, y, z) + OFFSET
    return evaluate


@pytest.mark.parametrize(("kind", "settings"), [
    ("sphere", {"radius": 0.5e-3}),
    ("capsule", {"radius": 0.25e-3, "height": 0.8e-3, "axis": (1.0, 0.3, 0.4)})])
def test_g18_route_two_matches_route_one_on_the_sphere_and_the_capsule(kind, settings, dtype, monkeypatch):
    if planted("sampled-offset"):
        monkeypatch.setattr(sampled_route, "evaluate", offset_distance(sampled_route.evaluate))
    position = (0.11e-3, 0.0, -0.07e-3)
    one, two = subject(kind, settings), subject(kind, settings, route="sampled")
    pixel = float(np.max(np.abs(frame(two, position) - frame(one, position))))
    a, b = motion(one, position), motion(two, position)
    moved = max(float(np.linalg.norm(b[i] - a[i]) / np.linalg.norm(a[i])) for i in (1, 2))
    print(f"\n{kind} {dtype}: largest per-pixel difference {pixel:.3g}, motion {moved:.3g}")
    assert pixel <= SWAP_PIXEL_LINE
    assert moved <= SWAP_MOTION_LINE


def test_the_chord_of_an_attenuating_sphere_matches_route_one(dtype):
    see_through = Appearance.attenuating(2000.0)
    position = (0.05e-3, 0.0, 0.02e-3)
    one = subject("sphere", {"radius": 0.5e-3}, appearance=see_through)
    two = subject("sphere", {"radius": 0.5e-3}, route="sampled", appearance=see_through)
    if planted("long-chord"):
        found_chord = sampled_route.chord
        sampled_route_chord = lambda *a: 1.1 * found_chord(*a)  # noqa: E731 - the plant
        pytest.MonkeyPatch().setattr(sampled_route, "chord", sampled_route_chord)
    a, b = 1.0 - frame(one, position), 1.0 - frame(two, position)
    found = abs(float(b.sum() - a.sum())) / float(a.sum())
    print(f"\nthe absorbed light's relative difference {found:.3g}; largest per pixel {np.max(np.abs(b - a)):.3g}")
    assert found <= CHORD_LINE


# -- the located error's fix ---------------------------------------------------------------------------------------------

def drawn_pose(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A coil near the image's centre, rolled about its axis, turned within 20 degrees in the image and 10 in depth."""
    rng = np.random.default_rng(2000 + seed)
    position = np.array([rng.uniform(-0.1e-3, 0.1e-3), 0.0, rng.uniform(-0.1e-3, 0.1e-3)])
    turns = [(0, rng.uniform(0.0, 2 * math.pi)), (1, math.radians(rng.uniform(-20.0, 20.0))),
             (2, math.radians(rng.uniform(-10.0, 10.0)))]
    q = np.array(IDENTITY)
    for axis, angle in turns:
        about = np.zeros(4)
        about[0], about[1 + axis] = math.cos(angle / 2), math.sin(angle / 2)
        q = np.asarray(quaternion.multiply(jnp.asarray(about), jnp.asarray(q)))
    return position, q / np.linalg.norm(q)


def two_branch_pixels(body: Subject, position, orientation, view=(LENS, POSE)) -> np.ndarray:
    """The pixels both of whose branches' lines cross them: two near-zero minima along the ray."""
    rays = render_module.ray_table(view[0], view[1])
    rotation, origin = render_module._local(body, jnp.asarray(position), jnp.asarray(orientation), jnp.float64)
    p = render_module._parameters(body, jnp.float64)
    o = render_module._into(rotation, tuple(rays.origin[i] - origin[i] for i in range(3)))
    d = render_module._into(rotation, rays.direction)
    du, dv = render_module._into(rotation, rays.along_u), render_module._into(rotation, rays.along_v)
    marched = body.placed.sampled
    held = sampled_route.march(marched, p, o, d, du)
    lines = render_module._lines(body, p, o, d, du, dv, sampled_route.form(marched, held))
    near = coverage.near(lines[0]) & coverage.near(lines[1])
    return np.nonzero(np.asarray(near).reshape(-1))[0]


def single_edge(monkeypatch) -> None:
    """The plant: the second branch dropped, so no pixel is a corner and each takes one straight edge."""
    def pieces(sampled, held):
        def made(p, o, d):
            first = sampled_route.evaluate(sampled, p, *(o[i] + held.ta * d[i] for i in range(3)))
            return first, first + sampled.enclosing
        return made
    monkeypatch.setattr(sampled_route, "pieces", pieces)


def corner_errors(seed: int, monkeypatch, width: int = 128) -> tuple[np.ndarray, int]:
    view = across(width, field=3e-3)
    position, orientation = drawn_pose(seed)
    body = subject("helix", COIL)
    pixels = two_branch_pixels(body, position, orientation, view)
    if planted("single-edge"):
        single_edge(monkeypatch)
    found = 1.0 - frame(body, position, orientation, view).reshape(-1)[pixels]
    truth = sampled_truth.truth(body, view[0], view[1], position, orientation, pixels, side=32, signs=512)
    return np.abs(found - truth), len(pixels)


@pytest.mark.parametrize("width", [128, 256])
def test_the_two_branch_pixels_are_within_the_corner_line_of_the_truth(width, monkeypatch, float64):
    errors, count = corner_errors(SEED, monkeypatch, width)
    print(f"\nwidth {width}, seed {SEED}: {count} two-branch pixels, largest error {errors.max():.4f}, mean {errors.mean():.4f}")
    assert count > 0, "the drawn pose shows no crossing of the coil's turns: the gate is silent"
    assert float(errors.max()) <= CORNER_LINE


# -- the window ----------------------------------------------------------------------------------------------------------

def test_g21_a_windowed_coil_equals_its_whole_frame_and_does_not_overflow(dtype):
    whole = subject("helix", COIL)
    size = 0.3 if planted("small-window") else 1.0
    windowed = subject("helix", COIL, window=max(1, int(size * 50)))
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        position, orientation = drawn_pose(int(rng.integers(0, 2 ** 31)))
        held = {"body": (jnp.asarray(position), jnp.asarray(orientation))}
        a = render([whole], LENS, POSE, held)
        b = render([windowed], LENS, POSE, held)
        assert not bool(b.flags["window_overflow:body"])
        np.testing.assert_array_equal(np.asarray(b.transmission), np.asarray(a.transmission))


# -- the held march's derivative -----------------------------------------------------------------------------------------

def test_route_twos_derivative_on_the_sphere_is_route_ones(dtype):
    """Where route 1 holds the exact derivative (the sphere's closed form), route 2's equals it: the darkness's and its
    centroid's derivatives in x and z."""
    position = np.array([0.03e-3, 0.0, 0.02e-3])
    one = motion(subject("sphere", {"radius": 0.5e-3}), position)
    two = motion(subject("sphere", {"radius": 0.5e-3}, route="sampled"), position)
    apart = max(float(np.linalg.norm(two[i] - one[i]) / np.linalg.norm(one[i])) for i in (1, 2))
    print(f"\nroute 2's derivative against route 1's: {apart:.3g}")
    assert apart <= SWAP_MOTION_LINE


def test_a_held_branchs_derivative_converges_on_the_least_distances_along_the_ray(float64):
    """The envelope's rule the held march rests on: a branch's first-order distance at its held point has, in the
    body's position, the derivative of the least first-order distance along the ray (found densely, 2049 samples
    within two spacings of the held point refined by the parabola, and differenced at 1e-9 m), but for the parabola's
    error in the held point, second order in the samples' spacing: on the coil's rays near its outline, the 99th
    percentile of the difference is under 5 % of the derivative's largest, and twice the samples cut it at least by
    2.5 times (theory: 4)."""
    import dataclasses

    body = subject("helix", COIL)
    position, orientation = drawn_pose(3)
    rays = render_module.ray_table(LENS, POSE)
    p = render_module._parameters(body, jnp.float64)
    at = jnp.asarray(position)
    footprint = float(-POSE.position[1]) / LENS.fx

    def frame_rays(where):
        rotation, origin = render_module._local(body, where, jnp.asarray(orientation), jnp.float64)
        o = render_module._into(rotation, tuple(rays.origin[i] - origin[i] for i in range(3)))
        d = render_module._into(rotation, rays.direction)
        return o, d, render_module._into(rotation, rays.along_u)

    def percentile(marched) -> tuple[float, float]:
        def first(where):
            o, d, du = frame_rays(where)
            held = sampled_route.march(marched, p, o, d, du)
            return sampled_route.pieces(marched, held)(p, o, d)[0].reshape(-1), held

        value, held = first(at)
        near = np.nonzero(np.abs(np.asarray(value)) < 2.0 * footprint)[0]
        assert len(near) > 50
        centre, spacing = np.asarray(held.ta).reshape(-1)[near], np.asarray(held.dt).reshape(-1)[near]
        t = jnp.asarray(centre[:, None] + 2.0 * spacing[:, None] * np.linspace(-1.0, 1.0, 2049)[None, :])

        def least(where):
            o, d, _ = frame_rays(where)
            o = tuple(jnp.broadcast_to(c, d[0].shape).reshape(-1)[near][:, None] for c in o)
            d = tuple(c.reshape(-1)[near][:, None] for c in d)
            g = sampled_route.evaluate(marched, p, *(o[i] + t * d[i] for i in range(3)))
            k = jnp.clip(jnp.argmin(g, axis=1), 1, g.shape[1] - 2)[:, None]
            before, low, after = (jnp.take_along_axis(g, k + j, 1)[:, 0] for j in (-1, 0, 1))
            return low - (before - after) ** 2 / (8.0 * (before - 2.0 * low + after))

        found, largest = [], 0.0
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = 1e-9
            tangent = np.asarray(jax.jvp(lambda a: first(a)[0], (at,), (jnp.asarray(step / 1e-9),))[1])[near]
            difference = (np.asarray(least(at + step)) - np.asarray(least(at - step))) / 2e-9
            found.append(np.abs(tangent - difference))
            largest = max(largest, float(np.max(np.abs(tangent))))
        return float(np.quantile(np.concatenate(found), 0.99)), largest

    marched = body.placed.sampled
    coarse, largest = percentile(marched)
    fine, _ = percentile(dataclasses.replace(marched, samples=2 * marched.samples - 1))
    print(f"\n99th percentile of the difference: {coarse:.3g} at {marched.samples} samples, {fine:.3g} at "
          f"{2 * marched.samples - 1}, of derivatives up to {largest:.3g}")
    assert coarse <= 0.05 * largest
    assert fine <= coarse / 2.5
