"""Route 2, the sampled route (``mime.sensing._sampled``), against route 1 and the sign-only truth.

- **G18, the swap test.** A sphere (its box edge) and a capsule imaged by route 2 (``route="sampled"``) against route 1,
  per pixel and in the silhouette's motion, in float64 and float32. Their sign functions are their exact distances, so
  route 2 differs from route 1 only by where the march holds its point (the parabola's error, read at the second order).
  **Control** (``CAMERA_PLANT=sampled-offset``): the march's first-order distance offset by a tenth of a pixel fails it,
  every run. (Its first plant, the distance 10 % short, cancelled: the line is the distance over its gradient, and both
  were scaled; 0 of 30 runs caught it, so it said nothing.)
- **The chord**: an attenuating sphere's transmission by route 2's chord (its samples' sign changes, interpolated)
  against route 1's closed-form chord.
- **The located error's fix** (camera PR 5's first step and its fix round): on a coil, whose turns meet in concave
  notches in its image, the pixels where two branches of its outline cross (two near-zero minima along the ray) take
  route 1's corner sub-samples, each the exact share of the union of the two branches' half-planes, each branch read at
  its sub-ray's own least; against the sign-only truth (``sampled_truth``, 64 x 64 samples a pixel, in float64), every
  such pixel is within ``corner_line`` and the single edge (the second branch dropped) is not. The line is derived from
  the scheme's own error terms (``scripts/camera_route2/corner_line.py``), before its confirmation's draws ran
  (``predictions_camera_pr5a_fix.json``): each branch's curvature in a sub-cell, the truth's floor, float32's rounding.
  Two coils, three widths, both dtypes. **Control** (``CAMERA_PLANT=single-edge``, a pose drawn from
  ``DETECTION_SEED``): the single edge fails it, every run. **The derivation's premise** (``unread_leasts``, at four
  times the resolution of the scheme's own verification, across the whole march): on every such pixel's sub-rays each
  least that reaches its sub-cell is read by a piece; its own control, on every draw: the same count with the second
  branch's reading lost is not zero. **The run flag** (``corner_unverified``, ``corner_bound``): its count is recorded
  on every clean draw (a flagged draw within the line is a false alarm), and its control, on every draw
  (``unread_second``: the second branch lost before the verification, which then bounds what it finds without reading
  it), flags pixels, each within the line and its bound. A second plant (``CAMERA_PLANT=misread``): the second branch
  read a tenth of a pixel out (or ``CAMERA_PLANT_SIZE``), the tail round's defect class. ``CORNER_RECORD`` names a file
  each run appends its numbers to; ``CORNER_TRUTH_CACHE`` a folder that keeps each draw's truth across a probe's
  passes.
- **The motion's derivative across a pixel**: a coil's darkness centroid's forward-mode derivative, at 16 positions over
  one pixel of motion, is within ``RIPPLE_LINE`` of the central difference over that pixel. **Control**
  (``CAMERA_PLANT=lesser``): the PR's first composition (one straight edge a cell, the least of the two branches'
  lines, each read at its pixel's held point), whose image ripples with the sub-samples' period, fails it, every run.
- **An ellipsoid** (no closed form) by route 2: its image's area against the projected ellipse's, in both dtypes.
- **G21, the window**: a moving helix rendered in a window equals its whole-frame render where nothing overflows, and
  its overflow flag stays down over poses within the declared motion; **control** (``CAMERA_PLANT=small-window``): the
  window at 0.3 of its size raises the flag and its image differs, every run.
- **The held march's derivative** (the switch scan's ``cleared_by`` for its ``stop_gradient``): route 2's forward-mode
  derivative of the image's darkness and its centroid equals the central difference of the same renderer."""

from __future__ import annotations

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sampled_truth
from camera_cases import PLANT, PLANT_SIZE, SEED, across, planted

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
#: The located error's line (``corner_line``): the bound ``scripts/camera_route2/corner_line.py`` derives for the fixed
#: scheme, written before its confirmation's draws ran, never fitted between the arms (the first lines, 0.04 and 1/16,
#: are withdrawn: the first was fitted on the screw, the second a sub-sample's area placed after its draws were seen).
#: Each coil's least notch branch radius over the drawn poses (m), from that script's geometry.
NOTCH_RADIUS = {"coil": 24.21e-6, "coil2": 62.17e-6}
#: The cell term's coefficient: two branches, each over at most 9 of a pixel's 4 x 4 sub-cells, each off its tangent by
#: at most 0.1263 s^3 / rho (s a quarter of a pixel), in a pixel's share times the radius in pixels.
CELL_COEFFICIENT = 0.035525
#: The refinement's residual and the truth's sign samples' (each under 1e-4), and the truth's floor at 64 x 64 samples a
#: pixel (Hoeffding, at 4e-8 a pixel).
REFINED, TRUTH_SIGNS, TRUTH_FLOOR = 1e-4, 1e-4, 0.0123
TRUTH_SIDE = 64
#: The camera's distance from the coil (m), which float32's rounding of a sub-ray's point scales with.
DISTANCE = 0.1


def corner_line(coil: str, width: int, dtype: str) -> float:
    """The derived line on a two-branch pixel's share, for ``coil`` seen ``width`` pixels across a 3 mm field."""
    pixel = 3e-3 / width
    rounding = 2.0 * math.sqrt(2.0) * 5.0 * 2.0 ** -24 * DISTANCE / pixel if dtype == "float32" else 0.0
    return CELL_COEFFICIENT * pixel / NOTCH_RADIUS[coil] + REFINED + TRUTH_SIGNS + TRUTH_FLOOR + rounding


#: The motion's derivative across a pixel: the design's G20 line, 1 % of the motion.
RIPPLE_LINE = 0.01


LENS, POSE = across(128, field=3e-3)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
COIL = {"radius": 0.4e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.08e-3}
#: The confirmation's second coil: a shorter pitch and another number of turns, its notches sharper (to 21.8 degrees).
COILS = {"coil": COIL, "coil2": {**COIL, "pitch": 0.52e-3, "turns": 3.0}}


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


#: The dense profile's steps a march spacing, across each sub-ray's whole march: four times the scheme's own
#: verification's (``_sampled.VERIFY_STEPS``), so that the guard is not the scheme's own reading of the same samples.
DENSE_STEPS = 4 * sampled_route.VERIFY_STEPS


def judged_sub_rays(body: Subject, position, orientation, pixels: np.ndarray, view=(LENS, POSE),
                    verify: bool = True) -> tuple:
    """The pixels' corner sub-rays, ``(sub-offset, pixel)``, in float64: the march's held result (told the pixels are
    corners where ``verify``, so that the scheme verifies its reading there), the parameters, the rays and the ray
    table's derivatives, and the scheme's two read points and its bound on each (``_sampled._leasts``)."""
    lens = view[0]
    rays = render_module.ray_table(lens, view[1], u=jnp.asarray(pixels % lens.width, jnp.float64),
                                   v=jnp.asarray(pixels // lens.width, jnp.float64))
    rotation, origin = render_module._local(body, jnp.asarray(position), jnp.asarray(orientation), jnp.float64)
    p = render_module._parameters(body, jnp.float64)
    o = render_module._into(rotation, tuple(rays.origin[i] - origin[i] for i in range(3)))
    d = render_module._into(rotation, rays.direction)
    du, dv = render_module._into(rotation, rays.along_u), render_module._into(rotation, rays.along_v)
    marched = body.placed.sampled
    held = sampled_route.march(marched, p, o, d, du)
    # Every sub-ray of every pixel at once: (sub-offset, pixel).
    offsets = np.asarray(coverage.sub_offsets(coverage.CORNER_SAMPLES))
    su, sv = jnp.asarray(offsets[:, 0])[:, None], jnp.asarray(offsets[:, 1])[:, None]
    moved = tuple(d[i][None, :] + su * du[i][None, :] + sv * dv[i][None, :] for i in range(3))
    scale = 1.0 / jnp.sqrt(sum(c * c for c in moved))
    ray = tuple(c * scale for c in moved)
    o = tuple(jnp.broadcast_to(c, ray[0].shape) for c in o)
    du, dv = (tuple(jnp.broadcast_to(c, ray[0].shape) for c in along) for along in (du, dv))
    if verify:
        held = held._replace(corner=jnp.ones_like(held.has2), du=du, dv=dv)
    first, second, bound = sampled_route._leasts(marched, held, p, o, ray)
    return marched, held, p, o, ray, du, dv, (first, jnp.where(held.has2, second, first)), bound


def lost_reads(body: Subject, position, orientation, pixels: np.ndarray, view=(LENS, POSE)) -> np.ndarray:
    """Where the scheme's verification starts, ``(sub-offset, pixel)``: a sub-ray of a two-minimum ray whose two
    branches the search reads at one point (within half a spacing), before the verification."""
    *_, held, _, _, _, _, _, read, _ = judged_sub_rays(body, position, orientation, pixels, view, verify=False)
    return np.asarray(held.has2[None, :] & (jnp.abs(read[0] - read[1]) < 0.5 * held.dt[None, :]))


def unread_leasts(body: Subject, position, orientation, pixels: np.ndarray, view=(LENS, POSE),
                  verify: bool = True) -> int:
    """How many leasts on the pixels' corner sub-rays the scheme does not read: local minima of the sign function's
    first-order distance along the sub-ray (a dense profile across its whole march, ``DENSE_STEPS`` a spacing) whose
    half-plane takes some of the sub-cell, in a sub-cell the read branches' union does not already take whole, with no
    point the scheme's search reads on that sub-ray (``_sampled._leasts``: either piece's, after the merge rule and the
    verification, or before it where not ``verify``) within half a march spacing. The derivation's premise is that each
    branch crossing a sub-cell is read at its sub-ray's own least. Both tests are the coverage's own exact shares
    (``coverage.covered`` and ``union_covered``, a sub-cell a side), not a reach: a least whose half-plane misses the
    sub-cell (a farther turn's, or a second minimum that is a fold of the sign function's turn) changes nothing the
    cell reads, nor does any least in a sub-cell the read branches already cover whole (the union is the whole cell
    whatever else crosses it; the half-diagonal, sufficient and not necessary, counted one such at 2667). How far a
    read least's value lies from the truth (the quartic's, at an arc's end its guard's) is the line's refinement term,
    not this count's."""
    marched, held, p, o, ray, du, dv, read, _ = judged_sub_rays(body, position, orientation, pixels, view, verify)
    steps = jnp.arange(DENSE_STEPS * (marched.samples - 1) + 1, dtype=jnp.float64) / DENSE_STEPS
    t = held.t[None, :, :1] + held.dt[None, :, None] * steps[None, None, :]

    def profile(dx, dy, dz):
        return sampled_route.evaluate(marched, p, *(o[i][..., None] + t * (dx, dy, dz)[i][..., None]
                                                    for i in range(3)))

    values, along_u = jax.jvp(profile, ray, du)
    _, along_v = jax.jvp(profile, ray, dv)
    sub_cell = 1.0 / coverage.CORNER_SAMPLES
    reaches = coverage.covered(coverage.line(values, along_u, along_v), sub_cell) > 0.0
    minimum = jnp.concatenate([jnp.zeros_like(values[..., :1], dtype=bool),
                               (values[..., 1:-1] < values[..., :-2]) & (values[..., 1:-1] <= values[..., 2:]),
                               jnp.zeros_like(values[..., :1], dtype=bool)], axis=-1)
    half = 0.5 * held.dt[None, :, None]
    unread = (jnp.abs(read[0][..., None] - t) > half) & (jnp.abs(read[1][..., None] - t) > half)

    def read_line(at):
        def value(dx, dy, dz):
            return sampled_route.evaluate(marched, p, *(o[i] + at * (dx, dy, dz)[i] for i in range(3)))

        g, gu = jax.jvp(value, ray, du)
        _, gv = jax.jvp(value, ray, dv)
        return coverage.line(g, gu, gv)

    first, second = read_line(read[0]), read_line(read[1])
    taken = jnp.maximum(jnp.maximum(coverage.covered(first, sub_cell), coverage.covered(second, sub_cell)),
                        coverage.union_covered(first, second, sub_cell))
    open_cell = (taken < 1.0)[..., None]
    return int(np.sum(np.asarray(minimum & reaches & unread & open_cell)))


def _unverified_none(held, first) -> tuple:
    """A planted form's pieces' unverified number where its corner sub-samples read one (``Form.at_corners``): none."""
    return () if held.corner is None else (jnp.zeros_like(first),)


def single_edge(monkeypatch) -> None:
    """The plant: the second branch dropped, so no pixel is a corner and each takes one straight edge."""
    def pieces(sampled, held):
        def made(p, o, d):
            first = sampled_route.evaluate(sampled, p, *(o[i] + held.ta * d[i] for i in range(3)))
            return first, first + sampled.enclosing, *_unverified_none(held, first)
        return made
    monkeypatch.setattr(sampled_route, "pieces", pieces)


class enabled_x64:
    """Float64 within, the run's dtype restored after: the gate's oracle and its located pixels, whatever the run's."""

    def __enter__(self):
        self.held = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)

    def __exit__(self, *exc):
        jax.config.update("jax_enable_x64", self.held)


def lost_branch(found_leasts):
    """The guard's control: the scheme's reading with its second branch lost (the second piece read at the first's
    point after the verification, as the two halves of ``_sampled._leasts``' list mis-sliced would read it)."""
    def leasts(sampled, held, p, o, d):
        first, _, bound = found_leasts(sampled, held, p, o, d)
        return first, first, bound
    return leasts


def unread_second(found_gathered, found_verified):
    """The flag's control: the search's second branch lost before the verification (its point the first's, as if
    merged), and the verification left to find the leasts it reaches without reading them, so that each is in its
    bound: what a reading the verification cannot mend looks like."""
    def gathered(at, start, other, spacing, near, read):
        point, merged = found_gathered(at, start, other, spacing, near, read)
        half = point.shape[0] // 2
        return point.at[half:].set(point[:half]), merged.at[half:].set(True)

    def verified(*arguments, **written):
        return found_verified(*arguments, **written, reads=False)
    return gathered, verified


def seen_bound(body: Subject, position, orientation, view, dtype) -> tuple[np.ndarray, np.ndarray]:
    """The body's share of each pixel and each pixel's bound on its share where its reading is not verified (route 2's
    ``Seen.unverified``), through ``_render._box_coverage`` as ``_render._seen`` takes it, in ``dtype``."""
    rays = render_module.ray_table(view[0], view[1])
    rotation, origin = render_module._local(body, jnp.asarray(position, dtype), jnp.asarray(orientation, dtype), dtype)
    p = render_module._parameters(body, dtype)
    o = render_module._into(rotation, tuple(rays.origin[i] - origin[i] for i in range(3)))
    d = render_module._into(rotation, rays.direction)
    du, dv = render_module._into(rotation, rays.along_u), render_module._into(rotation, rays.along_v)
    marched = body.placed.sampled
    held = sampled_route.march(marched, p, o, d, du)
    bound: list = []
    share = render_module._box_coverage(body, p, o, d, du, dv, coverage.CORNER_SAMPLES,
                                        sampled_route.form(marched, held), bound)
    return np.asarray(share, np.float64).reshape(-1), np.asarray(bound[0], np.float64).reshape(-1)


#: The second plant's size (pixels): the tail round's defect, a branch's least on corner sub-rays misread so that the
#: branch is read too far out, measured at 0.10 to 0.17 px (seeds 2052, 2129 and 2049); the least measured, every ray.
#: ``CAMERA_PLANT_SIZE``, where set, takes its place: the ladder of sizes that finds the least caught on every draw.
MISREAD = PLANT_SIZE or 0.1


def misread(monkeypatch, width: int) -> None:
    """The plant: the second branch read ``MISREAD`` of a pixel's footprint at the coil farther out, on every ray."""
    found_pieces = sampled_route.pieces

    def pieces(sampled, held):
        made = found_pieces(sampled, held)

        def read_out(p, o, d):
            first, second, *unverified = made(p, o, d)
            return first, second + MISREAD * 3e-3 / width, *unverified
        return read_out
    monkeypatch.setattr(sampled_route, "pieces", pieces)


def held_truth(body: Subject, view, position, orientation, pixels, side: int, signs: int, key: str) -> np.ndarray:
    """The sign-only truth on the pixels, kept in ``CORNER_TRUTH_CACHE`` (a folder) where it is set, so that a
    probe's arms over one draw (clean, then each plant) make it once."""
    folder = os.environ.get("CORNER_TRUTH_CACHE")
    path = os.path.join(folder, f"{key}-{side}-{signs}.npz") if folder else None
    if path and os.path.exists(path):
        held = np.load(path)
        if np.array_equal(held["pixels"], pixels):
            return held["truth"]
    found = sampled_truth.truth(body, view[0], view[1], position, orientation, pixels, side=side, signs=signs)
    if path:
        os.makedirs(folder, exist_ok=True)
        np.savez(path, pixels=pixels, truth=found)
    return found


def corner_errors(seed: int, monkeypatch, width: int = 128, coil: str = "coil", side: int = TRUTH_SIDE,
                  signs: int = 512, guard: dict | None = None, dtype: str = "float64") -> tuple[np.ndarray, int]:
    """Each two-branch pixel's error against the sign-only truth, at the pose drawn from ``seed``: the pixels and the
    truth in float64, the frame in the run's dtype (``dtype``). ``guard``, where given, takes the count of the leasts
    on their sub-rays the scheme does not read (``unread_leasts``) and the same count with the guard's control planted
    (``lost_branch``); the run flag (``corner_unverified``, its pixels, and ``corner_bound``, the largest bound); and
    the flag's control (``unread_second``): the pixels it flags, its largest bound, and how far the most any pixel's
    error lies past the line and its own bound (at most zero: the bound holds)."""
    view = across(width, field=3e-3)
    position, orientation = drawn_pose(seed)
    with enabled_x64():
        body = subject("helix", COILS[coil])
        pixels = two_branch_pixels(body, position, orientation, view)
        truth = held_truth(body, view, position, orientation, pixels, side, signs, f"{coil}-{width}-{seed}")
        if guard is not None:
            guard["unread"] = unread_leasts(body, position, orientation, pixels, view)
            with pytest.MonkeyPatch.context() as control:
                control.setattr(sampled_route, "_leasts", lost_branch(sampled_route._leasts))
                guard["lost_branch"] = unread_leasts(body, position, orientation, pixels, view)
    if guard is not None:
        line = corner_line(coil, width, dtype)
        with pytest.MonkeyPatch.context() as control:
            gathered, verified = unread_second(sampled_route._gathered, sampled_route._verified)
            control.setattr(sampled_route, "_gathered", gathered)
            control.setattr(sampled_route, "_verified", verified)
            share, bound = seen_bound(subject("helix", COILS[coil]), position, orientation, view,
                                      jnp.float64 if dtype == "float64" else jnp.float32)
        flagged = bound[pixels] > 0.0
        guard["control_flagged"] = int(np.sum(flagged))
        guard["control_bound"] = float(np.max(bound[pixels]))
        guard["control_past"] = float(np.max(np.abs(share[pixels] - truth) - line - bound[pixels]))
        # How tight the bound is: each flagged pixel's error as a share of its bound (one, the bound reached).
        tight = np.abs(share[pixels] - truth)[flagged] / bound[pixels][flagged]
        guard["control_tightness"] = [float(np.median(tight)), float(np.max(tight))] if flagged.any() else None
    if planted("single-edge"):
        single_edge(monkeypatch)
    if planted("misread"):
        misread(monkeypatch, width)
    captured = render([subject("helix", COILS[coil])], view[0], view[1],
                      {"body": (jnp.asarray(position), jnp.asarray(orientation))})
    if guard is not None:
        guard["flagged"] = int(captured.flags["corner_unverified:body"])
        guard["flag_bound"] = float(captured.flags["corner_bound:body"])
    found = 1.0 - np.asarray(captured.transmission, dtype=np.float64).reshape(-1)[pixels]
    return np.abs(found - truth), len(pixels)


@pytest.mark.parametrize("coil", sorted(COILS))
@pytest.mark.parametrize("width", [128, 256, 512])
def test_the_two_branch_pixels_are_within_the_corner_line_of_the_truth(width, coil, monkeypatch, dtype):
    # The guard and the flag judge the scheme as it is: a planted run is judged by the line alone.
    guard: dict | None = None if PLANT else {}
    errors, count = corner_errors(SEED, monkeypatch, width, coil, guard=guard, dtype=dtype)
    line = corner_line(coil, width, dtype)
    largest = float(errors.max()) if count else float("nan")
    print(f"\n{coil}, width {width}, {dtype}, seed {SEED}: {count} two-branch pixels, largest error {largest:.4f} "
          f"(line {line:.4f}), mean {float(errors.mean()) if count else float('nan'):.4f}")
    if guard is not None:
        print(f"{guard['unread']} leasts unread ({guard['lost_branch']} with the second branch lost); "
              f"{guard['flagged']} pixels flagged corner_unverified (bound {guard['flag_bound']:.3g}); the flag's "
              f"control {guard['control_flagged']} (bound {guard['control_bound']:.3g}, past it "
              f"{guard['control_past']:.3g})")
    record = os.environ.get("CORNER_RECORD")
    if record:
        with open(record, "a") as handle:
            handle.write(json.dumps({"seed": SEED, "coil": coil, "width": width, "dtype": dtype, "plant": PLANT,
                                     "size": MISREAD if planted("misread") else None, "pixels": count,
                                     "largest": largest, "line": line, **(guard or {})}) + "\n")
    assert count > 0, "the drawn pose shows no crossing of the coil's turns: the gate is silent"
    assert largest <= line
    if guard is None:
        return
    assert guard["unread"] == 0, "a least that reaches a sub-cell is read by no piece: the line does not bound the pixel"
    assert guard["lost_branch"] > 0, "the guard does not see a lost branch at this draw: it cannot discriminate here"
    assert guard["control_flagged"] > 0, "the flag does not see a reading the verification cannot mend at this draw"
    assert guard["control_past"] <= 0.0, "a flagged pixel's error lies past the line and its bound: the bound fails"


# -- the motion's derivative across a pixel -----------------------------------------------------------------------------

def held_and_least(monkeypatch) -> None:
    """The plant: the PR's first composition, one straight edge a cell (the least of the two branches' lines), each
    branch read at its pixel's held point."""
    def cover(lines, size):
        return coverage.covered(coverage.lesser(lines[0], lines[1]), size)

    def pieces(sampled, held):
        def made(p, o, d):
            def at(t):
                return sampled_route.evaluate(sampled, p, *(o[i] + t * d[i] for i in range(3)))
            first = at(held.ta)
            return first, jnp.where(held.has2, at(held.tb), first + sampled.enclosing), *_unverified_none(held, first)
        return made

    monkeypatch.setattr(sampled_route, "_covered_union", cover)
    monkeypatch.setattr(sampled_route, "pieces", pieces)


def test_the_coils_motion_derivative_holds_across_a_pixel_of_motion(float64, monkeypatch):
    """At 16 positions over one pixel of motion along each image axis from a pose drawn near G20's straight one, the
    jvp of the coil's darkness centroid is within ``RIPPLE_LINE`` of the central difference over that pixel (relative
    to the motion's norm): the derivative of the image, not of a ripple in it."""
    if planted("lesser"):
        held_and_least(monkeypatch)
    rng = np.random.default_rng(SEED)
    start = np.array([rng.uniform(-0.05e-3, 0.05e-3), 0.0, rng.uniform(-0.05e-3, 0.05e-3)])
    pixel = float(-POSE.position[1]) / LENS.fx
    body = subject("helix", COIL)
    v, u = np.indices((LENS.height, LENS.width))

    def centroid(at):
        dark = 1.0 - render([body], LENS, POSE, {"body": (at, jnp.asarray(IDENTITY))}).transmission
        total = jnp.sum(dark)
        return jnp.stack([jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    found = jax.jit(centroid)
    tangent = jax.jit(lambda at, e: jax.jvp(centroid, (at,), (e,))[1])
    worst = 0.0
    for axis in (0, 2):
        e = np.zeros(3)
        e[axis] = 1.0
        reference = (np.asarray(found(jnp.asarray(start + pixel * e))) - np.asarray(found(jnp.asarray(start)))) / pixel
        for k in range(16):
            at = jnp.asarray(start + (k / 16.0) * pixel * e)
            derivative = np.asarray(tangent(at, jnp.asarray(e)))
            worst = max(worst, float(np.linalg.norm(derivative - reference) / np.linalg.norm(reference)))
    print(f"\nseed {SEED}: the jvp's largest relative deviation from the central difference over a pixel {worst:.4g}")
    assert worst <= RIPPLE_LINE


# -- an ellipsoid -----------------------------------------------------------------------------------------------------

#: The ellipsoid's area against the projected ellipse's: the box edge's curvature error over its outline, relative.
ELLIPSE_LINE = 2e-3


@pytest.mark.parametrize("axes", [(0.9e-3, 0.4e-3, 0.3e-3), (0.5e-3, 0.5e-3, 0.5e-3), (0.9e-3, 0.3e-3, 0.4e-3)])
def test_route_two_images_an_ellipsoid_with_its_projected_area(axes, dtype):
    """An ellipsoid (no closed form: route 2) seen along its second axis from 0.1 m covers, summed over the frame,
    the projected ellipse's area pi a c (the depth 0.1 m, the field 3 mm, its perspective under 1e-4), within
    ``ELLIPSE_LINE``; its first axis 1 % long is not (the control, every run)."""
    def area(semi_axes):
        body = Subject("body", "ellipsoid", {"semi_axes": list(semi_axes)}, appearance=Appearance.opaque(), moving=True)
        assert body.placed.sampled is not None
        found = render([body], LENS, POSE, {"body": (jnp.zeros(3), jnp.asarray(IDENTITY))})
        return float((1.0 - np.asarray(found.transmission, dtype=np.float64)).sum())

    pixel = 3e-3 / 128
    exact = math.pi * axes[0] * axes[2] / pixel ** 2
    found = area(axes) / exact - 1.0
    planted_found = area((1.01 * axes[0], axes[1], axes[2])) / exact - 1.0
    print(f"\n{axes} {dtype}: relative area {found:.3g}, the long axis's {planted_found:.3g}")
    assert abs(found) <= ELLIPSE_LINE
    assert abs(planted_found) > ELLIPSE_LINE


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
