"""The capped cylinder's silhouette, pixel by pixel: rods turned towards the line of sight (their caps opening from edge-on),
rolled in the image and moved by a fraction of a pixel, against the exact image (the hull of the rims' images,
``hull_oracle``), in both dtypes, through a pinhole without distortion; the review's turns (0 to 20 degrees) and drawn
ones (0 to 30, any roll and offset).

**The line**, per pixel: ``1 / s^2`` with ``s x s`` sub-samples (6.25 % at the default 4). Once a corner or a cap's
outline is in view every pixel takes its sub-samples, each its own straight line from the pieces' composed distance in
pixels; a sub-pixel's line is exact where the outline is straight across it and off by the outline's curvature
otherwise (``0.118 h^2 / rho`` for a sub-pixel of side ``h``, small here), and where the outline turns within a
sub-pixel (a cap's ellipse's end, where a side meets it, its radius ``b^2 / a`` a fraction of a sub-pixel) the turn
lies in at most two of a pixel's sub-pixels along the outline, each missing at most half of itself: ``h^2`` in all.
**Thin caps:** a cap whose ellipse is narrower than two sub-pixels is not resolved (its first-order line is a sub-pixel
off inside it and past its ends); the view raises ``thin_cap``, and the pixels about that cap are judged by the flag,
not the line. The flag must be raised exactly where a cap is that thin (its width computed here from the rim's
geometry), so a miss above the line near an unflagged cap fails.

The review found the cylinder's composed edge taking the wrong piece (a least of raw values whose gradients differ
thirteenfold at a cap seen near edge-on): 13.8 to 29 % at turns of 0 to 10 degrees, unflagged at 6 and 10. **The
control** (``CAMERA_PLANT=raw-composition``) renders with that code (the least of raw values, sub-samples only where two
pieces cross a pixel): the review's turns fail every run."""

from __future__ import annotations

import math

import hull_oracle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.sensing._coverage as coverage
import mime.sensing._forms as forms
import mime.sensing._render as render_module
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

WIDTH, HEIGHT, FOCAL, DISTANCE = 96, 64, 900.0, 30e-3
RADIUS, LENGTH = 0.31e-3, 2.07e-3
CENTRE = np.array((0.113e-3, 0.0, 0.071e-3))
SAMPLES = coverage.CORNER_SAMPLES
LINE = 1.0 / SAMPLES ** 2
REVIEWED = [0.0, 2.0, 4.0, 6.0, 10.0, 20.0]


def _reviewed_pieces(p, o, d):
    """The cylinder's pieces as the review found them: each cap the in-plane distance of the ray's crossing, the side
    and the span on the ray's whole line."""
    radius, half_length = p["radius"], p["length"] / 2.0
    _, q, _, axial = forms._side(radius, o, d)

    def cap(z):
        t = (z - o[2]) / forms._guard(d[2])
        x, y = o[0] + t * d[0], o[1] + t * d[1]
        return forms._safe_sqrt(x * x + y * y, (forms.GRAZING * radius) ** 2) - radius

    side = forms._safe_sqrt(q, (forms.GRAZING * radius) ** 2) - radius
    return cap(-half_length), cap(half_length), side, jnp.abs(axial) - half_length


def _raw_box_coverage(subject, p, o, d, du, dv, corner_samples):
    """The box edge as the review found it: the least of raw values, its line read through ``jvp``, sub-samples only
    where two pieces' lines cross the pixel."""
    form = subject.placed.form

    def edges(dx, dy, dz):
        pieces = _reviewed_pieces(p, o, (dx, dy, dz))
        return jnp.minimum(jnp.minimum(pieces[0], pieces[1]), jnp.maximum(pieces[2], pieces[3])), pieces

    (g, pieces), linear = jax.linearize(edges, *d)
    gu, pieces_u = linear(*du)
    gv, pieces_v = linear(*dv)
    single = coverage.box(g, gu, gv)
    if not form.corners or corner_samples <= 1:
        return single
    corner = sum(coverage.near(coverage.line(*piece)).astype(g.dtype)
                 for piece in zip(pieces, pieces_u, pieces_v, strict=True)) >= 2.0
    total = jnp.zeros_like(g)
    for su, sv in coverage.sub_offsets(corner_samples):
        moved = tuple(d[i] + su * du[i] + sv * dv[i] for i in range(3))
        scale = 1.0 / jnp.sqrt(sum(c * c for c in moved))
        h, sub_linear = jax.linearize(lambda x, y, z: edges(x, y, z)[0], *(c * scale for c in moved))
        total = total + coverage.box(h, sub_linear(*du), sub_linear(*dv), 1.0 / corner_samples)
    return jnp.where(corner, total / corner_samples ** 2, single)


def _axis(turn: float, roll: float) -> np.ndarray:
    """Along x, turned by ``turn`` degrees about z (towards the line of sight), then rolled by ``roll`` degrees about
    the line of sight (y)."""
    a, r = math.radians(turn), math.radians(roll)
    return np.array([math.cos(a) * math.cos(r), math.sin(a), math.cos(a) * math.sin(r)])


def judged(turn: float, roll: float, offset: tuple[float, float], monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("raw-composition"):
        monkeypatch.setattr(render_module, "_box_coverage", _raw_box_coverage)
    axis = _axis(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * DISTANCE / FOCAL
    camera = Pinhole(WIDTH, HEIGHT, FOCAL, FOCAL, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
    s = math.sqrt(0.5)
    pose = CameraPose(jnp.asarray([0.0, -DISTANCE, 0.0]), jnp.asarray([s, -s, 0.0, 0.0]))
    body = Subject("rod", "cylinder", {"radius": RADIUS, "length": LENGTH, "axis": tuple(axis)},
                   appearance=Appearance.opaque(), moving=True)
    frame = render([body], camera, pose, {"rod": (jnp.asarray(centre), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    covered = 1.0 - np.asarray(frame.transmission, dtype=np.float64)
    rims = hull_oracle.cylinder_outline(centre, axis, RADIUS, LENGTH)
    exact = hull_oracle.shares(rims, FOCAL, DISTANCE, WIDTH, HEIGHT)
    # The pixels about each thin cap (its rim's image's box and two pixels about it) are judged by the flag.
    judged_here = np.ones_like(exact, dtype=bool)
    thin = False
    for end, rim in zip((-LENGTH / 2, LENGTH / 2), rims, strict=True):
        if hull_oracle.minor_axis(centre + end * axis, axis, RADIUS, FOCAL, DISTANCE) < 2.0 / SAMPLES:
            thin = True
            image = hull_oracle.project(rim, FOCAL, DISTANCE, WIDTH, HEIGHT)
            low, high = np.floor(image.min(axis=0)).astype(int) - 2, np.ceil(image.max(axis=0)).astype(int) + 2
            judged_here[max(0, low[1]):high[1] + 1, max(0, low[0]):high[0] + 1] = False
    worst = float(np.max(np.abs(covered - exact)[judged_here]))
    floor = 2 * float(jnp.finfo(frame.transmission.dtype).eps) * FOCAL
    assert worst <= LINE + floor, (worst, LINE)
    assert bool(frame.flags["thin_cap:rod"]) == thin


@pytest.mark.parametrize("turn", REVIEWED)
def test_the_reviews_turned_rods(dtype, turn, monkeypatch):
    judged(turn, 0.0, (0.0, 0.0), monkeypatch)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(turn=st.floats(0.0, 30.0), roll=st.floats(0.0, 90.0), du=st.floats(0.0, 1.0), dv=st.floats(0.0, 1.0))
def test_turned_rods_drawn(dtype, turn, roll, du, dv, monkeypatch):
    judged(turn, roll, (du, dv), monkeypatch)
