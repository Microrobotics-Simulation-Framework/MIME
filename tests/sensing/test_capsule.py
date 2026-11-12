"""The capsule's closed form: its silhouette, pixel by pixel, against the exact image of a capsule (the hull of its two
end balls' silhouettes, ``hull_oracle``), and its chord along rays well inside it against a bisection of the ray's
distance from the segment written here, at the review's tilts and drawn tilts, turns and sub-pixel offsets, in both
dtypes, through a pinhole without distortion.

**The line**, per pixel: the capsule's edge function is the ray's distance from its segment less the radius, a distance
in the image (times the depth over the focal length), so the box edge's straight line is wrong only by the outline's
curvature across the pixel. For an exact distance and an outline of curvature radius ``rho`` pixels, the line is the
tangent at the nearest point and the outline leaves it by ``s^2 / (2 rho)`` at ``s`` along it; across a pixel's square
(a chord at most ``sqrt 2``) the area between them is at most ``(sqrt 2 / 2)^3 / (3 rho)``, ``0.118 / rho``. The
first-order line of a distance read at the pixel's centre, not at the nearest point, turns the line by up to
``|d| / rho`` (``|d|`` up to ``sqrt 2 / 2``), which moves at most half a pixel's chord times that turn's tangent:
another ``0.354 / rho`` at most. The line is ``0.48 / rho``, ``rho`` the least curvature radius of the capsule's outline
(its end balls' silhouette ellipses at their ends, ``b^2 / a``), plus the float32 floor (a pixel's footprint rounds
by ``8 eps`` of the depth over the focal length: ``8 eps f`` pixels of coverage).

**The chord**, along rays at least 2 px inside the outline: within ``16 eps`` of the depth (the line's parameter's
rounding at the camera's distance) plus the oracle's bisection floor.

**The control** (``CAMERA_PLANT=capsule-half``): the capsule's segment clamped at half its length in its edge function
(the review's mutant, imaged at about 60 % of its area), caught every run."""

from __future__ import annotations

import math

import hull_oracle
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.sensing._forms as forms
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, ray_table, render

WIDTH, HEIGHT, FOCAL, DISTANCE = 96, 64, 900.0, 30e-3
RADIUS, LENGTH = 0.31e-3, 2.07e-3
CENTRE = np.array((0.113e-3, 0.0, 0.071e-3))
PINHOLE = np.array([0.0, -DISTANCE, 0.0])
#: The review's tilts (degrees about z, towards the line of sight).
REVIEWED = [0.0, 6.0, 30.0, 60.0, 85.0]


def _axis(turn: float, roll: float) -> np.ndarray:
    """The capsule's axis: along x, turned by ``turn`` degrees about z (towards the line of sight), then rolled by
    ``roll`` degrees about the line of sight (y)."""
    a, r = math.radians(turn), math.radians(roll)
    axis = np.array([math.cos(a), math.sin(a), 0.0])
    return np.array([axis[0] * math.cos(r) - axis[2] * math.sin(r), axis[1], axis[0] * math.sin(r) + axis[2] * math.cos(r)])


def _camera() -> tuple[Pinhole, CameraPose]:
    s = math.sqrt(0.5)
    return (Pinhole(WIDTH, HEIGHT, FOCAL, FOCAL, (WIDTH - 1) / 2, (HEIGHT - 1) / 2),
            CameraPose(jnp.asarray(PINHOLE), jnp.asarray([s, -s, 0.0, 0.0])))


def _clamped_at_half(p, o, d):
    """The review's mutant: the segment's foot clamped at half its length."""
    return forms.CAPSULE.pieces({"radius": p["radius"], "height": p["height"] / 2}, o, d)


def _least_curvature_radius(centre: np.ndarray, axis: np.ndarray) -> float:
    """The least curvature radius (pixels) of the end balls' silhouette ellipses, ``b^2 / a``: the semi-axes of the
    cone's section, ``f`` times ``tan`` of the cone's half-angle over the axis's cosine (along the radial direction)
    and the half-angle's tangent (across), from the optical axis."""
    least = math.inf
    for end in (-LENGTH / 2, LENGTH / 2):
        ball = centre + end * axis
        sight = ball - PINHOLE
        span = float(np.linalg.norm(sight))
        cosine = sight[1] / span
        half = math.asin(RADIUS / span)
        across = FOCAL * math.tan(half) / cosine
        along = FOCAL * math.sin(half) * math.cos(half) / (cosine ** 2 - math.sin(half) ** 2)
        a, b = max(across, along), min(across, along)
        least = min(least, b * b / a)
    return least


def coverage_error(turn: float, roll: float, offset: tuple[float, float], monkeypatch: pytest.MonkeyPatch,
                   dtype: str) -> tuple[float, float]:
    """The worst per-pixel coverage error against the exact image, and the line."""
    if planted("capsule-half"):
        monkeypatch.setattr(forms.CAPSULE, "pieces", _clamped_at_half)
    axis = _axis(turn, roll)
    centre = CENTRE + np.array([offset[0], 0.0, -offset[1]]) * DISTANCE / FOCAL
    camera, pose = _camera()
    body = Subject("capsule", "capsule", {"radius": RADIUS, "height": LENGTH, "axis": tuple(axis)},
                   appearance=Appearance.opaque(), moving=True)
    frame = render([body], camera, pose, {"capsule": (jnp.asarray(centre), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    covered = 1.0 - np.asarray(frame.transmission, dtype=np.float64)
    exact = hull_oracle.shares(hull_oracle.capsule_outline(centre, axis, RADIUS, LENGTH, PINHOLE), FOCAL, DISTANCE,
                               WIDTH, HEIGHT)
    floor = 8 * float(np.finfo(dtype).eps) * FOCAL
    return float(np.max(np.abs(covered - exact))), 0.48 / _least_curvature_radius(centre, axis) + floor


@pytest.mark.parametrize("turn", REVIEWED)
def test_the_capsules_silhouette_at_the_reviews_tilts(dtype, turn, monkeypatch):
    worst, line = coverage_error(turn, 0.0, (0.0, 0.0), monkeypatch, dtype)
    assert worst <= line, (worst, line)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(turn=st.floats(0.0, 89.0), roll=st.floats(0.0, 90.0), du=st.floats(0.0, 1.0), dv=st.floats(0.0, 1.0))
def test_the_capsules_silhouette_drawn(dtype, turn, roll, du, dv, monkeypatch):
    worst, line = coverage_error(turn, roll, (du, dv), monkeypatch, dtype)
    assert worst <= line, (worst, line)


def _chord_oracle(origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The chord of each ray inside the capsule: the ray's distance from the segment is convex along it, so its least
    is found by ternary search and each crossing of the radius by bisection on either side (200 steps each)."""

    def distance(t: np.ndarray) -> np.ndarray:
        point = origin + t[..., None] * direction - centre
        s = np.clip(point @ axis, -LENGTH / 2, LENGTH / 2)
        return np.linalg.norm(point - s[..., None] * axis, axis=-1)

    low, high = np.zeros(direction.shape[:-1]), np.full(direction.shape[:-1], 2 * DISTANCE)
    for _ in range(200):
        a, b = low + (high - low) / 3, high - (high - low) / 3
        nearer = distance(a) < distance(b)
        low, high = np.where(nearer, low, a), np.where(nearer, b, high)
    least = (low + high) / 2
    ends = []
    for inside_end, outside_end in ((least, np.zeros_like(least)), (least, np.full_like(least, 2 * DISTANCE))):
        a, b = inside_end.copy(), outside_end.copy()
        for _ in range(200):
            middle = (a + b) / 2
            inside = distance(middle) < RADIUS
            a, b = np.where(inside, middle, a), np.where(inside, b, middle)
        ends.append((a + b) / 2)
    return np.where(distance(least) < RADIUS, ends[1] - ends[0], 0.0)


@pytest.mark.parametrize("turn", REVIEWED)
def test_the_capsules_chord(dtype, turn):
    axis = _axis(turn, 0.0)
    camera, pose = _camera()
    body = Subject("capsule", "capsule", {"radius": RADIUS, "height": LENGTH, "axis": tuple(axis)},
                   appearance=Appearance.attenuating(1000.0), moving=True)
    frame = render([body], camera, pose, {"capsule": (jnp.asarray(CENTRE), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    found = -np.log(np.asarray(frame.transmission, dtype=np.float64)) / 1000.0
    rays = ray_table(camera, pose)
    direction = np.stack([np.asarray(c, dtype=np.float64) for c in rays.direction], axis=-1)
    expected = _chord_oracle(PINHOLE, direction, CENTRE, axis)
    exact = hull_oracle.shares(hull_oracle.capsule_outline(CENTRE, axis, RADIUS, LENGTH, PINHOLE), FOCAL, DISTANCE,
                               WIDTH, HEIGHT)
    # Rays at least 2 px inside the outline: every pixel within two of them wholly covered.
    deep = exact >= 1.0
    for _ in range(2):
        deep = deep & np.roll(deep, 1, 0) & np.roll(deep, -1, 0) & np.roll(deep, 1, 1) & np.roll(deep, -1, 1)
    assert deep.sum() > 50
    tolerance = 16 * float(np.finfo(dtype).eps) * DISTANCE + 1e-12
    assert float(np.max(np.abs(found - expected)[deep])) <= tolerance
