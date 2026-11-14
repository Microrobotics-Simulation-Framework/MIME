"""Bodies behind the camera: a ray starts at the pinhole, so a body wholly behind the camera's plane covers nothing, and
one straddling it covers only what its part in front shows. Each closed form's edge and chord are taken on the ray
(``t >= 0``), and exact disc coverage, whose conic is an ellipse only for a ball wholly in front, gives way to the box
edge on the ray otherwise. In both dtypes:

- **wholly behind** (5 mm behind, in the field's cone mirrored): a sphere (exact disc coverage and the box edge), a
  capsule and a cylinder cover no pixel, with ``behind`` raised (the review found 10 807 to 19 138 px of 19 200 imaged);
- **straddling** the camera's plane in a wide field (f 60 px, the sphere's centre 0.2 mm ahead of the pinhole, R 0.3 mm,
  0.25 mm aside: the review found nothing imaged by the disc and 11 554 px by the box edge): each pixel against a
  sampled truth of the rays that meet the ball ahead of the pinhole (``SUB x SUB`` rays a pixel): exactly away from
  the outline, and on it within the sampling's quantum and the box edge's straight line on the outline's least
  curvature radius.

**The control** (``CAMERA_PLANT=whole-line``) images each body along the ray's whole line, as the review found it
(behind the camera too): the wholly-behind bodies cover pixels and the straddling ball's phantom half appears, every
run."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted

import mime.sensing._forms as forms
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

WIDTH, HEIGHT = 160, 120
SUB = 12


def _camera(focal: float) -> tuple[Pinhole, CameraPose]:
    s = math.sqrt(0.5)
    return (Pinhole(WIDTH, HEIGHT, focal, focal, (WIDTH - 1) / 2, (HEIGHT - 1) / 2),
            CameraPose(jnp.asarray([0.0, -0.03, 0.0]), jnp.asarray([s, -s, 0.0, 0.0])))


def _whole_line(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("whole-line"):
        monkeypatch.setattr(forms, "_on_ray", lambda ahead, on_line, at_start, nearest: on_line)
        monkeypatch.setattr(forms, "ahead", lambda interval: interval)


BODIES = {
    "sphere-disc": ("sphere", {"radius": 0.5e-3}, "disc"),
    "sphere-box": ("sphere", {"radius": 0.5e-3}, "box_edge"),
    "capsule": ("capsule", {"radius": 0.5e-3, "height": 1e-3, "axis": (1.0, 0.0, 0.0)}, "box_edge"),
    "cylinder": ("cylinder", {"radius": 0.5e-3, "length": 1e-3, "axis": (1.0, 0.0, 0.0)}, "box_edge"),
}


@pytest.mark.parametrize("body", list(BODIES))
def test_a_body_wholly_behind_covers_nothing(dtype, body, monkeypatch):
    _whole_line(monkeypatch)
    kind, settings, coverage = BODIES[body]
    camera, pose = _camera(600.0)
    subject = Subject("b", kind, settings, appearance=Appearance.opaque(), moving=True, coverage=coverage)
    frame = render([subject], camera, pose, {"b": (jnp.asarray((0.0, -0.035, 0.0)), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    assert bool(frame.flags["behind:b"])
    assert float(np.max(1.0 - np.asarray(frame.transmission))) == 0.0


#: The straddling ball: its centre 0.25 mm aside and 0.2 mm ahead of the pinhole, its radius 0.3 mm.
CENTRE, RADIUS = np.array([0.25e-3, -0.03 + 0.2e-3, 0.0]), 0.3e-3


def _sampled_truth(focal: float) -> np.ndarray:
    """Each pixel's share of ``SUB x SUB`` rays meeting the ball ahead of the pinhole."""
    offsets = (np.arange(SUB) + 0.5) / SUB - 0.5
    rows = (np.arange(HEIGHT)[:, None] + offsets[None, :]).ravel()
    columns = (np.arange(WIDTH)[:, None] + offsets[None, :]).ravel()
    y, x = np.meshgrid((rows - (HEIGHT - 1) / 2) / focal, (columns - (WIDTH - 1) / 2) / focal, indexing="ij")
    direction = np.stack([x, np.ones_like(x), -y], axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    w = np.array([0.0, -0.03, 0.0]) - CENTRE
    b = direction @ w
    reach = b * b - (w @ w - RADIUS ** 2)
    far = -b + np.sqrt(np.maximum(reach, 0.0))
    met = (reach > 0) & (far > 0)
    return met.reshape(HEIGHT, SUB, WIDTH, SUB).mean(axis=(1, 3))


def _least_curvature_radius(focal: float) -> float:
    """The least curvature radius, in pixels, of the straddling ball's outline within the image (two pixels about it):
    the image of the tangent cone's generators ahead of the pinhole, sampled finely, its curvature from the turn of its
    tangent along its length."""
    pinhole = np.array([0.0, -0.03, 0.0])
    sight = CENTRE - pinhole
    axis = sight / np.linalg.norm(sight)
    sine = RADIUS / np.linalg.norm(sight)
    first = np.cross(axis, [0.0, 0.0, 1.0])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    phi = np.linspace(0.0, 2.0 * np.pi, 200001)[:, None]
    generator = np.sqrt(1 - sine ** 2) * axis + sine * (np.cos(phi) * first + np.sin(phi) * second)
    ahead = generator[:, 1] > 1e-3
    u = focal * generator[:, 0] / generator[:, 1] + (WIDTH - 1) / 2
    v = -focal * generator[:, 2] / generator[:, 1] + (HEIGHT - 1) / 2
    points = np.stack([u, v], axis=-1)
    tangent = np.gradient(points, axis=0)
    turn = np.gradient(np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0])))
    radius = np.linalg.norm(tangent, axis=-1) / np.maximum(np.abs(turn), 1e-300)
    seen = ahead & (u > -2) & (u < WIDTH + 1) & (v > -2) & (v < HEIGHT + 1)
    return float(np.min(radius[seen]))


def test_a_ball_straddling_the_camera_shows_its_front_only(dtype, monkeypatch):
    _whole_line(monkeypatch)
    focal = 60.0
    camera, pose = _camera(focal)
    truth = _sampled_truth(focal)
    # Away from the outline (a pixel and its eight neighbours all met or all missed by the sampled rays), exactly;
    # on it within the sampling's quantum (a straight edge crossing SUB columns of rays: 1 / SUB) and the box edge's
    # straight line on the outline's least curvature radius (0.118 / rho, the bound test_capsule derives).
    padded = np.pad(truth, 1, mode="edge")
    neighbours = np.stack([padded[1 + dj:1 + dj + HEIGHT, 1 + di:1 + di + WIDTH] for dj in (-1, 0, 1) for di in (-1, 0, 1)])
    clear, solid = np.all(neighbours == 0.0, axis=0), np.all(neighbours == 1.0, axis=0)
    line = 1.0 / SUB + 0.118 / _least_curvature_radius(focal)
    for coverage in ("disc", "box_edge"):
        ball = Subject("b", "sphere", {"radius": RADIUS}, appearance=Appearance.opaque(), moving=True,
                       coverage=coverage)
        frame = render([ball], camera, pose, {"b": (jnp.asarray(CENTRE), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
        covered = 1.0 - np.asarray(frame.transmission, dtype=np.float64)
        assert bool(frame.flags["behind:b"])
        assert clear.sum() > 1000 and solid.sum() > 1000
        assert float(np.max(covered[clear])) == 0.0 and float(np.min(covered[solid])) == 1.0, coverage
        assert float(np.max(np.abs(covered - truth))) <= line, coverage
