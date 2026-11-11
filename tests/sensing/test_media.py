"""G7, media on two levels: a translucent sphere in tinted water in a tinted tube wall, seen across the tube, transmits
``exp(-mu_w (L_o - L_l) - mu_f (L_l - L_s) - mu_s L_s)`` (``L_o``, ``L_l``, ``L_s`` the chords of the wall's outer
cylinder, the lumen and the sphere), computed here from rays and chords written apart from ``mime.sensing``, at every
pixel 2 px or more from each silhouette (coverage there exactly none or all), in both dtypes; an opaque body is coverage
(finite everywhere); and a robot resting on the floor within its tolerance raises no flag, one pushed into the wall does.

**The controls:** multiplied transmittances, a child attenuating by its own coefficient without its parent's displaced
(``CAMERA_PLANT=no-displacement``), fail the transmission; an opaque body written as an infinite attenuation
(``CAMERA_PLANT=infinite``) gives ``inf x 0`` beyond its silhouette and fails the finiteness check."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from camera_cases import TUBE, across, planted, tube_subjects

import mime.sensing._render as render_module
from mime.sensing import Appearance, Subject, render
from mime.sensing._subjects import Look

WALL, WATER, SPHERE = 200.0, 20.0, 2000.0
RADIUS = 0.7e-3
CENTRE = (0.3e-3, 0.0, -0.9e-3)


def oracle_rays(width: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Each pixel's ray through a pinhole without distortion, written afresh: the camera at ``(0, -0.1, 0)``, looking
    along +y, the image's x along +x and its y along -z."""
    height = width * 3 // 4
    focal = 0.1 / (13.3e-3 / width)
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    x, y = (u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal
    direction = np.stack([x, np.ones_like(x), -y], axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return np.array([0.0, -0.1, 0.0]), direction, focal


def chord_cylinder(origin: np.ndarray, d: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The chord of the unbounded cylinder about the x axis, and the ray's distance from its axis less the radius."""
    a = d[..., 1] ** 2 + d[..., 2] ** 2
    b = origin[1] * d[..., 1] + origin[2] * d[..., 2]
    c = origin[1] ** 2 + origin[2] ** 2 - radius ** 2
    disc = b * b - a * c
    closest = np.sqrt(np.maximum(c + radius ** 2 - b * b / a, 0.0))
    return 2 * np.sqrt(np.maximum(disc, 0.0)) / a, closest - radius


def chord_sphere(origin: np.ndarray, d: np.ndarray, centre: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    w = origin - centre
    b = d @ w
    closest = np.linalg.norm(w[None, None, :] - b[..., None] * d, axis=-1)
    return 2 * np.sqrt(np.maximum(radius ** 2 - closest ** 2, 0.0)), closest - radius


def scene(sphere: Appearance, centre: tuple[float, float, float] = CENTRE) -> tuple[list[Subject], dict[str, object]]:
    held = tube_subjects(Appearance.attenuating(WALL), Appearance.attenuating(WATER))
    held.append(Subject("umr", "sphere", {"radius": RADIUS}, appearance=sphere, parent="water", moving=True))
    return held, {"umr": (jnp.asarray(centre), jnp.asarray((1.0, 0.0, 0.0, 0.0)))}


def test_the_displaced_medium_rule_is_the_closed_form(dtype, monkeypatch):
    if planted("no-displacement"):
        def multiplied(appearance, parent, chord, dtype):
            return jnp.exp(-jnp.asarray(appearance.attenuation, dtype=dtype) * chord)
        monkeypatch.setattr(render_module, "_transmits", multiplied)
    width = 160
    camera, pose = across(width)
    subjects, poses = scene(Appearance.attenuating(SPHERE))
    found = np.asarray(render(subjects, camera, pose, poses).transmission, dtype=np.float64)
    origin, d, focal = oracle_rays(width)
    outer, outer_edge = chord_cylinder(origin, d, TUBE["inner_radius"] + TUBE["wall"])
    lumen, lumen_edge = chord_cylinder(origin, d, TUBE["inner_radius"])
    ball, ball_edge = chord_sphere(origin, d, np.asarray(CENTRE), RADIUS)
    expected = np.exp(-WALL * (outer - lumen) - WATER * (lumen - ball) - SPHERE * ball)
    footprint = 0.1 / focal
    clear = np.minimum.reduce([np.abs(outer_edge), np.abs(lumen_edge), np.abs(ball_edge)]) > 2.0 * footprint
    assert clear.sum() > 0.8 * clear.size and (ball_edge < -2.0 * footprint).sum() > 100
    # Judged on the scale it came from: a chord's rounding is a few units in the last place of the ray's parameter
    # (the camera 0.1 m away), times each attenuation (float32: about 4e-4 of the transmission; float64 under 1e-12).
    eps = float(np.finfo(dtype).eps)
    tolerance = max(1e-10 if dtype == "float64" else 2e-5, 16 * eps * (WALL + WATER + SPHERE) * 0.1)
    assert float(np.max(np.abs(found[clear] / expected[clear] - 1.0))) <= tolerance


def test_an_opaque_body_is_coverage_and_finite_everywhere(dtype):
    camera, pose = across(160)
    sphere = Appearance(Look.ATTENUATING, attenuation=math.inf) if planted("infinite") else Appearance.opaque()
    subjects, poses = scene(sphere)
    frame = render(subjects, camera, pose, poses)
    assert bool(jnp.all(jnp.isfinite(frame.transmission))) and bool(jnp.all(jnp.isfinite(frame.expected)))
    origin, d, focal = oracle_rays(160)
    _, ball_edge = chord_sphere(origin, d, np.asarray(CENTRE), RADIUS)
    inside = ball_edge < -2.0 * 0.1 / focal
    assert float(jnp.max(jnp.where(jnp.asarray(inside), frame.transmission, 0.0))) == 0.0


def test_a_robot_on_the_floor_is_inside_and_one_through_the_wall_is_flagged(dtype):
    camera, pose = across(160)
    floor = -TUBE["inner_radius"] + RADIUS
    resting, poses = scene(Appearance.opaque(), (0.0, 0.0, floor + 1e-7))
    assert int(render(resting, camera, pose, poses).flags["outside_parent:umr"]) == 0
    through, poses = scene(Appearance.opaque(), (0.0, 0.0, floor - 0.1 * RADIUS))
    assert int(render(through, camera, pose, poses).flags["outside_parent:umr"]) > 0
