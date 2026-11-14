"""G15, the box edge, and G3, a sphere's silhouette: the coverage a sphere's pixels take and its derivative in the
sphere's centre and the focal length, against the exact oracle of ``sphere_oracle`` (each pixel's exact covered area,
and the boundary integral of its derivative), at silhouette radii of 15, 6 and 3 px, through a pinhole without
distortion, in both dtypes; and exact disc coverage against the same (to the oracle's quadrature, and in float32 to the
rounding of the camera's rotation and of the disc's own arithmetic: 2 eps f of coverage).

The lines are the predictions' (``predictions_camera_core.json``, gate 15): per pixel the relative L2 over the
silhouette's band, and the silhouette's motion (its equivalent radius and centroid as one vector). **The control**
(``CAMERA_PLANT=sigmoid``) puts spike 1's soft edge at 1 px in place of the box edge: it fails the per-pixel lines."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sphere_oracle
from camera_cases import planted

import mime.sensing._coverage as coverage
from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

WIDTH, HEIGHT = 160, 120
DISTANCE, RADIUS = 30e-3, 0.5e-3
#: The predicted lines by radius: the box edge's derivative and coverage per pixel, and its motion.
BOX = {15: (0.08, 0.03, 0.005), 6: (0.12, 0.06, 0.015), 3: (0.2, 0.12, 0.04)}
DISC = (1e-5, 0.005)
#: Off the optical axis, by a fraction of a pixel in each direction.
OFFSET_PX = (20.37, -12.61)


def theta_for(radius_px: int) -> np.ndarray:
    focal = radius_px * DISTANCE / RADIUS
    return np.array([OFFSET_PX[0] * DISTANCE / focal, 0.0, -OFFSET_PX[1] * DISTANCE / focal, focal])


def image(theta: jax.Array, mode: str, window: int | None = None) -> jax.Array:
    """The sphere's coverage image at ``theta`` (its centre's world x, depth offset y and z, and the focal length)."""
    camera = Pinhole(WIDTH, HEIGHT, theta[3], theta[3], (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
    s = np.sqrt(0.5)
    pose = CameraPose(jnp.asarray([0.0, -DISTANCE, 0.0]), jnp.asarray([s, -s, 0.0, 0.0]))
    ball = Subject("ball", "sphere", {"radius": RADIUS}, appearance=Appearance.opaque(), moving=True, coverage=mode,
                   window=window)
    frame = render([ball], camera, pose, {"ball": (theta[:3], jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    return 1.0 - frame.transmission


@functools.cache
def truth(radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    theta = theta_for(radius_px)
    args = (RADIUS, DISTANCE, (WIDTH - 1) / 2, (HEIGHT - 1) / 2, (HEIGHT, WIDTH))
    return sphere_oracle.areas(theta, *args), sphere_oracle.derivatives(theta, *args)


def measured(radius_px: int, mode: str, monkeypatch: pytest.MonkeyPatch) -> tuple[float, float, float]:
    """The worst per-pixel derivative error, the per-pixel coverage error and the motion error over the parameters."""
    if planted("sigmoid"):
        monkeypatch.setattr(coverage, "covered", lambda edge, size=1.0: coverage.sigmoid(edge, size))
    areas, derivatives = truth(radius_px)
    theta = jnp.asarray(theta_for(radius_px))
    # Exact disc coverage is always windowed: under a derivative its window is declared (the silhouette and its edge).
    window = radius_px + 4 if mode == "disc" else None
    covered = np.asarray(image(theta, mode, window), dtype=np.float64)
    jacobian = np.asarray(jax.jacfwd(lambda t: image(t, mode, window))(theta), dtype=np.float64)
    jacobian = np.moveaxis(jacobian, -1, 0)
    band = (areas > 0) & (areas < 1)
    coverage_error = float(np.linalg.norm((covered - areas)[band]) / np.linalg.norm(areas[band]))
    pixel, motion = [], []
    for k in range(4):
        pixel.append(np.linalg.norm(jacobian[k] - derivatives[k]) / np.linalg.norm(derivatives[k]))
        ours, exact = sphere_oracle.motion(covered, jacobian[k]), sphere_oracle.motion(areas, derivatives[k])
        motion.append(np.linalg.norm(ours - exact) / np.linalg.norm(exact))
    return float(max(pixel)), coverage_error, float(max(motion))


@pytest.mark.parametrize("radius_px", [15, 6, 3])
def test_the_box_edge_within_its_measured_lines(dtype, radius_px, monkeypatch):
    pixel, covered, motion = measured(radius_px, "box_edge", monkeypatch)
    lines = BOX[radius_px]
    assert pixel <= lines[0] and covered <= lines[1] and motion <= lines[2], (pixel, covered, motion)


@pytest.mark.parametrize("radius_px", [15, 6, 3])
def test_exact_disc_coverage_is_exact(dtype, radius_px, monkeypatch):
    areas, _ = truth(radius_px)
    covered = np.asarray(image(jnp.asarray(theta_for(radius_px)), "disc"), dtype=np.float64)
    # Exact up to the oracle's quadrature and the run's rounding: the camera's rotation, made from its quaternion in the
    # run's dtype, is off by a unit in the last place in an entry, which turns the line of sight by eps and moves the
    # image eps f px, a pixel's coverage by that times its edge's chord (sqrt 2): 1.41 eps f; and the disc's own
    # arithmetic, 0.3 eps f measured in float32 with exact inputs (an exact rotation, the pose's numbers representable,
    # the oracle at the inputs as rounded) at 15, 6 and 3 px. The floor is their sum, 2 eps f. (The position's rounding,
    # which the review named, moves it by under a tenth of that: 0.03 eps f with the camera's distance made exact.)
    floor = DISC[0] + 2 * float(jnp.finfo(dtype).eps) * float(theta_for(radius_px)[3])
    assert float(np.max(np.abs(covered - areas))) <= floor
    pixel, _, motion = measured(radius_px, "disc", monkeypatch)
    assert pixel <= DISC[1] and motion <= DISC[1], (pixel, motion)


def test_the_silhouette_matches_its_conic(dtype, monkeypatch):
    """G3: the covered area and its centroid against the conic's (area and centre, closed form), at 15 px; and a move
    of a tenth of a pixel across the view moves the coverage centroid by its tenth of a pixel. Controls: the principal
    point half a pixel off (``CAMERA_PLANT=principal-half``) and a hard edge (``CAMERA_PLANT=hard-edge``)."""
    if planted("hard-edge"):
        monkeypatch.setattr(coverage, "covered", lambda edge, size=1.0: (edge.distance < 0).astype(edge.distance.dtype))
    theta = theta_for(15)
    shift = 0.5 if planted("principal-half") else 0.0
    _, centre, area = sphere_oracle.ellipse(theta, RADIUS, DISTANCE, (WIDTH - 1) / 2 - shift, (HEIGHT - 1) / 2)
    covered = np.asarray(image(jnp.asarray(theta), "box_edge"), dtype=np.float64)
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    found = np.array([np.sum(u * covered), np.sum(v * covered)]) / covered.sum()
    assert abs(covered.sum() / area - 1) <= 0.003
    assert float(np.max(np.abs(found - centre))) <= 0.01
    moved = theta + np.array([0.1 * DISTANCE / theta[3], 0.0, 0.0, 0.0])
    after = np.asarray(image(jnp.asarray(moved), "box_edge"), dtype=np.float64)
    _, centre_after, _ = sphere_oracle.ellipse(moved, RADIUS, DISTANCE, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
    step = np.sum(u * after) / after.sum() - found[0]
    expected = centre_after[0] - (centre[0] + shift)
    assert abs(step / expected - 1) <= 0.02, (step, expected)


def quads_with_the_origin_on_a_side(count: int = 200) -> list[list[tuple[float, float]]]:
    """Convex quadrilaterals (a pixel's corners in the disc's units) with the circle's centre on one of their sides,
    the side's ends at drawn distances from it either side, from a tenth of the radius to ten times it."""
    rng = np.random.default_rng(15)
    found = []
    for _ in range(count):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        u, n = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        a, b = 10.0 ** rng.uniform(-1.0, 1.0, size=2)
        depth = 10.0 ** rng.uniform(-1.0, 1.0)
        first, second = -a * u, b * u
        found.append([tuple(first), tuple(second), tuple(second + depth * n), tuple(first + depth * n)])
    return found


def is_clear(quad: list[tuple[float, float]]) -> bool:
    """``_box_coverage``'s ``clear``, as it forms it from a pixel's corners: none inside the circle and no side meeting it."""
    corners = [coverage._crosses(*quad[i], *quad[(i + 1) % 4]) for i in range(4)]
    inside = [x * x + y * y < 1.0 for x, y in quad]
    return not any(inside) and not any(bool(c) for c in corners)


def test_a_pixel_with_the_discs_centre_on_a_side_is_never_clear_of_the_circle(float64):
    """Why ``camera.disc.holds`` is reached only where no side crosses the circle, and so is a jump no harness row can
    place a point at: a side through the centre meets the circle (two ends outside it, the segment crosses it twice;
    an end inside, a corner is inside), so ``clear`` is false there and ``held`` is not chosen."""
    assert not any(is_clear(quad) for quad in quads_with_the_origin_on_a_side())


def test_control_a_side_that_never_meets_the_circle_makes_the_pixel_clear(float64, monkeypatch):
    """The control: a ``_crosses`` that says no side meets the circle makes the pixels whose ends are both outside it
    clear, so the check above can fail (every draw with both ends beyond the radius, which the draws include)."""
    monkeypatch.setattr(coverage, "_crosses", lambda ax, ay, bx, by: False)
    assert sum(is_clear(quad) for quad in quads_with_the_origin_on_a_side()) > 20
