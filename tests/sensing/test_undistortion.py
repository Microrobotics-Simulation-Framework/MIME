"""G2, the distortion's inverse: each pixel's ray against a float64 Newton solve of the distortion written here
(iterated to convergence, its residual checked) on the radial map's first monotone branch (inside the validity radius,
the first turn of ``r (1 + k1 r^2 + k2 r^4 + k3 r^6)``, found here by a fine scan and bisection), on drawn lenses at
physical calibrations' sizes (|k1| to 0.4), in both dtypes; and the pixels beyond that radius flagged.

The tolerance is 64 units in the last place of the ray's direction in the run's dtype, times the inverse's own
condition (one over the distortion's least singular value there, at least one): near the validity radius the map flattens
and a point's rounding moves its inverse further. A pixel the ray table flags beyond the radius is not compared, and
may be one the oracle reaches only where the inverse is ill-conditioned (its condition above ``FLAG_CONDITION``: the
tangential terms move the fold off the radial map's, within a few hundredths of the radius).

**Every ray the table does not flag is a solution**, judged on the ray itself: its point (the direction's ``x / z``,
``y / z``), mapped through the distortion in float64, lands on its pixel within the table's own convergence line (1024
units in the last place of the distorted radius plus one) plus the rounding of the point and of the map's evaluation in
the run's dtype (16 units, times the map's largest singular value), on the first branch to its turn's rounding. (As
first written the gate asked instead that every pixel the oracle cannot reach be flagged. In float32 that premise fails
near the fold: a pixel within float32's convergence line of the fold's peak has a solution to float32's rounding, which
the float64 oracle, held to 1e-14, calls unreachable: at each of the review's three failing float32 lenses, a pixel whose
float32 point lands within 0.002 to 0.007 px. Two of them also showed a real defect: the undistortion took one more
Newton step after judging its point, which near the fold's nearly singular Jacobian moved a ray 1.08 px off its pixel
at the first and past the fold, onto the second branch, at the second, unflagged. The point returned is now the point
judged; the three lenses are examples here.)

**The controls:** ``CAMERA_PLANT=undistort-truncated`` stops the solve after one radial step and no tangential one;
drawn lenses then miss by up to the square of their distortion. (As first written it truncated the radial solve alone,
which the tangential steps then finished: caught 4 of 30, the four the clean arm also failed.)
``CAMERA_PLANT=step-after-check`` takes one more step's value after the check: the first of the review's lenses (an
example here) fails in float32, every run (float64's rounding does not reach the fold's band at it). The undistortion's step count
was raised to twelve after two float64 draws (examples here, ``SLOW``) kept a ray up to 3.7 times this gate's line off
with the point judged after eight or nine: from the radial root a pixel near the fold took ten steps to converge."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import planted
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import mime.sensing._detector as detector
from mime.sensing import BrownConrady, CameraPose, Pinhole, ray_table

WIDTH, HEIGHT = 48, 36
#: The inverse's condition above which a pixel the oracle reaches may still be flagged beyond the validity radius.
FLAG_CONDITION = 4.0


def first_turn(lens: tuple[float, ...], cap: float) -> float:
    """The radial map's first turn in ``r^2`` up to ``cap``: a scan of 20 001 points, then bisection."""
    k1, k2, _, _, k3 = lens
    grid = np.linspace(0.0, cap, 20001)
    slope = 1 + grid * (3 * k1 + grid * (5 * k2 + 7 * k3 * grid))
    bad = np.nonzero(slope <= 0)[0]
    if len(bad) == 0:
        return cap
    low, high = grid[bad[0] - 1], grid[bad[0]]
    for _ in range(80):
        middle = (low + high) / 2
        low, high = (middle, high) if 1 + middle * (3 * k1 + middle * (5 * k2 + 7 * k3 * middle)) > 0 else (low, middle)
    return float(low)


def oracle(lens: tuple[float, ...], xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
    """Float64 Newton on the distortion, with its Jacobian written afresh here, from the radial map's own inverse on
    its first branch (by bisection, 200 steps) and iterated to convergence; each point's validity (its residual under
    1e-14 of its radius plus one, on the first branch) and the inverse's condition there returned."""
    k1, k2, p1, p2, k3 = lens
    distorted = np.hypot(xd, yd)
    turn = first_turn(lens, (4 * distorted.max() + 1) ** 2)
    low, high = np.zeros_like(distorted), np.full_like(distorted, np.sqrt(turn))
    for _ in range(200):
        middle = (low + high) / 2
        m2 = middle * middle
        below = middle * (1 + m2 * (k1 + m2 * (k2 + m2 * k3))) < distorted
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    scale = np.where(distorted > 0, low / np.where(distorted > 0, distorted, 1), 1)
    x, y = xd * scale, yd * scale
    for _ in range(100):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        fx = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        fy = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        dradial = 2 * k1 + 4 * k2 * r2 + 6 * k3 * r2 ** 2
        a = radial + x * x * dradial + 2 * p1 * y + 6 * p2 * x
        b = x * y * dradial + 2 * p1 * x + 2 * p2 * y
        d = radial + y * y * dradial + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * b
        x, y = x - (d * fx - b * fy) / det, y - (a * fy - b * fx) / det
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    residual = np.hypot(x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd,
                        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd)
    dradial = 2 * k1 + 4 * k2 * r2 + 6 * k3 * r2 ** 2
    a = radial + x * x * dradial + 2 * p1 * y + 6 * p2 * x
    b = x * y * dradial + 2 * p1 * x + 2 * p2 * y
    d = radial + y * y * dradial + 6 * p1 * y + 2 * p2 * x
    least = np.abs((a + d) / 2 - np.sqrt(((a - d) / 2) ** 2 + b * b))
    valid = (residual < 1e-14 * (1 + distorted)) & (r2 < turn)
    return x, y, valid, 1 / np.maximum(np.minimum(least, 1.0), 1e-300)


lenses = st.tuples(st.floats(-0.4, 0.3), st.floats(-0.2, 0.2), st.floats(-3e-3, 3e-3), st.floats(-3e-3, 3e-3),
                   st.floats(-0.1, 0.1))


def unsolved(lens: tuple[float, ...], direction: np.ndarray, xd: np.ndarray, yd: np.ndarray, eps: float) -> np.ndarray:
    """Where a ray (``direction``, ``(3, height, width)``, as the run's dtype rounded it) is not a solution of the
    distortion on its first branch: its point mapped through the distortion in float64 lands off its pixel by more than
    the table's own convergence line, ``1024 eps (1 + r_d)``, plus the rounding of the point and of the map's evaluation
    in the run's dtype (``16 eps (1 + r_d)`` times the map's largest singular value); or the point lies beyond the
    first turn by more than that turn's rounding (``8 eps`` of it)."""
    k1, k2, p1, p2, k3 = lens
    x, y = direction[0] / direction[2], direction[1] / direction[2]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    residual = np.hypot(x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd,
                        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd)
    dradial = 2 * k1 + 4 * k2 * r2 + 6 * k3 * r2 ** 2
    a = radial + x * x * dradial + 2 * p1 * y + 6 * p2 * x
    b = x * y * dradial + 2 * p1 * x + 2 * p2 * y
    d = radial + y * y * dradial + 6 * p1 * y + 2 * p2 * x
    largest = np.abs((a + d) / 2) + np.sqrt(((a - d) / 2) ** 2 + b * b)
    distorted = np.hypot(xd, yd)
    line = 1024 * eps * (1 + distorted) + 16 * eps * (1 + distorted) * np.maximum(largest, 1.0)
    turn = first_turn(lens, (4 * distorted.max() + 1) ** 2)
    return (residual > line) | (r2 > turn * (1 + 8 * eps))


_undistort = BrownConrady.undistort


def _step_after_check(self, xd, yd, steps=None):
    """The control: the undistortion as the review found it, one more Newton step's value taken after the check."""
    x, y, found = _undistort(self, xd, yd, steps)
    x, y = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y)
    return (*self._newton(x, y, xd, yd), found)


def rays_against_oracle(lens: tuple[float, ...], focal: float) -> tuple[float, float, int, int]:
    """The worst angle between a ray and the oracle's (radians, over the inverse's condition) where both find one, the
    tolerance, the pixels flagged where the oracle finds a well-conditioned inverse, and the rays not flagged that are
    not solutions (``unsolved``)."""
    camera = Pinhole(WIDTH, HEIGHT, focal, focal * 1.01, (WIDTH - 1) / 2 + 1.5, (HEIGHT - 1) / 2 - 0.5,
                     BrownConrady(*lens))
    pose = CameraPose(jnp.zeros(3), jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    held = detector.UNDISTORT_STEPS, detector.TANGENTIAL_STEPS
    if planted("undistort-truncated"):
        detector.UNDISTORT_STEPS, detector.TANGENTIAL_STEPS = 1, 0
    undistort = BrownConrady.undistort
    if planted("step-after-check"):
        BrownConrady.undistort = _step_after_check
    try:
        rays = ray_table(camera, pose)
    finally:
        detector.UNDISTORT_STEPS, detector.TANGENTIAL_STEPS = held
        BrownConrady.undistort = undistort
    v, u = np.meshgrid(np.arange(HEIGHT, dtype=np.float64), np.arange(WIDTH, dtype=np.float64), indexing="ij")
    xd, yd = (u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy
    x, y, valid, condition = oracle(lens, xd, yd)
    norm = np.sqrt(x * x + y * y + 1)
    expected = np.stack([x / norm, y / norm, 1 / norm])
    found = np.stack([np.asarray(c, dtype=np.float64) for c in rays.direction])
    angle = np.linalg.norm(found - expected, axis=0)
    eps = float(jnp.finfo(rays.direction[0].dtype).eps)
    beyond = np.asarray(rays.beyond)
    return (float(np.max(np.where(valid & ~beyond, angle / condition, 0.0))), 64 * eps,
            int(np.sum(beyond & valid & (condition <= FLAG_CONDITION))),
            int(np.sum(unsolved(lens, found, xd, yd, eps) & ~beyond)))


#: The review's three failing float32 draws: the first two the step after the check, all three the first premise.
REVIEWED = [((-0.3700219549351325, 0.0, 0.002954171859948173, 0.0, 0.0), 0.8),
            ((-0.3984375, -0.18532702511409785, 0.0, 0.002557456254940388, 0.0), 0.96484375),
            ((-0.375, 0.015625, 0.0029999999999999996, 0.0, 0.001953125), 0.8125)]
#: Two float64 draws whose rays near the fold converged only after nine and after ten steps (the fix round's CI, stress
#: 19 of 20, and its probe, seed 122).
SLOW = [((-0.34765625, -0.09375, 0.0008000246296658098, 0.0, 0.0), 0.875),
        ((-0.28697896535758216, -0.14833635426816824, 0.0004523831568567214, 0.0, 0.0), 0.8660469190672981)]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(lens=lenses, field=st.floats(0.8, 1.6))
@example(lens=REVIEWED[0][0], field=REVIEWED[0][1])
@example(lens=REVIEWED[1][0], field=REVIEWED[1][1])
@example(lens=REVIEWED[2][0], field=REVIEWED[2][1])
@example(lens=SLOW[0][0], field=SLOW[0][1])
@example(lens=SLOW[1][0], field=SLOW[1][1])
def test_each_ray_is_the_distortions_inverse(dtype, lens, field):
    worst, tolerance, flagged_well_conditioned, unflagged_unsolved = rays_against_oracle(lens, field * WIDTH)
    assert worst <= tolerance, f"{worst:.3g} rad against {tolerance:.3g}"
    assert flagged_well_conditioned == 0 and unflagged_unsolved == 0, (flagged_well_conditioned, unflagged_unsolved)


@pytest.mark.parametrize("k1", [-0.9, 0.0])
def test_pixels_beyond_the_validity_radius_are_flagged(dtype, k1):
    """A wide lens with a strong barrel (``k1 = -0.9`` over a field of 1.6 normalised units across) folds back inside
    the image: its outer pixels have no inverse, and are flagged; an undistorted lens flags none."""
    lens = (k1, 0.0, 0.0, 0.0, 0.0)
    _, _, flagged_well_conditioned, unflagged_unsolved = rays_against_oracle(lens, 0.6 * WIDTH)
    camera = Pinhole(WIDTH, HEIGHT, 0.6 * WIDTH, 0.6 * WIDTH * 1.01, (WIDTH - 1) / 2 + 1.5, (HEIGHT - 1) / 2 - 0.5,
                     BrownConrady(*lens))
    flagged = int(jnp.sum(ray_table(camera, CameraPose(jnp.zeros(3), jnp.asarray((1.0, 0.0, 0.0, 0.0)))).beyond))
    assert flagged_well_conditioned == 0 and unflagged_unsolved == 0
    assert (flagged > 0) == (k1 != 0.0)


def test_the_validity_radius_is_the_radial_maps_turning_point(float64):
    """``k1 = -0.9``: the radial map ``r (1 + k1 r^2)`` turns at ``r^2 = -1 / (3 k1)``, where the flag starts."""
    turning = math.sqrt(1 / 2.7)
    lens = BrownConrady(-0.9)
    assert bool(lens.monotone(jnp.asarray(turning * 0.999), jnp.asarray(0.0)))
    assert not bool(lens.monotone(jnp.asarray(turning * 1.001), jnp.asarray(0.0)))
