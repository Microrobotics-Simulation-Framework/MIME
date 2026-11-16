"""G20 on the coil: the march at its declared count against twice it (image and motion), and against the sign-only truth
(motion, image), at a straight and a tilted pose; the controls (the march's distance offset by a tenth of a pixel, the
image against the truth; 8 samples, against the N to 2N line). Motion as camera truth 1 reads it: the vector (the area's
equivalent radius, the centroid), by forward mode for the march and by central differences at a quarter pixel for the
truth (64 x 64 samples a pixel, 512 signs). python scripts/camera_route2/g20.py OUT.json, from the repository's root."""

from __future__ import annotations

import json
import math
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "tests/sensing")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import sampled_truth  # noqa: E402
import test_sampled as T  # noqa: E402

import mime.sensing._sampled as sampled_route  # noqa: E402
from mime.sensing import render  # noqa: E402


def about(axis: int, degrees: float) -> np.ndarray:
    q = np.zeros(4)
    q[0], q[1 + axis] = math.cos(math.radians(degrees) / 2), math.sin(math.radians(degrees) / 2)
    return q


def product(a, b):
    w1, v1, w2, v2 = a[0], a[1:], b[0], b[1:]
    return np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])


POSES = {"straight": (np.array([0.02e-3, 0.0, 0.01e-3]), np.array([1.0, 0.0, 0.0, 0.0])),
         "tilted": (np.array([0.05e-3, 0.0, -0.03e-3]), product(about(1, 25.0), about(2, 15.0)))}
STEP = 0.25 * 0.1 / T.LENS.fx  # a quarter pixel at the coil
TRUTH_SIDE = T.TRUTH_SIDE


def at(count: int, function, *arguments):
    """``function(coil, *arguments)`` with the march taking exactly ``count`` samples a ray: the count is read whenever a
    subject is made or placed, and the render places it again, so the constants are held across the whole call."""
    held = sampled_route.SAMPLES, sampled_route.SPACING
    sampled_route.SAMPLES, sampled_route.SPACING = count, 1e6
    try:
        body = T.subject("helix", T.COIL)
        assert body.placed.sampled.samples == count
        return function(body, *arguments)
    finally:
        sampled_route.SAMPLES, sampled_route.SPACING = held


def image(body, position, orientation) -> np.ndarray:
    found = render([body], T.LENS, T.POSE, {"body": (jnp.asarray(position), jnp.asarray(orientation))})
    return 1.0 - np.asarray(found.transmission, dtype=np.float64)


def aggregates(cover: np.ndarray) -> np.ndarray:
    """Camera truth 1's vector: the area's equivalent radius (px) and the centroid (px), whose motions are compared."""
    v, u = np.indices(cover.shape)
    total = cover.sum()
    return np.array([math.sqrt(total / math.pi), (cover * u).sum() / total, (cover * v).sum() / total])


def motion(body, position, orientation) -> np.ndarray:
    """(2, 3): the derivative in x and in z of the darkness's total and centroid, by forward mode."""
    v, u = np.indices((T.LENS.height, T.LENS.width))

    def of(at):
        dark = 1.0 - render([body], T.LENS, T.POSE, {"body": (at, jnp.asarray(orientation))}).transmission
        total = jnp.sum(dark)
        return jnp.stack([jnp.sqrt(total / jnp.pi), jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    jacobian = np.asarray(jax.jacfwd(of)(jnp.asarray(position)))
    return np.stack([jacobian[:, 0], jacobian[:, 2]])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def truth_image(body, position, orientation, band: np.ndarray, outside: np.ndarray) -> np.ndarray:
    cover = outside.copy().reshape(-1)
    pixels = np.nonzero(band.reshape(-1))[0]
    cover[pixels] = sampled_truth.truth(body, T.LENS, T.POSE, position, orientation, pixels, side=TRUTH_SIDE, signs=512)
    return cover.reshape(band.shape)


def main() -> None:
    out = sys.argv[1]
    result = {}
    count = T.subject("helix", T.COIL).placed.sampled.samples
    fine_count, truth_body = 2 * count - 1, T.subject("helix", T.COIL)
    for name, (position, orientation) in POSES.items():
        a, b = at(count, image, position, orientation), at(fine_count, image, position, orientation)
        ma, mb = at(count, motion, position, orientation), at(fine_count, motion, position, orientation)
        band = dilate((b > 1e-9) & (b < 1 - 1e-9), 2)
        outside = np.round(b)
        truth = truth_image(truth_body, position, orientation, band, outside)
        moved = []
        for axis in (0, 2):
            step = np.zeros(3)
            step[axis] = STEP
            plus = truth_image(truth_body, position + step, orientation, band,
                               np.round(at(fine_count, image, position + step, orientation)))
            minus = truth_image(truth_body, position - step, orientation, band,
                                np.round(at(fine_count, image, position - step, orientation)))
            moved.append((aggregates(plus) - aggregates(minus)) / (2 * STEP))
        moved = np.stack(moved)
        own = []
        for axis in (0, 2):
            step = np.zeros(3)
            step[axis] = STEP
            plus_own, minus_own = (aggregates(at(count, image, position + sign * step, orientation)) for sign in (1, -1))
            own.append((plus_own - minus_own) / (2 * STEP))
        own = np.stack(own)
        found = sampled_route.evaluate
        sampled_route.evaluate = lambda s, p, x, y, z: found(s, p, x, y, z) + T.OFFSET  # noqa: E731 - the plant
        mp = at(count, motion, position, orientation)
        offset_image = at(count, image, position, orientation)
        sampled_route.evaluate = found
        motion_eight = at(8, motion, position, orientation)
        result[name] = {
            "samples": count,
            "n_to_2n": {"largest_pixel": float(np.max(np.abs(a - b))), "motion_x": rel(ma[0], mb[0]),
                        "motion_z": rel(ma[1], mb[1])},
            "truth": {"largest_pixel": float(np.max(np.abs(a - truth))), "L1_px2": float(np.abs(a - truth).sum()),
                      "band_pixels": int(band.sum()), "motion_x": rel(ma[0], moved[0]), "motion_z": rel(ma[1], moved[1]),
                      "motion_x_2n": rel(mb[0], moved[0]), "motion_z_2n": rel(mb[1], moved[1])},
            "own_finite_difference": {"motion_x": rel(ma[0], own[0]), "motion_z": rel(ma[1], own[1])},
            "control_offset": {"motion_x": rel(mp[0], moved[0]), "motion_z": rel(mp[1], moved[1]),
                               "largest_pixel": float(np.max(np.abs(offset_image - truth)))},
            "control_eight_samples": {"largest_pixel": float(np.max(np.abs(at(8, image, position, orientation) - b))),
                                      "motion_x": rel(motion_eight[0], mb[0]), "motion_z": rel(motion_eight[1], mb[1])},
        }
        print(name, json.dumps(result[name]), flush=True)
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
