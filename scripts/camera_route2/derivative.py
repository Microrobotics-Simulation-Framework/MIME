"""Route 2's derivative on the coil, measured: what the forward-mode derivative of the image's aggregates is against
central differences, by the scheme's parts. ``python scripts/camera_route2/derivative.py OUT.json``, from the
repository's root (the coil of ``tests/sensing/test_sampled.py`` at 128 x 96, float64).

At G20's straight and tilted poses, the aggregates are the darkness's total (px^2) and its centroid (u, v; px), their
motion per pixel of the body's motion in x and in z:

- ``step_sweep``: the central difference of the aggregates against its step, 1/256 to 2 px, with the fix and with
  the PR's first scheme (one straight edge a cell, the least of the two branches' lines, each read at its pixel's held
  point): where it holds, it is the reference;
- ``ripple``: the jvp at 32 positions over one pixel of motion against the central difference over that pixel, by
  variant: ``first`` (the PR's first scheme), ``union`` (the exact union, each branch at its pixel's held point),
  ``least`` (one straight edge a cell, each branch at its sub-ray's own least), ``fix`` (both);
- ``truth``: the sign-only truth's motion (64 x 64 samples a pixel, 512 signs, central differences at a quarter pixel)
  against the fix's jvp at the pose."""

from __future__ import annotations

import json
import math
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path[:0] = ["tests/sensing", "tests"]
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sampled_truth  # noqa: E402
import test_sampled as T  # noqa: E402

import mime.sensing._sampled as sampled_route  # noqa: E402
from mime.sensing import _coverage as coverage  # noqa: E402
from mime.sensing import render  # noqa: E402


def about(axis: int, degrees: float) -> np.ndarray:
    q = np.zeros(4)
    q[0], q[1 + axis] = math.cos(math.radians(degrees) / 2), math.sin(math.radians(degrees) / 2)
    return q


def product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, v1, w2, v2 = a[0], a[1:], b[0], b[1:]
    return np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])


POSES = {"straight": (np.array([0.02e-3, 0.0, 0.01e-3]), np.array([1.0, 0.0, 0.0, 0.0])),
         "tilted": (np.array([0.05e-3, 0.0, -0.03e-3]), product(about(1, 25.0), about(2, 15.0)))}
PIXEL = 0.1 / T.LENS.fx
STEPS = (1 / 256, 1 / 64, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
AXES = {"x": 0, "z": 2}
FOUND_PIECES, FOUND_COVER = sampled_route.pieces, sampled_route._covered_union


def variant(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the scheme's parts: the first scheme's held pieces and one straight edge, or the fix's."""
    monkeypatch.undo()
    first = pytest.MonkeyPatch()
    T.held_and_least(first)
    held_pieces, least_cover = sampled_route.pieces, sampled_route._covered_union
    first.undo()
    pieces = held_pieces if name in ("first", "union") else FOUND_PIECES
    cover = least_cover if name in ("first", "least") else FOUND_COVER
    monkeypatch.setattr(sampled_route, "pieces", pieces)
    monkeypatch.setattr(sampled_route, "_covered_union", cover)


def aggregates(body, orientation):
    v, u = np.indices((T.LENS.height, T.LENS.width))

    def of(at):
        dark = 1.0 - render([body], T.LENS, T.POSE, {"body": (at, jnp.asarray(orientation))}).transmission
        total = jnp.sum(dark)
        return jnp.stack([total, jnp.sum(dark * u) / total, jnp.sum(dark * v) / total])

    return jax.jit(of), jax.jit(lambda at, e: jax.jvp(of, (at,), (e,))[1])


def truth_motion(body, position, orientation, axis: int) -> np.ndarray:
    """The truth's aggregates' motion per pixel of motion along ``axis``, by central differences at a quarter pixel,
    over the pixels any of the three frames touches (the rest taken from the fix's rounded frame, all ones or zeros)."""
    def image(at):
        return 1.0 - np.asarray(render([body], T.LENS, T.POSE, {"body": (jnp.asarray(at), jnp.asarray(orientation))})
                                .transmission, dtype=np.float64)

    e = np.zeros(3)
    e[axis] = 0.25 * PIXEL
    frames = [image(position + k * e) for k in (-1, 1)]
    edge = np.zeros(frames[0].shape, bool)
    for f in frames:
        edge |= (f > 1e-9) & (f < 1 - 1e-9)
    band = np.zeros_like(edge)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            band |= np.roll(np.roll(edge, dy, 0), dx, 1)
    pixels = np.nonzero(band.reshape(-1))[0]
    v, u = np.indices(frames[0].shape)
    sums = []
    for f, k in zip(frames, (-1, 1), strict=True):
        cover = np.round(f).reshape(-1)
        cover[pixels] = sampled_truth.truth(body, T.LENS, T.POSE, position + k * e, orientation, pixels,
                                            side=T.TRUTH_SIDE, signs=512)
        cover = cover.reshape(f.shape)
        total = cover.sum()
        sums.append(np.array([total, (cover * u).sum() / total, (cover * v).sum() / total]))
    return (sums[1] - sums[0]) / 0.5


def main() -> None:
    out: dict = {}
    monkeypatch = pytest.MonkeyPatch()
    body = T.subject("helix", T.COIL)
    for pose, (position, orientation) in POSES.items():
        entry: dict = {"step_sweep": {}, "ripple": {}, "truth": {}}
        for scheme in ("first", "fix"):
            variant(scheme, monkeypatch)
            found, tangent = aggregates(body, orientation)
            sweep = {}
            for name, axis in AXES.items():
                e = np.zeros(3)
                e[axis] = PIXEL
                jvp = np.asarray(tangent(jnp.asarray(position), jnp.asarray(e)))
                sweep[name] = {"jvp": jvp.tolist(), "steps": {
                    str(h): ((np.asarray(found(jnp.asarray(position + h * e))) -
                              np.asarray(found(jnp.asarray(position - h * e)))) / (2 * h)).tolist() for h in STEPS}}
            entry["step_sweep"][scheme] = sweep
        for scheme in ("first", "union", "least", "fix"):
            variant(scheme, monkeypatch)
            found, tangent = aggregates(body, orientation)
            ripple = {}
            for name, axis in AXES.items():
                e = np.zeros(3)
                e[axis] = PIXEL
                reference = np.asarray(found(jnp.asarray(position + e))) - np.asarray(found(jnp.asarray(position)))
                centroid = []
                for k in range(32):
                    jvp = np.asarray(tangent(jnp.asarray(position + (k / 32) * e), jnp.asarray(e)))
                    centroid.append(float(np.linalg.norm(jvp[1:] - reference[1:]) / np.linalg.norm(reference[1:])))
                ripple[name] = {"largest": max(centroid), "mean": float(np.mean(centroid)),
                                "reference": reference.tolist()}
            entry["ripple"][scheme] = ripple
            print(pose, scheme, json.dumps(ripple), flush=True)
        variant("fix", monkeypatch)
        found, tangent = aggregates(body, orientation)
        for name, axis in AXES.items():
            e = np.zeros(3)
            e[axis] = PIXEL
            jvp = np.asarray(tangent(jnp.asarray(position), jnp.asarray(e)))
            truth = truth_motion(body, position, orientation, axis)
            entry["truth"][name] = {"truth": truth.tolist(), "jvp": jvp.tolist(),
                                    "centroid_relative": float(np.linalg.norm(jvp[1:] - truth[1:]) /
                                                               np.linalg.norm(truth[1:]))}
        print(pose, "truth", json.dumps(entry["truth"]), flush=True)
        out[pose] = entry
    monkeypatch.undo()
    with open(sys.argv[1], "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
