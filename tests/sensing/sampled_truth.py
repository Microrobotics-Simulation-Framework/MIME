"""The sign-only truth for route 2's gates (camera truth 1's, on any kind with a sign function): a pixel's coverage is
the share of its jittered sub-pixel rays that meet the body, a ray meeting it where the sign of the kind's sign function
goes negative along it: ``signs`` samples across the body's enclosing ball, the least sample and its neighbours
refined by a parabola (a negative vertex is a hit). It reads no distance, no gradient and no edge measure, so it is
independent of the march it judges; its own floor is its samples a pixel and its sign samples (camera truth 1: under
0.02 a pixel at 2304 samples a pixel and 512 signs, and at four times either)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.core import quaternion
from mime.sensing import CameraPose, Pinhole, Subject
from mime.sensing import _render as render_module


def jittered(side: int, seed: int) -> np.ndarray:
    """``side x side`` stratified sub-pixel offsets in ``[-0.5, 0.5)``, each jittered within its cell."""
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    grid = (np.stack([xx, yy], -1).reshape(-1, 2) + 0.5) / side
    return grid + (np.random.default_rng(seed).random(grid.shape) - 0.5) / side - 0.5


def _hits(subject: Subject, rotation: Any, origin: Any, camera_position: Any, directions: Any, signs: int) -> Any:
    """Whether each world ray from the pinhole along ``directions`` (``(n, 3)``) meets the body."""
    marched = subject.placed.sampled
    assert marched is not None
    p = {name: value if isinstance(value, int) else jnp.asarray(value) for name, value in
         subject.placed.parameters.items()}
    o = tuple(jnp.sum((camera_position - origin) * rotation[:, i]) for i in range(3))
    d = tuple(directions @ rotation[:, i] for i in range(3))
    b = o[0] * d[0] + o[1] * d[1] + o[2] * d[2]
    c = o[0] ** 2 + o[1] ** 2 + o[2] ** 2 - marched.enclosing ** 2
    half = jnp.sqrt(jnp.maximum(b * b - c, 0.0))
    t0, t1 = jnp.maximum(-b - half, 0.0), jnp.maximum(-b + half, 0.0)
    k = jnp.arange(signs, dtype=directions.dtype)
    t = t0[:, None] + (t1 - t0)[:, None] * k / (signs - 1)
    found = marched.kind.sign_function(p, marched.region, tuple(o[i] + t * d[i][:, None] for i in range(3))).value
    at = jnp.argmin(found, axis=1)
    least = jnp.take_along_axis(found, at[:, None], 1)[:, 0]
    before = jnp.take_along_axis(found, jnp.clip(at - 1, 0, signs - 1)[:, None], 1)[:, 0]
    after = jnp.take_along_axis(found, jnp.clip(at + 1, 0, signs - 1)[:, None], 1)[:, 0]
    curvature = before - 2.0 * least + after
    vertex = least - jnp.where(curvature > 0.0, (before - after) ** 2 / (8.0 * jnp.where(curvature > 0, curvature, 1.0)),
                               0.0)
    return (half > 0.0) & (vertex < 0.0)


def truth(subject: Subject, camera: Pinhole, pose: CameraPose, position: Any, orientation: Any, pixels: np.ndarray,
          side: int = 48, signs: int = 512, seed: int = 7, chunk: int = 8192) -> np.ndarray:
    """The coverage of each pixel of ``pixels`` (flat indices) by the body ``subject`` at ``position``,
    ``orientation``: the share of its ``side x side`` jittered sub-pixel rays that meet it (an undistorted pinhole)."""
    rotation, origin = render_module._local(subject, jnp.asarray(position, jnp.float64),
                                            jnp.asarray(orientation, jnp.float64), jnp.float64)
    frame = quaternion.rotation_matrix(jnp.asarray(pose.orientation, jnp.float64))
    camera_position = jnp.asarray(pose.position, jnp.float64)
    offsets = jittered(side, seed)
    u = (pixels % camera.width)[:, None] + offsets[None, :, 0]
    v = (pixels // camera.width)[:, None] + offsets[None, :, 1]
    x, y = (u.reshape(-1) - camera.cx) / camera.fx, (v.reshape(-1) - camera.cy) / camera.fy
    local = np.stack([x, y, np.ones_like(x)], -1)
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    hit = jax.jit(lambda directions: _hits(subject, rotation, origin, camera_position, directions, signs))
    found = np.zeros(len(local), dtype=bool)
    for start in range(0, len(local), chunk):
        part = local[start:start + chunk]
        padded = np.concatenate([part, np.repeat(part[:1], chunk - len(part), 0)]) if len(part) < chunk else part
        found[start:start + len(part)] = np.asarray(hit(jnp.asarray(padded) @ frame.T))[:len(part)]
    return found.reshape(len(pixels), -1).mean(axis=1)
