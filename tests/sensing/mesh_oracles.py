"""The static mesh gates' oracles as functions of a planted error's size and a draw, for the gate kit's
``discriminating_range.py`` (``function(size, seed) -> bool``, True where the oracle flags the plant): the range each gate
discriminates. Size zero is the clean call.

- :func:`chord_error`: G22's chord comparison against brute force, a chord off by ``size`` metres on one judged ray of a
  drawn scene (float32, the coarser dtype).
- :func:`shifted_mesh`: G23's sagitta bracket, the traced tube moved ``size`` pixels across the line of sight, the camera
  placed at a drawn sub-pixel offset (float32, S = 4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from mesh_cases import brute, placed, rotation, tube
from test_mesh_bvh import MARGIN, RAYS, rays
from test_mesh_swap import ORIGIN, tube_bounds

from mime.sensing import CameraPose, Pinhole, ray_table
from mime.sensing._mesh import KEYS, mesh_part, traced, tree, triangles_of
from mime.sensing._subjects import Appearance

_traced = jax.jit(traced, static_argnames=("meshes",))
ALONG_X = rotation((2 ** -0.5, 0.0, 2 ** -0.5, 0.0))
LOOK_ACROSS = (2 ** -0.5, -(2 ** -0.5), 0.0, 0.0)


def chord_error(size: float, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    radius = rng.uniform(50e-6, 1e-3)
    points, faces = tube(radius * rng.uniform(0.5, 0.9), radius, 4 * radius, 32)
    points = placed(points, rng.uniform(-1e-3, 1e-3, 3), ALONG_X)
    triangles = triangles_of(points, faces)
    owners = np.zeros(len(faces), dtype=np.int64)
    origin = rng.uniform(-5e-3, 5e-3, 3)
    directions = rays(triangles, origin, seed)
    run = np.float32
    oracle = brute(triangles.astype(run).astype(np.float64), owners, 1, origin.astype(run).astype(np.float64),
                   directions.astype(run).astype(np.float64))
    judged = (oracle.margin >= MARGIN["float32"]) & (oracle.count[:, 0] > 0)
    built = tree(triangles, owners)
    found = _traced(tuple(jnp.asarray(built.arrays[key], dtype=run) for key in KEYS), meshes=1,
                    origin=jnp.asarray(origin, dtype=run), directions=jnp.asarray(directions, dtype=run),
                    live=jnp.ones(RAYS, dtype=bool))
    chord = np.asarray(found.chord, dtype=np.float64)[:, 0]
    target = int(np.flatnonzero(judged)[0])
    chord[target] += size
    allowed = 64.0 * float(np.finfo(run).eps) * oracle.reach
    return bool(np.any((np.abs(chord - oracle.chord[:, 0]) > allowed)[judged]))


def shifted_mesh(size: float, seed: int) -> bool:
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(seed)
    fx = 750.0
    footprint = 1e-3 / fx
    camera = Pinhole(160, 120, fx, fx, 79.5, 59.5)
    offset = np.array([rng.uniform(0.0, footprint), 0.0, rng.uniform(0.0, footprint)])
    pose = CameraPose(ORIGIN + offset, np.asarray(LOOK_ACROSS))
    points, faces = tube(60e-6, 70e-6, 480e-6, 64)
    world = placed(points, np.array([0.0, 0.0, size * footprint]), ALONG_X)
    built = tree(triangles_of(world, faces), np.zeros(len(faces), dtype=np.int64))
    table = ray_table(camera, pose)
    arrays = tuple(jnp.asarray(built.arrays[key], dtype=jnp.float32) for key in KEYS)
    image = np.asarray(mesh_part((Appearance.attenuating(3000.0),), arrays, table, 4), dtype=np.float64)
    d = np.stack([np.asarray(c, dtype=np.float64) for c in table.direction], axis=-1)
    du = np.stack([np.asarray(c, dtype=np.float64) for c in table.along_u], axis=-1)
    dv = np.stack([np.asarray(c, dtype=np.float64) for c in table.along_v], axis=-1)
    steps = [(i + 0.5) / 4 - 0.5 for i in range(4)]
    sub = np.stack([d + su * du + sv * dv for sv in steps for su in steps], axis=2)
    sub /= np.linalg.norm(sub, axis=-1, keepdims=True)
    import test_mesh_swap

    held = test_mesh_swap.ORIGIN
    test_mesh_swap.ORIGIN = ORIGIN + offset
    try:
        _, low, high = tube_bounds(sub, 64)
    finally:
        test_mesh_swap.ORIGIN = held
    least = np.exp(-3000.0 * high).mean(axis=2)
    most = np.exp(-3000.0 * low).mean(axis=2)
    slack = 64.0 * float(np.finfo(np.float32).eps)
    return bool(np.max(np.maximum(least - slack - image, image - most - slack)) > 0.0)


__all__ = ["chord_error", "shifted_mesh"]
