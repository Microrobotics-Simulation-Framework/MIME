"""The MRS rung's separated quadrature written again in float64 NumPy, as a second implementation of its formula: the
kind's force points and its quadrature ``refinement`` times finer placed by a pose, each quadrature point in the cell
of the force point the map names, the regularised Stokeslet of every force point against every quadrature point summed
over each cell (``numpy.add.at``), each cell's wrench from its area and centroid, and the dense solve. The map is the
one ``mime.scene.quadrature_owners`` makes, which ``test_mrs_separated.py`` holds to a brute-force nearest search; the
rest shares no code with the node but, with a wall, the wall's correction itself (the node's, read at the force
points the oracle places). It bounds how far the node departs from its formula, not how far the formula departs from
the flow."""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from mime.scene import quadrature_counts, quadrature_owners


def rotation(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def motions_of(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    arm = points - centre
    return np.stack([*(np.broadcast_to(e, arm.shape) for e in np.eye(3)), *(np.cross(e, arm) for e in np.eye(3))],
                    axis=-1).reshape(-1, 6)


def discretisation(kind: Any, settings: Mapping[str, Any], spacing: float, refinement: int,
                   patches: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The force points, the quadrature's points and weights, in the kind's frame, and the map."""
    patches = tuple(patches)
    sampling = kind.sampling(settings, spacing, patches)
    force = np.asarray(sampling.samples(kind, settings).points, dtype=np.float64).reshape(-1, 3)
    counts = quadrature_counts(kind, settings, spacing / refinement, patches)
    owners = quadrature_owners(kind, settings, sampling, counts)
    found = [kind.samples(settings, patch, count) for patch, count in zip(patches, counts, strict=True)]
    points = np.concatenate([np.asarray(f.points, dtype=np.float64).reshape(-1, 3) for f in found])
    weights = np.concatenate([np.asarray(f.weights, dtype=np.float64).reshape(-1) for f in found])
    return force, points, weights, owners


def resistance(force: np.ndarray, points: np.ndarray, weights: np.ndarray, owners: np.ndarray, blob: float,
               viscosity: float, pose: Sequence[float],
               wall: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """The symmetric part of ``W^T A^-1 M`` about the pose's position, the body at the pose; with ``wall`` (the
    wall's ``3N x 3N`` correction at the force points in the world, per unit force, the viscosity in it), each cell's
    column of it times the cell's area added to ``A``: the wall's image at the force points, as the node takes it."""
    pose = np.asarray(pose, dtype=np.float64)
    turn = rotation(pose[3:])
    x = pose[:3] + force @ turn.T
    y = pose[:3] + points @ turn.T
    count = len(x)
    area = np.bincount(owners, weights=weights, minlength=count)
    centroid = np.stack([np.bincount(owners, weights=weights * y[:, i], minlength=count) for i in range(3)], 1) \
        / area[:, None]
    eps_squared = blob * blob * area.sum() / count
    cells = np.zeros((count, count, 3, 3))
    for start in range(0, len(y), 4096):
        r = x[:, None, :] - y[None, start:start + 4096, :]
        r2 = np.sum(r * r, axis=-1)
        s = weights[None, start:start + 4096] / (8 * math.pi * viscosity * (r2 + eps_squared) ** 1.5)
        pair = s[..., None, None] * (np.eye(3) * (r2 + 2 * eps_squared)[..., None, None]
                                     + r[..., :, None] * r[..., None, :])
        np.add.at(cells, (slice(None), owners[start:start + 4096]), pair)
    matrix = cells.transpose(0, 2, 1, 3).reshape(3 * count, 3 * count)
    if wall is not None:
        matrix = matrix + np.asarray(wall(x), dtype=np.float64) * np.repeat(area, 3)[None, :]
    wrench = motions_of(centroid, pose[:3]) * np.repeat(area, 3)[:, None]
    found = wrench.T @ np.asarray(jnp.linalg.solve(jnp.asarray(matrix), jnp.asarray(motions_of(x, pose[:3]))))
    return (found + found.T) / 2
