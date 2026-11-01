"""``Drag.MRS`` passes the hydrodynamic family's whole suite: sign, frame, units, material reach, oracle, gradients,
swap and MADDENING's battery on its node. The family's own rig fills a tube, which the free-space wall refuses, so the
suite runs on the same family with a rig of its own: the family's screw (de Jongh's FL-9), coarsely sampled, in an
unbounded medium (labelled in the pull request: a law the family holds whose rig it cannot stand in). A ball would not
do: its resistance couples no translation to rotation, so that block is rounding alone, and the suite judges each block
on its own scale, where turning the world or scaling the viscosity moves rounding by more than its own size. The
suite's oracle is the rung's formula again, in float64 NumPy (the kind's samples placed by the pose, the regularised Stokeslet's matrix and the rigid motions), its
dense solve LAPACK's through ``jax.numpy`` (NumPy's, threaded, took a hundred times as long on a loaded machine):
a second implementation, which bounds how far the node departs from its formula, not how far the formula departs from
the flow (``test_mrs.py`` holds that against closed forms). The swap replaces it with the Stokes sphere of the same
radius.

The family walk holds the drag entry to its surrogate rung, the first round-trip case's: this is the mrs rung's half
(``docs/ratchets/builds_unchecked.txt``)."""

import dataclasses
import math
from functools import cache

import jax.numpy as jnp
import numpy as np
from mrs_cases import mrs, free_space
from hydro_oracles import StokesSphere

from mime.testing import NodeCase, assert_conforms, unit_quaternion
from mime.testing._mime_families import HYDRODYNAMIC, PROBE, SCREW
from mime.units import mm

SPACING = mm(1.5)
SETTINGS = {"radius": SCREW.radius.value, "length": SCREW.length.value, "wavenumber": SCREW.wavenumber.value,
            "modulation": SCREW.modulation.value}


def free_rig(law):
    return free_space("hydrodynamic_free_rig", {PROBE: SCREW}, law)


FREE = dataclasses.replace(HYDRODYNAMIC, rig=free_rig)


def rotation(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@cache
def screw_samples() -> tuple[np.ndarray, np.ndarray]:
    """The screw's samples in its own frame, on its closed surface, at the rig's spacing: made once, in the float64
    the module's tests run in."""
    held = [SCREW.samples(SETTINGS, patch, SCREW.sample_counts(SETTINGS, SPACING.value, patch))
            for patch in ("lateral", "caps")]
    return (np.concatenate([np.asarray(h.points, dtype=np.float64) for h in held]),
            np.concatenate([np.asarray(h.weights, dtype=np.float64) for h in held]))


def mrs_reference(read, settings) -> dict[str, np.ndarray]:
    """The rung's resistance again, in float64 NumPy, at the drawn pose."""
    pose = np.asarray(read["pose"], dtype=np.float64)
    held, weights = screw_samples()
    points = pose[:3] + held @ rotation(pose[3:]).T
    eps_squared = 0.25 * weights.sum() / len(weights)
    r = points[:, None, :] - points[None, :, :]
    r2 = np.sum(r * r, axis=-1)
    blocks = (np.eye(3) * (r2 + 2 * eps_squared)[..., None, None] + r[..., :, None] * r[..., None, :]) \
        / (8 * math.pi * settings["viscosity"] * (r2 + eps_squared) ** 1.5)[..., None, None]
    matrix = blocks.transpose(0, 2, 1, 3).reshape(3 * len(points), 3 * len(points))
    arm = points - pose[:3]
    motions = np.stack([*(np.broadcast_to(e, arm.shape) for e in np.eye(3)), *(np.cross(e, arm) for e in np.eye(3))],
                       axis=-1).reshape(-1, 6)
    found = motions.T @ np.asarray(jnp.linalg.solve(jnp.asarray(matrix), jnp.asarray(motions)))
    return {"resistance": (found + found.T) / 2, "background_load": np.zeros(6)}


NODE = NodeCase(options={"boundary_bounds": {"pose": (-4e-3, 4e-3)},
                         "constrain_boundary": lambda given: {**given, "pose": unit_quaternion(given["pose"])}})


def test_the_mrs_rung_conforms_to_the_hydrodynamic_family_in_free_space_and_swaps_with_the_stokes_sphere():
    assert_conforms(mrs(SPACING), FREE, oracle=mrs_reference, swap_with=StokesSphere(), node_case=NODE)
