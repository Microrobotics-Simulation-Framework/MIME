"""The rigid-body library against the 0.2 port's (``mime.control.kinematics``) on the AR4, the weakest oracle. The
port's library ran once, in the port's own environment, at 16 random states inside the joint limits
(``generate_port_multibody_references.py``; ``references/port_multibody.npz`` and its provenance: MIME-port041 at
6485516, float64, CPU); ``mime.core.multibody`` runs here on the same states. The port's ``mime`` is never imported
beside this one.

**Declared before the first run.** Both read the same file (its SHA-256 is checked against the provenance) and
compute the same quantities by different sequences of operations: the port composes 4 x 4 matrices, this library
quaternions. So every quantity agrees within 64 float64 roundings of its largest magnitude:

- each moving link's frame (the port's joint frames): position and rotation matrix;
- each moving link's frame Jacobian (the port's ``ee_jacobian`` with no offset, at the link frame's origin);
- each moving link's centre of mass;
- the mass matrix, and inverse dynamics at a velocity and acceleration under gravity.

No difference is allowed: the port merges the fixed links into their moving ones as this library welds them, and the
AR4 declares no armature, which the port does not model. The control: a planted mutation (one link's mass off by a
part in a million; one joint's axis reversed) is caught.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import multibody_cases as cases
from mime.core import quaternion
from mime.core.multibody import Articulation, MultibodyTree, forward_kinematics, frame_jacobian, inverse_dynamics, \
    mass_matrix

REFERENCES = Path(__file__).parent / "references"
PORT = np.load(REFERENCES / "port_multibody.npz")
PROVENANCE = json.loads((REFERENCES / "port_multibody_provenance.json").read_text())
TREE = cases.ar4()
LINKS = tuple(PROVENANCE["links"])
ROUNDINGS = 64
TOLERANCE = ROUNDINGS * float(np.finfo(np.float64).eps)


def moved(found: np.ndarray, port: np.ndarray) -> float:
    """The largest difference, relative to the port's largest magnitude in the quantity."""
    return float(np.max(np.abs(np.asarray(found) - port))) / float(np.max(np.abs(port)))


QUANTITIES = ("position", "rotation", "centre_of_mass", "jacobian", "mass_matrix", "inverse_dynamics")


def outputs(model: MultibodyTree | Articulation, quantities: tuple[str, ...] = QUANTITIES) -> dict[str, np.ndarray]:
    """The quantities asked for at every reference state, compiled once with the model's numbers as constants."""
    links = jnp.asarray([TREE.frame_names.index(link) for link in LINKS])
    articulation = model.articulation if isinstance(model, MultibodyTree) else model

    def at(q, qd, qdd, gravity):
        poses = forward_kinematics(model, q)[links]
        turned = jax.vmap(lambda p: quaternion.rotation_matrix(p[3:]))(poses)
        every = {"position": lambda: poses[:, :3], "rotation": lambda: turned,
                 "centre_of_mass": lambda: poses[:, :3] + jnp.einsum("bij,bj->bi", turned,
                                                                     jnp.asarray(articulation.centres_of_mass)),
                 "jacobian": lambda: jnp.stack([frame_jacobian(model, q, link) for link in LINKS]),
                 "mass_matrix": lambda: mass_matrix(model, q),
                 "inverse_dynamics": lambda: inverse_dynamics(model, q, qd, qdd, gravity)}
        return {quantity: every[quantity]() for quantity in quantities}

    run = jax.jit(at)
    found = [run(*(jnp.asarray(x) for x in (q, qd, qdd, PORT["gravity"])))
             for q, qd, qdd in zip(PORT["q"], PORT["qd"], PORT["qdd"], strict=True)]
    return {key: np.stack([np.asarray(f[key]) for f in found]) for key in quantities}


def port() -> dict[str, np.ndarray]:
    world = PORT["joint_world"]
    return {"position": world[:, :, :3, 3], "rotation": world[:, :, :3, :3],
            "centre_of_mass": PORT["centre_of_mass_world"], "jacobian": PORT["jacobian"],
            "mass_matrix": PORT["mass_matrix"], "inverse_dynamics": PORT["inverse_dynamics"]}


def test_both_read_the_same_file_at_the_same_states():
    assert hashlib.sha256(cases.AR4_URDF.read_bytes()).hexdigest() == PROVENANCE["urdf_sha256"]
    assert TREE.articulation.dof_names == tuple(PROVENANCE["joints"])
    lower, upper = TREE.articulation.lower, TREE.articulation.upper
    assert np.all((PORT["q"] >= lower) & (PORT["q"] <= upper))


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_each_quantity_agrees_with_the_ports(quantity):
    misfit = moved(outputs(TREE, (quantity,))[quantity], port()[quantity])
    print(f"\n{quantity}: {misfit / float(np.finfo(np.float64).eps):.1f} roundings")
    assert misfit <= TOLERANCE


def test_a_planted_mutation_is_caught():
    """One link's mass off by a part in a million moves the mass matrix; one joint's axis reversed moves the link
    frames' rotations. Each compared alone, so that each compiles only what it compares."""
    articulation = TREE.articulation
    heavier = np.array(articulation.masses)
    heavier[2] *= 1 + 1e-6
    reversed_axes = np.array(articulation.axes)
    reversed_axes[3] *= -1
    for planted, caught in ((dataclasses.replace(articulation, masses=heavier), "mass_matrix"),
                            (dataclasses.replace(articulation, axes=reversed_axes), "rotation")):
        assert moved(outputs(planted, (caught,))[caught], port()[caught]) > TOLERANCE
