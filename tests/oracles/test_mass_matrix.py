"""The shared 6 x 6 mass-matrix type (F0-3 item 3, ``mime.core.multibody.MassMatrix``): a rigid body's is
``spatial_inertia(m, 0, J)`` to the bit; an added part sums in; the world form turns it by the body's rotation; the
kinetic energy is ``V^T M V / 2``. **Controls**: construction refuses a matrix that is not 6 x 6, not finite, not
symmetric to its rounding or not positive definite, each by its own words; a semidefinite one (a free spin about a
filament's axis) is refused too, since a body's mass matrix is definite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.core import quaternion
from mime.core.multibody import MassMatrix
from mime.core.multibody._dynamics import spatial_inertia
from mime.core.multibody._mass_matrix import DEFINITE_ULPS, scaled_least_eigenvalue

MASS = 8.5e-5


def physical_tensor(draws: np.random.Generator) -> np.ndarray:
    """A body's inertia tensor at FL-9's scale: distinct principal moments within the triangle inequality, about drawn
    axes."""
    least = draws.uniform(0.5e-10, 1.5e-10)
    middle = least * draws.uniform(1.1, 1.6)
    moments = np.array([least, middle, min(middle * draws.uniform(1.1, 1.6), 0.975 * (least + middle))])
    axes = np.linalg.qr(draws.normal(size=(3, 3)))[0]
    return axes @ np.diag(moments) @ axes.T


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def x64(request):
    with jax.enable_x64(request.param):
        yield request.param


def test_a_rigid_bodys_is_spatial_inertia_about_its_centre_to_the_bit(x64):
    draws = np.random.default_rng(1)
    for _ in range(20):
        tensor = jnp.asarray(physical_tensor(draws))
        held = MassMatrix.rigid(MASS, tensor).matrix
        expected = spatial_inertia(jnp.asarray(MASS, dtype=tensor.dtype), jnp.zeros(3, dtype=tensor.dtype), tensor)
        assert held.dtype == tensor.dtype and np.array_equal(np.asarray(held), np.asarray(expected))


def test_the_world_form_an_added_part_and_the_kinetic_energy(x64):
    draws = np.random.default_rng(2)
    tensor = physical_tensor(draws)
    turn = draws.normal(size=4)
    rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(turn / np.linalg.norm(turn))), dtype=np.float64)
    held = MassMatrix.rigid(MASS, jnp.asarray(tensor))
    eps = float(jnp.finfo(held.matrix.dtype).eps)
    world = np.asarray(held.in_world(rotation), dtype=np.float64)
    assert np.max(np.abs(world[3:, 3:] - rotation @ tensor @ rotation.T)) <= 64 * eps * np.max(np.abs(tensor))
    assert np.max(np.abs(world[:3, :3] - MASS * np.eye(3))) <= 64 * eps * MASS and not np.any(world[:3, 3:])
    added = np.diag([1e-5] * 3 + [1e-11] * 3)
    summed = np.asarray(held.plus(added).matrix, dtype=np.float64)
    assert np.allclose(summed, np.asarray(held.matrix, dtype=np.float64) + added, rtol=4 * eps, atol=0.0)
    twist = np.array([1e-3, -2e-3, 5e-4, 3.0, -1.0, 2.0])
    expected = 0.5 * (MASS * twist[:3] @ twist[:3] + twist[3:] @ tensor @ twist[3:])
    assert float(held.kinetic_energy(twist)) == pytest.approx(expected, rel=64 * eps)


@pytest.mark.parametrize("plant, said", [
    ("shape", "6 x 6"), ("not finite", "finite"), ("asymmetric", "symmetric"), ("indefinite", "positive definite"),
    ("semidefinite", "positive definite"),
])
def test_control_construction_refuses_a_matrix_that_is_not_a_bodys(x64, plant, said):
    good = np.asarray(MassMatrix.rigid(MASS, jnp.asarray(np.diag([1e-10, 2e-10, 2.5e-10]))).matrix, dtype=np.float64)
    bad = good.copy()
    if plant == "shape":
        bad = good[:5, :5]
    elif plant == "not finite":
        bad[2, 2] = np.nan
    elif plant == "asymmetric":
        bad[3, 4] = 1e-12
    elif plant == "indefinite":
        bad[5, 5] = -1e-10
    else:
        bad[5, 5] = 0.0
    with pytest.raises(ValueError, match=said):
        MassMatrix(jnp.asarray(bad))
    MassMatrix(jnp.asarray(good))


def test_the_definiteness_judgement_is_on_the_diagonal_scaled_matrix():
    """A translation's kilograms and a rotation's kilogram square metres differ by thirteen orders: scaled by its
    diagonal, a body's matrix with no coupling has least eigenvalue one, and one whose products leave it barely
    definite stands just above the floor, in either unit."""
    diagonal = np.diag([MASS] * 3 + [1e-10, 2e-10, 2.5e-10])
    assert scaled_least_eigenvalue(diagonal) == pytest.approx(1.0, rel=1e-15)
    barely = diagonal.copy()
    barely[3, 4] = barely[4, 3] = (1.0 - 1e-6) * np.sqrt(1e-10 * 2e-10)
    assert 0.0 < scaled_least_eigenvalue(barely) < 2e-6 and scaled_least_eigenvalue(barely) > DEFINITE_ULPS * 2.3e-16
