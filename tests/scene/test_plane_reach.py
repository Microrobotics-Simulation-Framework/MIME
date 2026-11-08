"""The plane's reach (``Plane.reach``, the support function of the wall's half-space behind it): along the normal it
reaches the plane, ``|d| offset``, and along any other direction without bound. The normal is a direction given to
rounding, so the reach along it is read in a cone about the normal (``REACH_CONE`` machine epsilons across), not at an
exact float zero: for a unit normal tilted off the axes the direction ``d = n`` has a rounding-sized component across
the normal, and the exact test answered infinity for most of them (1608 of 2000 random normals, measured by the
review of the kind's first pull request). Float64 and float32, a normal given at any length."""

import jax.numpy as jnp
import numpy as np
import pytest

from mime.scene import Plane
from mime.scene._geometry import REACH_CONE
from mime.units import m

OFFSET = 0.3
DRAWS = 2000


def normals(seed: int) -> np.ndarray:
    """Random unit normals, drawn in all directions (seeded: the draw is the same every run)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(DRAWS, 3))
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def reach(normal: np.ndarray, direction: np.ndarray, dtype) -> float:
    plane = Plane(normal=tuple(float(c) for c in normal), offset=m(OFFSET))
    parameters = {"normal": jnp.asarray(normal, dtype=dtype), "offset": jnp.asarray(OFFSET, dtype=dtype)}
    return float(plane.reach(parameters, jnp.asarray(direction, dtype=dtype)))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["float64", "float32"])
def test_along_a_tilted_normal_the_reach_is_the_offset_times_the_direction_length(dtype):
    rel = 1e-12 if dtype == jnp.float64 else 1e-5
    for n in normals(11):
        # the direction as the dtype holds it: the normal of the plane the node holds, and a longer one
        held = np.asarray(jnp.asarray(n, dtype=dtype), dtype=np.float64)
        for scale in (1.0, 2.5):
            found = reach(n, scale * held, dtype)
            assert found == pytest.approx(scale * OFFSET, rel=rel), (n, scale, found)


def test_control_the_exact_test_was_wrong_for_most_tilted_normals():
    """The test above can fail: the old predicate, written again (the component across the normal exactly zero), is
    false for a large share of these draws in float32, which is what the cone is for."""
    missed = 0
    for n in normals(11):
        held = jnp.asarray(n, dtype=jnp.float32)
        unit = held / jnp.sqrt(jnp.sum(held * held))
        across = held - jnp.dot(held, unit) * unit
        missed += int(float(jnp.sum(across * across)) != 0.0)
    assert missed > DRAWS // 4, missed


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["float64", "float32"])
def test_off_the_normal_the_reach_is_unbounded(dtype):
    """A direction off the normal by a milliradian (a thousand times the cone), opposite it, and across it: infinity."""
    rng = np.random.default_rng(5)
    for n in normals(7)[:200]:
        other = np.cross(n, rng.normal(size=3))
        other /= np.linalg.norm(other)
        for direction in (n + 1e-3 * other, -n, other):
            assert reach(n, direction / np.linalg.norm(direction), dtype) == np.inf, (n, direction)
    assert REACH_CONE * np.finfo(np.float32).eps < 1e-3 / 100


def test_a_normal_given_at_any_length_is_the_same_plane():
    n = np.array([0.0, 0.6, 0.8])
    for length in (0.5, 1.0, 2.7):
        plane = Plane(normal=tuple(float(c) for c in length * n), offset=m(OFFSET))
        parameters = {"normal": jnp.asarray(length * n), "offset": jnp.asarray(OFFSET)}
        assert float(plane.reach(parameters, jnp.asarray(n))) == pytest.approx(OFFSET, rel=1e-12)
