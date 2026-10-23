"""The quaternion helpers: the rotation's exponential is finite with finite derivatives at a zero rotation, where 0.2's
update divided by ``|omega|``, and agrees with its closed form elsewhere; rotations compose and invert. And the body at
rest: its step and its derivative are finite there.

The control: 0.2's increment, copied here, gives a NaN derivative at a zero rotation, and so does a body stepped with
it."""

from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.core import quaternion
from mime.nodes.body import RigidInertialNode


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def increment_of_0_2(omega: Any, dt: float) -> Any:
    """0.2's ``quat_from_angular_velocity`` (``MIME-port041/src/mime/core/quaternion.py:77``), unchanged."""
    half_angle = jnp.linalg.norm(omega) * dt / 2.0
    axis = omega / jnp.maximum(jnp.linalg.norm(omega), 1e-30)
    return jnp.concatenate([jnp.cos(half_angle)[None], axis * jnp.sin(half_angle)])


def test_the_exponential_at_rest_is_the_identity_with_a_finite_derivative():
    assert np.array_equal(np.asarray(quaternion.exponential(jnp.zeros(3))), [1.0, 0.0, 0.0, 0.0])
    jacobian = np.asarray(jax.jacfwd(quaternion.exponential)(jnp.zeros(3)))
    assert np.array_equal(jacobian, np.vstack([np.zeros(3), 0.5 * np.eye(3)]))
    assert np.all(np.isfinite(np.asarray(jax.grad(lambda w: jnp.sum(quaternion.exponential(w)))(jnp.zeros(3)))))


def test_the_control_0_2s_increment_has_a_nan_derivative_at_rest():
    gradient = np.asarray(jax.grad(lambda w: jnp.sum(increment_of_0_2(w, 1e-3)))(jnp.zeros(3)))
    assert not np.all(np.isfinite(gradient))


@pytest.mark.parametrize("angle", [1e-6, 1e-4 * (1 - 1e-9), 1e-4 * (1 + 1e-9), 0.3, 2.5, 6.0])
def test_the_exponential_is_the_closed_form_on_both_sides_of_its_series(angle):
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    found = np.asarray(quaternion.exponential(jnp.asarray(angle * axis)))
    expected = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    assert np.allclose(found, expected, rtol=0.0, atol=2e-16)
    assert abs(float(np.linalg.norm(found)) - 1.0) < 1e-15


def test_rotations_compose_and_invert():
    p, q = quaternion.exponential(jnp.asarray([0.4, -1.1, 0.2])), quaternion.exponential(jnp.asarray([-0.7, 0.3, 2.0]))
    r_p, r_q = np.asarray(quaternion.rotation_matrix(p)), np.asarray(quaternion.rotation_matrix(q))
    assert np.allclose(np.asarray(quaternion.rotation_matrix(quaternion.multiply(p, q))), r_p @ r_q, atol=1e-15)
    assert np.allclose(r_p @ r_p.T, np.eye(3), atol=1e-15)
    v = jnp.asarray([1.0, -2.0, 0.5])
    assert np.allclose(np.asarray(quaternion.rotate_back(p, quaternion.rotate(p, v))), np.asarray(v), atol=1e-15)
    assert np.allclose(np.asarray(quaternion.rotate(quaternion.conjugate(p), v)),
                       np.asarray(quaternion.rotate_back(p, v)), atol=1e-15)


class _RigidBodyOf0_2(RigidInertialNode):
    """The body turned by 0.2's increment instead."""

    @override
    def advance(self, p, state, boundary_inputs, dt):
        out = super().advance(p, state, boundary_inputs, dt)
        omega = out["twist"][3:]
        turned = quaternion.normalized(quaternion.multiply(increment_of_0_2(omega, dt), state["pose"][3:]))
        return {**out, "pose": jnp.concatenate([out["pose"][:3], turned])}


@pytest.mark.parametrize(("cls", "finite"), [(RigidInertialNode, True), (_RigidBodyOf0_2, False)],
                         ids=["rigid_inertial", "control: 0.2's increment"])
def test_the_body_at_rest_is_finite_with_finite_derivatives(cls, finite):
    node = cls("body", 5e-4, mass=8.5e-5, inertia=(1e-10,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
               initial_twist=(0.0,) * 6)
    rest = node.initial_state()

    def stepped(state):
        out = node.update(state, {}, 5e-4)
        return jnp.sum(out["pose"]) + jnp.sum(out["twist"])

    assert np.all(np.isfinite(np.asarray(node.update(rest, {}, 5e-4)["pose"])))
    gradient = jax.grad(stepped)(rest)
    assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in gradient.values()) is finite
