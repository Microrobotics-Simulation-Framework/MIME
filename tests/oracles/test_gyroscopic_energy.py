"""The rigid body's gyroscopic term keeps a torque-free body's kinetic energy and angular momentum. A body of three
different inertias, nothing to damp it and no load, spinning at about 200 rad/s: Euler's equations conserve both, and
the scheme (the term linearly implicit at the midpoint, ``MIME-ANO-130``) holds each to a part in a thousand over a
second at a 1 ms step, where the term taken explicitly gained 2.6 times the kinetic energy in the same second, and
the live test rigs (a body in a rotating field, with no drag) went non-finite in about four seconds.

Two spins: one about no principal axis, and one about the intermediate axis, where the motion is unstable and the
body flips (the invariants still hold). The control: the same node with the term taken explicitly, planted by
overriding the step's matrix, gains more than half again its kinetic energy in the second on both, every run (the
draws are fixed, and the scheme has no randomness).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.body import RigidInertialNode

INERTIA = (1.0e-10, 1.2e-10, 0.8e-10)
SPINS = {"general": (150.0, 100.0, 100.0), "intermediate": (30.0, 200.0, 30.0)}
SECOND = 1.0


class ExplicitGyroscope(RigidInertialNode):
    """The planted control: the step's matrix without the gyroscopic Jacobian, the term explicit as it was."""

    def system(self, mass_matrix, resistance, gyroscopic, dt):
        return mass_matrix / dt + resistance


def invariants(node_class: type[RigidInertialNode], spin: tuple[float, float, float], dt: float):
    """The kinetic energy and the angular momentum's magnitude after each step of a second of the torque-free body,
    as ratios of their starting values."""
    node = node_class("probe", dt, mass=8.5e-5, inertia=INERTIA, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                      initial_twist=(0, 0, 0, *spin))
    inertia = np.diag(INERTIA)

    def step(state, _):
        state = node.update(state, {}, dt)
        w, x, y, z = state["pose"][3:]
        turn = jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=state["pose"].dtype)
        momentum = turn @ jnp.asarray(inertia, state["pose"].dtype) @ turn.T @ state["twist"][3:]
        return state, (0.5 * jnp.dot(state["twist"][3:], momentum), jnp.linalg.norm(momentum))

    _, (energy, momentum) = jax.lax.scan(step, node.initial_state(), None, length=round(SECOND / dt))
    start = np.asarray(INERTIA) * np.asarray(spin)
    return np.asarray(energy) / (0.5 * np.dot(start, spin)), np.asarray(momentum) / np.linalg.norm(start)


@pytest.mark.parametrize("spin", SPINS.values(), ids=SPINS.keys())
@pytest.mark.parametrize("dt", [1e-3, 5e-4])
def test_a_torque_free_body_keeps_its_energy_and_angular_momentum(spin, dt):
    energy, momentum = invariants(RigidInertialNode, spin, dt)
    print(f"\nkinetic energy {energy.min():.5f} .. {energy.max():.5f}, momentum {momentum.min():.5f} .. {momentum.max():.5f}")
    assert np.all(np.abs(energy - 1.0) < 2e-3) and np.all(np.abs(momentum - 1.0) < 2e-3)


@pytest.mark.parametrize("spin", SPINS.values(), ids=SPINS.keys())
def test_control_the_term_taken_explicitly_gains_energy_on_every_spin(spin):
    explicit, _ = invariants(ExplicitGyroscope, spin, 1e-3)
    kept, _ = invariants(RigidInertialNode, spin, 1e-3)
    print(f"\nexplicit {explicit[-1]:.3f}, kept {kept[-1]:.5f}")
    assert explicit[-1] > 1.5 and abs(kept[-1] - 1.0) < 2e-3
