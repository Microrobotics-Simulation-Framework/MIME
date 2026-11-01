"""The rigid body's gyroscopic term keeps a torque-free body's kinetic energy and angular momentum. A body of three
different inertias, nothing to damp it and no load, spinning at about 200 rad/s: Euler's equations conserve both, and
the scheme (the term linearly implicit at the midpoint, ``MIME-ANO-129``) holds each magnitude to a part in a thousand
over a second at a 1 ms step, where the term taken explicitly gained 2.6 times the kinetic energy in the same second,
and the live test rigs (a body in a rotating field, with no drag) went non-finite in about four seconds.

- *The invariants*, for a spin about no principal axis and a spin about the largest axis (``(30, 200, 30)`` with the
  largest moment second: the stable axis, where the body does not flip). The magnitude of the angular momentum, to
  3e-4 at 1 ms; its direction in the world, which is first order in the step and bounded, to 2.1 degrees at 1 ms
  (measured 1.99) and 1.1 at 0.5 ms (0.99).
- *The flip*: a spin started near the middle axis (``(200, 1, 1)`` with the moments ``(1.0, 1.2, 0.8)``) turns the
  body over at the elliptic half period (Landau and Lifshitz section 37, the closed form), each time its middle-axis
  component changes sign; the mean spacing of the flips is within 3e-4 of it at 1 ms (second order: measured 1.1e-4)
  and every one of the nine in three seconds is there.
- *Isotropic bodies* step as the term left out does, bit for bit, at any pose, on the node alone and in a compiled
  graph (script A, a second of it): the Jacobian is formed in the body frame from the differences of the moments, which
  are exactly zero for equal moments. (The first body-frame form, ``[I w]x - [w]x I``, was exactly zero on the node
  alone and not in the graph, where the compiler fuses ``a b - c d`` into the rounding residual of ``c d``.)

The controls: the same node with the term taken explicitly (planted by overriding the step's matrix, which is
``rigid_inertial`` 0.2.0) gains more than half again its kinetic energy in the second on both spins, and its flips
are off by 3.6e-3 of the half period (first order), every run (the draws are fixed, and the scheme has no
randomness); a Jacobian formed from the world inertia, as the first version of the fix did, is not exactly zero for
an isotropic body at a turned pose, and moves its result in the last bits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ellipk

from mime.dynamics.body import RigidInertialExplicit
from mime.dynamics.body._rigid_inertial import RATE_TIMES_STEP
from mime.nodes.body import RigidInertialExplicitNode, RigidInertialNode, gyroscopic_jacobian

INERTIA = (1.0e-10, 1.2e-10, 0.8e-10)
SPINS = {"general": (150.0, 100.0, 100.0), "largest": (30.0, 200.0, 30.0)}
SECOND = 1.0
#: The spin about the middle axis, x, with a perturbation on the other two.
MIDDLE_SPIN = (200.0, 1.0, 1.0)
FLIPS = 9


class ExplicitGyroscope(RigidInertialNode):
    """The planted control: the step's matrix without the gyroscopic Jacobian, the term explicit as it was."""

    def midpoint_mass(self, mass_matrix, gyroscopic, dt):
        return mass_matrix


def body_rotation(pose):
    """The rotation matrix of the unit quaternion in ``pose[3:]`` (w, x, y, z), body to world."""
    w, x, y, z = pose[3:]
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=pose.dtype)


def torque_free(node_class: type[RigidInertialNode], spin: tuple[float, float, float], dt: float, seconds: float):
    """Each step of ``seconds`` of the torque-free body: the kinetic energy, the angular momentum in the world and the
    angular velocity in the body's frame, as scanned arrays."""
    node = node_class("probe", dt, mass=8.5e-5, inertia=INERTIA, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                      initial_twist=(0, 0, 0, *spin))

    def step(state, _):
        state = node.update(state, {}, dt)
        turn = body_rotation(state["pose"])
        in_body = turn.T @ state["twist"][3:]
        momentum = turn @ (jnp.asarray(INERTIA, state["pose"].dtype) * in_body)
        return state, (0.5 * jnp.dot(in_body, jnp.asarray(INERTIA, in_body.dtype) * in_body), momentum, in_body)

    _, found = jax.lax.scan(step, node.initial_state(), None, length=round(seconds / dt))
    return tuple(np.asarray(f, dtype=np.float64) for f in found)


def invariants(node_class: type[RigidInertialNode], spin: tuple[float, float, float], dt: float):
    """The kinetic energy and the angular momentum's magnitude after each step of a second of the torque-free body,
    as ratios of their starting values."""
    energy, momentum, _ = torque_free(node_class, spin, dt, SECOND)
    start = np.asarray(INERTIA) * np.asarray(spin)
    return energy / (0.5 * np.dot(start, spin)), np.linalg.norm(momentum, axis=1) / np.linalg.norm(start)


def direction_drift_degrees(node_class: type[RigidInertialNode], spin: tuple[float, float, float], dt: float) -> float:
    """The most the angular momentum's direction in the world has turned from its start in a second."""
    _, momentum, _ = torque_free(node_class, spin, dt, SECOND)
    start = np.asarray(INERTIA) * np.asarray(spin)
    cosine = momentum @ start / (np.linalg.norm(momentum, axis=1) * np.linalg.norm(start))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).max())


def elliptic_half_period(spin: tuple[float, float, float]) -> float:
    """The closed-form time between the flips of the torque-free body's middle-axis component (Landau and Lifshitz
    section 37): half the period ``4 K(k) / lambda`` of the Jacobi functions that the body-frame rates are, with
    ``lambda`` and ``k`` from twice the energy ``E2`` and the squared angular momentum ``L2`` (the branch on which side
    of ``E2 I2`` the ``L2`` lies)."""
    i1, i2, i3 = np.sort(np.asarray(INERTIA, dtype=np.float64))
    rates = np.asarray(spin, dtype=np.float64)
    e2, l2 = float(np.sum(np.asarray(INERTIA) * rates ** 2)), float(np.sum((np.asarray(INERTIA) * rates) ** 2))
    if l2 > e2 * i2:
        rate = np.sqrt((i3 - i2) * (l2 - e2 * i1) / (i1 * i2 * i3))
        parameter = (i2 - i1) * (e2 * i3 - l2) / ((i3 - i2) * (l2 - e2 * i1))
    else:
        rate = np.sqrt((i1 - i2) * (l2 - e2 * i3) / (i1 * i2 * i3))
        parameter = (i2 - i3) * (e2 * i1 - l2) / ((i1 - i2) * (l2 - e2 * i3))
    return float(2.0 * ellipk(parameter) / rate)


def flip_spacing_error(node_class: type[RigidInertialNode], dt: float, seconds: float = 3.0) -> tuple[int, float]:
    """The number of flips of the middle-axis component (the sign changes of the body-frame x rate, each placed by
    linear interpolation) in ``seconds`` of the torque-free body, and the mean spacing of those flips as a fraction of
    the closed form's half period, less one; infinite where fewer than two flips happen or the run is not finite."""
    _, _, in_body = torque_free(node_class, MIDDLE_SPIN, dt, seconds)
    x = in_body[:, 0]
    if not np.isfinite(x).all():
        return 0, float("inf")
    at = np.nonzero(np.signbit(x[:-1]) != np.signbit(x[1:]))[0]
    times = (at + 1 + x[at] / (x[at] - x[at + 1])) * dt
    if len(times) < 2:
        return len(times), float("inf")
    return len(times), float(np.mean(np.diff(times)) / elliptic_half_period(MIDDLE_SPIN) - 1.0)


@pytest.mark.parametrize("spin", SPINS.values(), ids=SPINS.keys())
@pytest.mark.parametrize("dt", [1e-3, 5e-4])
def test_a_torque_free_body_keeps_its_energy_and_the_magnitude_of_its_angular_momentum(spin, dt):
    energy, momentum = invariants(RigidInertialNode, spin, dt)
    print(f"\nkinetic energy {energy.min():.5f} .. {energy.max():.5f}, "
          f"momentum {momentum.min():.5f} .. {momentum.max():.5f}")
    assert np.all(np.abs(energy - 1.0) < 3e-4) and np.all(np.abs(momentum - 1.0) < 3e-4)


@pytest.mark.parametrize("spin", SPINS.values(), ids=SPINS.keys())
def test_control_the_term_taken_explicitly_gains_energy_on_every_spin(spin):
    explicit, _ = invariants(ExplicitGyroscope, spin, 1e-3)
    kept, _ = invariants(RigidInertialNode, spin, 1e-3)
    print(f"\nexplicit {explicit[-1]:.3f}, kept {kept[-1]:.5f}")
    assert explicit[-1] > 1.5 and abs(kept[-1] - 1.0) < 3e-4


def test_the_direction_of_the_angular_momentum_drifts_first_order_and_bounded():
    """The claim stated as measured, not a property of the scheme the explicit term lacks (it drifts as much): the pose
    turns by the rate at the end of the step, so the direction is first order in the step (halving it halves the drift)
    and bounded (the second of the run below it is the same as the first)."""
    spin = SPINS["general"]
    at_1ms, at_half = direction_drift_degrees(RigidInertialNode, spin, 1e-3), \
        direction_drift_degrees(RigidInertialNode, spin, 5e-4)
    print(f"\ndirection drift {at_1ms:.3f} degrees at 1 ms, {at_half:.3f} at 0.5 ms")
    assert at_1ms < 2.1 and at_half < 1.1 and 1.9 < at_1ms / at_half < 2.1


def test_a_spin_started_near_the_middle_axis_flips_at_the_elliptic_period():
    """Run in float64: the flips' timing is set by how the small components grow from their start, and float32's
    rounding of them moves the spacing by 5e-3 at 1 ms, more than the error being tested."""
    with jax.enable_x64(True):
        flips, error = flip_spacing_error(RigidInertialNode, 1e-3)
        halved_flips, halved = flip_spacing_error(RigidInertialNode, 5e-4)
    print(f"\n{flips} flips, mean spacing off by {error:.2e} of the half period at 1 ms; {halved:.2e} at 0.5 ms")
    assert flips == halved_flips == FLIPS and abs(error) < 3e-4 and abs(halved) < 8e-5


def test_control_the_term_taken_explicitly_flips_at_the_wrong_period():
    with jax.enable_x64(True):
        flips, error = flip_spacing_error(ExplicitGyroscope, 1e-3)
    print(f"\nexplicit: {flips} flips, mean spacing off by {error:.2e}")
    assert not (flips == FLIPS and abs(error) < 3e-4)
    with jax.enable_x64(True):
        kept = flip_spacing_error(RigidInertialNode, 1e-3)
    assert kept[0] == FLIPS and abs(kept[1]) < 3e-4


def world_jacobian(rotation, moments, omega):
    """The Jacobian formed from the world inertia, ``[I w]x - [w]x I`` with ``I = R diag(I_b) R^T``, as the first
    version of the fix did."""
    inertia = rotation @ jnp.diag(moments) @ rotation.T
    return cross_matrix(inertia @ omega) - cross_matrix(omega) @ inertia


def cross_matrix(v):
    zero = jnp.zeros_like(v[0])
    return jnp.stack([jnp.stack([zero, -v[2], v[1]]), jnp.stack([v[2], zero, -v[0]]), jnp.stack([-v[1], v[0], zero])])


class WorldFrameJacobian(RigidInertialNode):
    """The planted control: the Jacobian from the world inertia."""

    def gyroscopic_jacobian(self, rotation, moments, omega):
        return world_jacobian(rotation, moments, omega)


def drawn_poses_and_rates(count: int = 200):
    """Unit quaternions and body rates of a physical scale (rad/s), fixed draws."""
    draws = np.random.default_rng(20261101)
    quaternions = draws.normal(size=(count, 4))
    return (quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True),
            draws.uniform(-300.0, 300.0, size=(count, 3)))


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_the_jacobian_is_exactly_zero_for_equal_moments_and_the_worlds_is_not(x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        found, planted = [], []
        for quaternion, omega in zip(*drawn_poses_and_rates(), strict=True):
            rotation = body_rotation(jnp.asarray((0.0, 0.0, 0.0, *quaternion), dtype=dtype))
            moments, rate = jnp.full(3, 1.7e-10, dtype=dtype), jnp.asarray(omega, dtype=dtype)
            found.append(float(jnp.max(jnp.abs(gyroscopic_jacobian(rotation, moments, rate)))))
            planted.append(float(jnp.max(jnp.abs(world_jacobian(rotation, moments, rate)))))
    print(f"\nbody frame: largest entry {max(found):.1e}; world inertia: smallest of the largest entries "
          f"{min(planted):.1e}")
    assert max(found) == 0.0 and sum(p > 0.0 for p in planted) > 0.9 * len(planted)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_the_jacobian_is_the_derivative_of_the_gyroscopic_term_for_unequal_moments(x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        worst = 0.0
        for quaternion, omega in zip(*drawn_poses_and_rates(20), strict=True):
            rotation = body_rotation(jnp.asarray((0.0, 0.0, 0.0, *quaternion), dtype=dtype))
            moments, rate = jnp.asarray(INERTIA, dtype=dtype), jnp.asarray(omega, dtype=dtype)
            inertia = rotation @ jnp.diag(moments) @ rotation.T
            expected = jax.jacfwd(lambda w, inertia=inertia: -jnp.cross(w, inertia @ w))(rate)
            found = gyroscopic_jacobian(rotation, moments, rate)
            worst = max(worst, float(jnp.max(jnp.abs(found - expected)) / jnp.max(jnp.abs(expected))))
    print(f"\nlargest difference from the derivative, relative to its largest entry: {worst:.1e}")
    assert worst < (1e-14 if x64 else 1e-5)


RESISTANCE = np.diag([1e-5] * 3 + [1e-11] * 3)


def isotropic_run(node_class: type[RigidInertialNode], moment: float, *, steps: int = 3000, dt: float = 2e-3):
    """Pose and twist after every step of an isotropic body, turned and spinning, against a resistance."""
    turn = np.array([0.8, 0.3, -0.4, 0.3])
    unit = turn / np.linalg.norm(turn)
    node = node_class("probe", dt, mass=8.5e-5, inertia=(moment,) * 3, initial_pose=(0, 0, 0, *unit),
                      initial_twist=(0, 0, 0, 150.0, 100.0, -100.0))
    given = {"resistance": jnp.asarray(RESISTANCE, dtype=node.initial_state()["twist"].dtype)}

    def step(state, _):
        state = node.update(state, given, dt)
        return state, (state["pose"], state["twist"])

    _, found = jax.lax.scan(step, node.initial_state(), None, length=steps)
    return tuple(np.asarray(f) for f in found)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("moment", [1e-10, 3.7e-11])
def test_an_isotropic_body_steps_as_the_term_explicit_does_bit_for_bit(x64, moment):
    with jax.enable_x64(x64):
        midpoint, explicit = isotropic_run(RigidInertialNode, moment), isotropic_run(ExplicitGyroscope, moment)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(midpoint, explicit, strict=True))


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("moment", [1e-10, 3.7e-11])
def test_control_the_jacobian_from_the_world_inertia_moves_an_isotropic_body(x64, moment):
    """The world form's error (3e-7 of the term in float32, 4e-16 in float64) is rounded away in ``M - dt K / 2`` for
    most steps; at this step (a rate times the step of 0.4) it first reaches the result at step 360 to 2638, so the
    run is long enough that it does on every run (the draws are fixed)."""
    with jax.enable_x64(x64):
        planted, explicit = isotropic_run(WorldFrameJacobian, moment), isotropic_run(ExplicitGyroscope, moment)
    assert any(a.tobytes() != b.tobytes() for a, b in zip(planted, explicit, strict=True))


def script_a_history(second: float = 1.0) -> bytes:
    """Every inexact field of script A (isotropic, turned, a drag resistance and a field torque) over ``second``, as
    bytes, from a build in this process."""
    from script_a import script_a

    from mime.units import s

    history = script_a().build().run(s(second), history=True).history
    assert history is not None
    fields = [np.ascontiguousarray(history[node][name]) for node in sorted(history) for name in sorted(history[node])]
    return b"".join(field.tobytes() for field in fields if np.issubdtype(field.dtype, np.inexact))


def test_script_a_in_its_compiled_graph_steps_as_the_term_explicit_does_bit_for_bit(monkeypatch):
    current = script_a_history()
    monkeypatch.setattr(RigidInertialNode, "midpoint_mass", lambda self, mass_matrix, gyroscopic, dt: mass_matrix)
    assert current == script_a_history()


def test_control_the_product_difference_form_of_the_jacobian_moves_script_a_in_its_compiled_graph(monkeypatch):
    """``[I w]x - [w]x I`` taken in the body frame, as the fix first had it: exactly zero for equal moments on the node
    alone (the stepping test above passes with it), and a rounding residual in the compiled graph, where the compiler
    contracts the products (an x86-64 or arm64 host with fused multiply-add, as every CI runner is)."""
    def product_difference(self, rotation, moments, omega):
        w = rotation.T @ omega
        return rotation @ (cross_matrix(moments * w) - cross_matrix(w) * moments[None, :]) @ rotation.T

    current = script_a_history()
    monkeypatch.setattr(RigidInertialNode, "gyroscopic_jacobian", product_difference)
    assert current != script_a_history()


def test_the_kept_version_is_the_explicit_term():
    """``rigid_inertial`` 0.2.0, as registered, builds the planted control's step."""
    assert RigidInertialExplicitNode.midpoint_mass is not RigidInertialNode.midpoint_mass
    energy, _ = invariants(RigidInertialExplicitNode, SPINS["general"], 1e-3)
    assert energy[-1] > 1.5


def energy_error_at_rate_times_step(node_class: type[RigidInertialNode], rate_times_step: float) -> float:
    """The most the torque-free body's kinetic energy departs from its start, as a fraction, in at least a second and at
    least fifty steps, with the step set so that the spin's magnitude times it is ``rate_times_step`` (float64)."""
    spin = SPINS["general"]
    dt = rate_times_step / float(np.linalg.norm(spin))
    with jax.enable_x64(True):
        energy, _, _ = torque_free(node_class, spin, dt, max(SECOND, 50 * dt))
    start = 0.5 * float(np.dot(np.asarray(INERTIA) * np.asarray(spin), spin))
    return float(np.max(np.abs(energy / start - 1.0)))


def test_the_rate_times_step_line_holds_the_energy_to_its_stated_error_and_is_exceeded_beyond_it():
    """The line's basis, as measured on this body: at the stated rate times the step the energy holds to 2e-3, and the
    error grows as its square, so that eight times the line it is 0.1 (the control: a line anywhere near this would not
    hold the 2e-3), while a run past it stays finite."""
    line = RATE_TIMES_STEP
    at_line, beyond = energy_error_at_rate_times_step(RigidInertialNode, line), \
        energy_error_at_rate_times_step(RigidInertialNode, 8 * line)
    print(f"\nenergy error {at_line:.2e} at |w| dt = {line}, {beyond:.2e} at {8 * line}")
    assert at_line < 2e-3 and beyond > 5e-2 and 8 < beyond / at_line < 100 and np.isfinite(beyond)


def test_the_limits_the_build_report_prints_state_the_line_and_the_anomaly():
    from mime.dynamics import RigidInertial

    stated = " ".join(RigidInertial.limits)
    assert f"{RATE_TIMES_STEP:g}" in stated and stated.count("MIME-ANO-129") == len(RigidInertial.limits)
    assert "gains" in " ".join(RigidInertialExplicit.limits)
