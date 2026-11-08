"""The chain segments' step identities and the exchange between a rotor and its host, on drawn
states and on the laws alone, each with a control that fails every run.

- **The joint and spinner port kinds' step identities** (``mime.testing.step_identity``, ``PORT_CASES``): a dynamic
  arm of three joints (one bounded, one with Coulomb friction) and a dynamic motor on a host frame, metered for one
  synthetic law on each input they pair and for the laws they hold themselves, close within their floors on every
  draw, both dtypes; work fed into the unattributed input is booked as unattributed. The control: a step that raises
  its end storage by 64 of its own floors, unbooked, fails on the first draw, every run.
- **The rotor's inequality across its axis and its host's carried inertia, one workless law**, settled on the law
  alone: the inequality's torque on the spin is the derivative of the host's kinetic energy ``(1/2) omega^T D(phi)
  omega`` in the rotor's angle, ``(omega x e) . D omega = d/d phi``, within 16 eps of its scale, on drawn axes,
  inequalities, angles and host rates, the scale being the torque's own terms before they cancel. The controls: the
  torque scaled by 0.99 and by 0.9, and its sign flipped, fail on every draw.
- **The spinners' gyroscopic exchange**, workless: the Cayley step keeps ``v^T M v`` within 16 eps of
  ``|v|^T |M| |v|`` for a skew ``G`` from drawn momenta and Jacobians; the control, a symmetric part added to ``G``,
  fails on every draw.
- **The exchange over a run**: on A0's apparatus with its magnet's mass (the arm carrying the rotor's inequality), the
  exchange's paired work on its two receivers with the arm's storage's jump between steps (``EnergyAccount.exchanges``)
  falls at first order in the step."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from ledger_cases import in_dtype, swinging_rotor

from mime.core import ledger, quaternion
from mime.dynamics import Arm, Motor
from mime.nodes.actuation import DynamicArmNode, DynamicMotorNode
from mime.nodes.actuation._spinner import cayley, deviator_torque, gyroscopic_matrix, skew, turned
from mime.testing import PORT_CASES, step_identity
from mime.units import s

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
SEGMENTS = {"joint": Arm, "spinner": Motor}
PROPERTY = settings(max_examples=200, deadline=None, database=None)


@DTYPES
@pytest.mark.parametrize("port", SEGMENTS)
def test_each_segment_s_step_identity_closes_on_every_draw(name, port):
    cls = SEGMENTS[port]
    with in_dtype(name):
        result = step_identity(cls.__new__(cls))
    print(f"\n{result}")
    assert result.passed, result


def gifted(base: type) -> type:
    """``base`` (a segment's node) with a step that raises its end storage by 64 of its own floors, unbooked."""

    class Gifted(base):  # type: ignore[misc, valid-type]
        def identity_terms(self, p: Any, held: Any, state: Any, boundary_inputs: Any, ended: Any, parts: Any,
                           dt: float) -> Any:
            account = super().identity_terms(p, held, state, boundary_inputs, ended, parts, dt)
            floor = ledger.IDENTITY_ULPS * jnp.finfo(account.scale.dtype).eps * account.scale
            return dataclasses.replace(account, storage_end=account.storage_end + 64.0 * floor)

    return Gifted


@DTYPES
@pytest.mark.parametrize("port", SEGMENTS)
def test_a_segment_s_step_that_gives_64_floors_unbooked_fails_its_step_identity_every_run(name, port):
    cls, case = SEGMENTS[port], PORT_CASES[port]
    with in_dtype(name):
        rig: Any = case.rig()
        planted = gifted(type(rig))(rig.name, float(rig.delta_t), **dict(rig.params))
        for _ in range(3):
            result = step_identity(cls.__new__(cls), planted, max_examples=20)
            print(f"\n{result}")
            assert result.failed and "of its floor" in result.detail
            assert result.n_examples <= 2, "Hypothesis stops at the first draw, which fails it, and replays it once"


def _unit(values: Any) -> np.ndarray:
    held = np.asarray(values, dtype=np.float64)
    return held / np.linalg.norm(held)


@st.composite
def exchanges(draw: Any) -> dict[str, np.ndarray]:
    """A rotor's axis and half difference across it (traceless, symmetric, ``D e = 0``, to 1e-4 kg m^2), its angle, its
    host's turn in the world and the host's angular velocity to 10 rad/s, each component drawn 0.1 to 1 of its scale
    with its own sign."""
    def signed(scale: float, size: int) -> np.ndarray:
        magnitudes = np.asarray(draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size)))
        signs = np.asarray(draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size)))
        return scale * magnitudes * signs

    axis = _unit(signed(1.0, 3))
    # Across the axis, at a drawn angle about it from the cross product with the world axis it is least along.
    first = _unit(np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))]))
    angle = draw(st.floats(0.0, 2.0 * np.pi))
    across = np.cos(angle) * first + np.sin(angle) * np.cross(axis, first)
    other = np.cross(axis, across)
    unequal = draw(st.floats(1e-6, 1e-4)) * (np.outer(across, across) - np.outer(other, other))
    return {"axis": axis, "unequal": unequal, "angle": signed(3.0, 1)[0], "host": _unit(signed(1.0, 4)),
            "rate": signed(10.0, 3)}


def _rotor(drawn: dict[str, np.ndarray], angle: Any) -> Any:
    """The rotor's pose at ``angle`` on its stator: the host's turn composed with the turn about the rotor's axis."""
    return quaternion.multiply(jnp.asarray(drawn["host"]), quaternion.exponential(angle * jnp.asarray(drawn["axis"])))


def _deviator(drawn: dict[str, np.ndarray], angle: Any) -> Any:
    """The rotor's half difference in the world at ``angle`` on its stator, as the motor turns it: its base frame's
    ``unequal`` turned by the rotor's pose."""
    return turned(jnp.asarray(drawn["unequal"]), quaternion.rotation_matrix(_rotor(drawn, angle)))


def _exchange_residual(drawn: dict[str, np.ndarray], factor: float) -> tuple[float, float, float]:
    """The inequality's torque on the spin (scaled by ``factor``) less the derivative of the host's kinetic energy in
    the rotor's angle, the scale of the arithmetic they are formed by, and the derivative.

    The scale is the torque's own terms, taken before they cancel. The torque is ``(w x e) . (D w)`` with ``D = R D0
    R^T`` the half difference in the world: ``D``'s entries are sums of products of size ``|R| |D0| |R|^T``, and the
    sums cancel, so each product's rounding is on the uncancelled size, not on the result's. Taking the absolute
    value of every factor, the terms' size is ``(|w| x |e|) . (|R| |D0| |R|^T |w|)``, the cross product with the signs
    of its two products dropped. A scale on the result's entries (``|w|^T |D| |w|``) was 5.6 times too small on the
    draw that flaked: the identity's residual was 18.3 units of eps on that scale on CI and 10.6 here (the rounding
    differs by platform, so the scale must not), and 3.3 and 1.9 units of eps on this one. Over 20,000 draws the
    residual reaches 5.5 units of eps on this scale (its 99.9th percentile 2.4), against the 16 allowed."""
    with jax.enable_x64(True):
        rate = jnp.asarray(drawn["rate"])
        world_axis = quaternion.rotate(jnp.asarray(drawn["host"]), jnp.asarray(drawn["axis"]))
        angle = jnp.asarray(drawn["angle"])
        torque = factor * deviator_torque(_deviator(drawn, angle), world_axis, rate)
        derivative = jax.grad(lambda phi: 0.5 * rate @ _deviator(drawn, phi) @ rate)(angle)
        rotation = jnp.abs(quaternion.rotation_matrix(_rotor(drawn, angle)))
        products = rotation @ jnp.abs(jnp.asarray(drawn["unequal"])) @ rotation.T
        scale = float((jnp.abs(skew(rate)) @ jnp.abs(world_axis)) @ (products @ jnp.abs(rate)))
        return float(torque - derivative), scale, float(derivative)


@PROPERTY
@given(exchanges())
def test_the_rotor_s_inequality_torque_is_the_derivative_of_its_host_s_energy_in_its_angle(drawn):
    residual, scale, _ = _exchange_residual(drawn, 1.0)
    assert abs(residual) <= ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale, (residual, scale)


@pytest.mark.parametrize("factor", [0.99, 0.9, -1.0])
def test_a_scaled_or_flipped_inequality_torque_fails_the_exchange_s_identity_on_every_draw(factor):
    """On every draw with an exchange to see: one whose torque is not zero to rounding (a host turning along the
    rotor's axis, or across it where the inequality's two moments lie, exchanges nothing, which no scale can fail).
    The planted error is 1 %, 10 % and 200 % (the sign) of the torque; a draw is kept when its torque is over 1000
    floors, so the 1 % error is over 10 floors on every kept draw, where the identity's own rounding (5.5 units of eps
    over 20,000 draws, a third of one floor) cannot hide it."""
    failed: list[bool] = []

    @PROPERTY
    @given(exchanges())
    def check(drawn: dict[str, np.ndarray]) -> None:
        residual, scale, derivative = _exchange_residual(drawn, factor)
        floor = ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale
        assume(abs(derivative) > 1e3 * floor)
        failed.append(abs(residual) > floor)

    check()
    assert failed and all(failed), f"{sum(failed)} of {len(failed)}"


@st.composite
def gyroscopic_cases(draw: Any) -> dict[str, np.ndarray]:
    """A three-joint tree's mass matrix (symmetric positive definite, to 1e-2 kg m^2), the angular velocity Jacobians
    of two frames, their spinners' momenta to 1e-3 N m s, the joints' rates to 2 rad/s, each component 0.1 to 1 of
    its scale with its own sign."""
    def signed(scale: float, size: int) -> np.ndarray:
        magnitudes = np.asarray(draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size)))
        signs = np.asarray(draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size)))
        return scale * magnitudes * signs

    entries = signed(1.0, 9).reshape(3, 3)
    mass = 1e-3 * (entries @ entries.T + np.eye(3))
    return {"mass": mass, "jacobians": signed(1.0, 18).reshape(2, 3, 3), "momenta": signed(1e-3, 6).reshape(2, 3),
            "rates": signed(2.0, 3)}


def _cayley_residual(drawn: dict[str, np.ndarray], symmetric: float) -> tuple[float, float]:
    with jax.enable_x64(True):
        mass, rates = jnp.asarray(drawn["mass"]), jnp.asarray(drawn["rates"])
        gyroscopic = gyroscopic_matrix(jnp.asarray(drawn["jacobians"]), jnp.asarray(drawn["momenta"]))
        # A symmetric part, on the mass matrix's scale per second: a damping-like term the Cayley map would book as
        # work, whatever the drawn spinners hand (a skew part that cancels to zero among them, too).
        gyroscopic = gyroscopic + symmetric * jnp.max(jnp.abs(mass)) * jnp.eye(3)
        after = cayley(mass, gyroscopic, rates, 5e-4)
        energy = float(after @ mass @ after - rates @ mass @ rates)
        return energy, float(jnp.abs(rates) @ jnp.abs(mass) @ jnp.abs(rates))


@PROPERTY
@given(gyroscopic_cases())
def test_the_cayley_step_keeps_the_kinetic_energy_for_a_skew_gyroscopic_matrix(drawn):
    change, scale = _cayley_residual(drawn, 0.0)
    assert abs(change) <= ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale, (change, scale)


def test_a_gyroscopic_matrix_with_a_symmetric_part_fails_on_every_draw():
    failed: list[bool] = []

    @PROPERTY
    @given(gyroscopic_cases())
    def check(drawn: dict[str, np.ndarray]) -> None:
        change, scale = _cayley_residual(drawn, 10.0)
        failed.append(abs(change) > ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale)

    check()
    assert failed and all(failed), f"{sum(failed)} of {len(failed)}"


def test_the_rotor_s_exchange_over_a_run_falls_at_first_order_in_the_step():
    """The rotor's inequality and the inertia its host carries, one workless law on two receivers: its paired work on
    both, with the jump of the host's storage between steps, over 0.5 s of a swinging rotor, at dt 1 and 0.5 ms
    (float64): first order, a ratio within 1.6 to 2.4 (measured 2.00 for the jump alone).

    **Its reach**: one rig, no draws. A zeroth-order error in the rotor's inequality torque moves the ratio away from 2
    (a torque 0.9 times as large reads 12.9, a flipped one 1.16: both fail), but the window is wide for a first-order
    claim, and a torque 0.99 times as large reads 2.07 and **passes**. It is a catch of gross errors only; the
    law-alone identity above (the torque scaled by 0.9 or flipped fails on every draw) is the sensitive half."""
    found = {}
    with in_dtype("float64"):
        for dt in (1e-3, 5e-4):
            built = swinging_rotor(dt).build()
            held = {d.law for d in ledger.deliveries(built.graph.get_node("swing.motor").constants())}
            assert "swing.motor.inertia" in held, "the motor is metered, its half of the exchange booked"
            account = built.run(s(0.5)).energy
            assert account is not None
            found[dt] = float(account.exchanges["swing.motor.inertia"])
    ratio = found[1e-3] / found[5e-4]
    print(f"\nthe rotor's exchange over 0.5 s: {found}, ratio {ratio:.3f}")
    assert 1.6 <= ratio <= 2.4, found


@st.composite
def arm_configurations(draw: Any) -> np.ndarray:
    """The three joints' positions, each 0.1 to 1 (a radian, a metre on the slide) with its own sign."""
    magnitudes = np.asarray(draw(st.lists(st.floats(0.1, 1.0), min_size=3, max_size=3)))
    return magnitudes * np.asarray(draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=3, max_size=3)))


@functools.cache
def _weight_functions(factor: float) -> Any:
    """The arm's holding forces and the gradient of its weight's energy scaled by ``factor``, each compiled once."""
    from mime.core.multibody import inverse_dynamics
    from mime.testing import joint_rig

    arm: Any = joint_rig()
    held = arm.constants()

    def hold(q: Any) -> Any:
        base = jnp.asarray(held["placement"], dtype=q.dtype)
        return inverse_dynamics(arm.articulation(held, q.dtype), q, jnp.zeros_like(q), None, arm.weight(held, base),
                                None)

    return jax.jit(hold), jax.jit(jax.grad(lambda q: factor * arm.energy(held, q)))


def _hold_and_gradient(factor: float, drawn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The joint forces that hold the arm still at ``drawn`` under gravity (the recursive Newton-Euler algorithm), and
    the gradient of its weight's energy scaled by ``factor``."""
    with jax.enable_x64(True):
        hold, gradient = _weight_functions(factor)
        q = jnp.asarray(drawn, dtype=jnp.float64)
        return np.asarray(hold(q)), np.asarray(gradient(q))


def _weight_misfit(factor: float, drawn: np.ndarray) -> float:
    """How far the gradient of the weight's energy is from the holding forces, over 16 eps of their arithmetic's scale
    (the arm's weight times its reach, 1.2 kg and 0.6 m)."""
    hold, gradient = _hold_and_gradient(factor, drawn)
    return float(np.max(np.abs(gradient - hold))) / (ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * 1.2
                                                     * 9.80665 * 0.6)


@PROPERTY
@given(arm_configurations())
def test_the_arm_s_weight_is_the_negative_gradient_of_its_energy(drawn):
    """The class of ``<arm>.weight`` (a potential) settled on the law alone: the joint forces that hold the arm still
    are the gradient of ``-sum m_i g . c_i(q)``, within 16 eps of their scale, at drawn configurations."""
    assert _weight_misfit(1.0, drawn) <= 1.0, _weight_misfit(1.0, drawn)


@pytest.mark.parametrize("factor", [0.9, -1.0])
def test_a_weight_scaled_or_flipped_in_its_energy_fails_the_gradient_check_on_every_draw(factor):
    failed: list[bool] = []

    @PROPERTY
    @given(arm_configurations())
    def check(drawn: np.ndarray) -> None:
        failed.append(_weight_misfit(factor, drawn) > 1.0)

    check()
    assert failed and all(failed), f"{sum(failed)} of {len(failed)}"


def test_the_segments_take_the_ledger_and_refuse_a_delivery_they_pair_nothing_for():
    arm: Any = PORT_CASES["joint"].rig()
    wrong = ledger.LedgerDelivery("resistance", "drag:case", ledger.DISSIPATIVE, ledger.SOLVED)
    with pytest.raises(ValueError, match="pairs nothing for"):
        DynamicArmNode(arm.name, float(arm.delta_t), **{**dict(arm.params), "ledger": (wrong.encoded(),)})
    motor: Any = PORT_CASES["spinner"].rig()
    with pytest.raises(ValueError, match="pairs nothing for"):
        DynamicMotorNode(motor.name, float(motor.delta_t), **{**dict(motor.params), "ledger": (wrong.encoded(),)})
