"""The chain segments' step identities and the exchange between a rotor and its host (F0-1 M6.2, M6.4), on drawn
states and on the laws alone, each with a control that fails every run.

- **The joint and spinner port kinds' step identities** (``mime.testing.step_identity``, ``PORT_CASES``): a dynamic
  arm of three joints (one bounded, one with Coulomb friction) and a dynamic motor on a host frame, metered for one
  synthetic law on each input they pair and for the laws they hold themselves, close within their floors on every
  draw, both dtypes; work fed into the unattributed input is booked as unattributed. The control: a step that raises
  its end storage by 64 of its own floors, unbooked, fails on the first draw, every run.
- **The rotor's inequality across its axis and its host's carried inertia, one workless law**, settled on the law
  alone: the inequality's torque on the spin is the derivative of the host's kinetic energy ``(1/2) omega^T D(phi)
  omega`` in the rotor's angle, ``(omega x e) . D omega = d/d phi``, within 16 eps of its scale, on drawn axes,
  inequalities, angles and host rates. The controls: the torque scaled by 0.9 and its sign flipped fail on every draw.
- **The spinners' gyroscopic exchange**, workless: the Cayley step keeps ``v^T M v`` within 16 eps of
  ``|v|^T |M| |v|`` for a skew ``G`` from drawn momenta and Jacobians; the control, a symmetric part added to ``G``,
  fails on every draw.
- **The exchange over a run**: on A0's apparatus with its magnet's mass (the arm carrying the rotor's inequality), the
  exchange's paired work on its two receivers with the arm's storage's jump between steps (``EnergyAccount.exchanges``)
  falls at first order in the step."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import chain_cases
import pytest
from chain_cases import massive_rpm
from hypothesis import given, settings
from hypothesis import strategies as st
from ledger_cases import in_dtype

from mime.core import ledger, quaternion
from mime.dynamics import Arm, Motor
from mime.nodes.actuation import DynamicArmNode, DynamicMotorNode
from mime.nodes.actuation._spinner import cayley, deviator_torque, gyroscopic_matrix, turned
from mime.experiment import Experiment
from mime.observations import EnergyLedger
from mime.scene import Scene, Z
from mime.testing import PORT_CASES, step_identity
from mime.units import ms, s, standard_gravity

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
    across = _unit(np.cross(axis, signed(1.0, 3)))
    other = np.cross(axis, across)
    unequal = draw(st.floats(1e-6, 1e-4)) * (np.outer(across, across) - np.outer(other, other))
    return {"axis": axis, "unequal": unequal, "angle": signed(3.0, 1)[0], "host": _unit(signed(1.0, 4)),
            "rate": signed(10.0, 3)}


def _deviator(drawn: dict[str, np.ndarray], angle: Any) -> Any:
    """The rotor's half difference in the world at ``angle`` on its stator, as the motor turns it: its base frame's
    ``unequal`` turned by the rotor's pose, the host's turn composed with the turn about its axis."""
    rotor = quaternion.multiply(jnp.asarray(drawn["host"]), quaternion.exponential(angle * jnp.asarray(drawn["axis"])))
    return turned(jnp.asarray(drawn["unequal"]), quaternion.rotation_matrix(rotor))


def _exchange_residual(drawn: dict[str, np.ndarray], factor: float) -> tuple[float, float]:
    """The inequality's torque on the spin (scaled by ``factor``) less the derivative of the host's kinetic energy in
    the rotor's angle, and the scale of the arithmetic they are formed by."""
    with jax.enable_x64(True):
        rate = jnp.asarray(drawn["rate"])
        world_axis = quaternion.rotate(jnp.asarray(drawn["host"]), jnp.asarray(drawn["axis"]))
        angle = jnp.asarray(drawn["angle"])
        torque = factor * deviator_torque(_deviator(drawn, angle), world_axis, rate)
        derivative = jax.grad(lambda phi: 0.5 * rate @ _deviator(drawn, phi) @ rate)(angle)
        scale = float(jnp.abs(rate) @ jnp.abs(_deviator(drawn, angle)) @ jnp.abs(rate))
        return float(torque - derivative), scale


@PROPERTY
@given(exchanges())
def test_the_rotor_s_inequality_torque_is_the_derivative_of_its_host_s_energy_in_its_angle(drawn):
    residual, scale = _exchange_residual(drawn, 1.0)
    assert abs(residual) <= ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale, (residual, scale)


@pytest.mark.parametrize("factor", [0.9, -1.0])
def test_a_scaled_or_flipped_inequality_torque_fails_the_exchange_s_identity_on_every_draw(factor):
    failed: list[bool] = []

    @PROPERTY
    @given(exchanges())
    def check(drawn: dict[str, np.ndarray]) -> None:
        residual, scale = _exchange_residual(drawn, factor)
        failed.append(abs(residual) > ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale)

    check()
    assert failed and all(failed), f"{sum(failed)} of {len(failed)}"


@st.composite
def gyroscopic_cases(draw: Any) -> dict[str, np.ndarray]:
    """A three-joint tree's mass matrix (symmetric positive definite, to 1e-2 kg m^2), the angular velocity Jacobians
    of two frames, their spinners' momenta to 1e-3 N m s, the joints' rates to 2 rad/s."""
    entries = np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=9, max_size=9))).reshape(3, 3)
    mass = 1e-3 * (entries @ entries.T + np.eye(3))
    jacobians = np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=18, max_size=18))).reshape(2, 3, 3)
    momenta = 1e-3 * np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=6, max_size=6))).reshape(2, 3)
    rates = 2.0 * np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3)))
    return {"mass": mass, "jacobians": jacobians, "momenta": momenta, "rates": rates}


def _cayley_residual(drawn: dict[str, np.ndarray], symmetric: float) -> tuple[float, float]:
    with jax.enable_x64(True):
        mass, rates = jnp.asarray(drawn["mass"]), jnp.asarray(drawn["rates"])
        gyroscopic = gyroscopic_matrix(jnp.asarray(drawn["jacobians"]), jnp.asarray(drawn["momenta"]))
        gyroscopic = gyroscopic + symmetric * jnp.max(jnp.abs(gyroscopic)) * jnp.eye(3)
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
        change, scale = _cayley_residual(drawn, 0.05)
        failed.append(abs(change) > ledger.IDENTITY_ULPS * float(np.finfo(np.float64).eps) * scale)

    check()
    assert failed and all(failed), f"{sum(failed)} of {len(failed)}"


def test_the_rotor_s_exchange_over_a_run_falls_at_first_order_in_the_step():
    found = {}
    held = chain_cases.experiment
    with in_dtype("float64"):
        for dt in (1e-3, 5e-4):
            chain_cases.experiment = lambda name, dt=dt: Experiment(name, scene=Scene.new(up=Z, units="m"),
                                                                     gravity=standard_gravity(), dt=ms(dt * 1e3))
            try:
                experiment = massive_rpm()
            finally:
                chain_cases.experiment = held
            experiment.observe(EnergyLedger())
            account = experiment.build().run(s(0.2)).energy
            assert account is not None
            found[dt] = account.exchanges["rpm.motor.inertia"]
    ratio = found[1e-3] / found[5e-4]
    print(f"\nthe rotor's exchange over 0.2 s: {found}, ratio {ratio:.3f}")
    assert 1.6 <= ratio <= 2.4, found


def test_the_segments_take_the_ledger_and_refuse_a_delivery_they_pair_nothing_for():
    arm: Any = PORT_CASES["joint"].rig()
    wrong = ledger.LedgerDelivery("resistance", "drag:case", ledger.DISSIPATIVE, ledger.SOLVED)
    with pytest.raises(ValueError, match="pairs nothing for"):
        DynamicArmNode(arm.name, float(arm.delta_t), **{**dict(arm.params), "ledger": (wrong.encoded(),)})
    motor: Any = PORT_CASES["spinner"].rig()
    with pytest.raises(ValueError, match="pairs nothing for"):
        DynamicMotorNode(motor.name, float(motor.delta_t), **{**dict(motor.params), "ledger": (wrong.encoded(),)})
