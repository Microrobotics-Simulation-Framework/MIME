"""Script A in float32 against the frozen float32 reference, with the differences declared for float32 before the
run (``script_a_expected_differences.json``: the f32 reference moved by the emulated difference between the two
schemes; the commands round alike in float32, so theirs is absent). Each line rounds on its own, so the tolerances are
at least ten times the reference's own float32-float64 gap in each kind of quantity. What it bounds is
``test_script_a.py``'s.

**Post hoc** (declared after the first run, before this one; ``post_hoc`` in the file): the phase lag missed its
tolerance. 0.3's implicit step quantizes the locked torque balance in float32 at one unit in the last place of the
inertial term ``(I/dt) omega``, a fixed lag offset of up to 4.5e-7 rad, above the declared 1e-7. The tolerance is
unchanged: the float32 lag is reported, not gated, against the reference, and the quantization is tested instead (its
bound, that it is the same whatever the axial start, and the control: solved for the change in velocity, the same
equation, the float32 lag holds at the declared tolerance).
"""

import jax
import pytest
from script_a import script_a
from script_a_runs import DECLARED, GATED, REFERENCE, compare, quantities, reported, run, table

import script_a_runs
from mime.core import jnp
from mime.nodes.body import RigidInertialNode
from mime.units import Hz, mm, s

LAG = "phase_lag_field_minus_moment_rad"
QUANTIZATION = DECLARED["post_hoc"][0]
assert QUANTIZATION["id"] == "float32-lock-quantization"
TOLERANCES = DECLARED["tolerances"]["f32"]
PREDICTED = DECLARED["predicted"]["f32_10hz"]
FLOAT64_LAG = DECLARED["predicted"]["f64_10hz"][LAG]


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32: it overrides the folder's float64 fixture, and restores the mode after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module")
def at_10_hz():
    history = run(10.0)
    assert history.position.dtype.name == "float32"
    return quantities(history, 10.0)


def test_script_a_in_float32_matches_its_float32_reference_with_the_declared_differences(at_10_hz):
    assert set(TOLERANCES) == set(GATED)
    rows = compare(at_10_hz, REFERENCE["f32"], PREDICTED, TOLERANCES)
    print("\nfloat32, 10 Hz (the lag reported, not gated: post hoc, its float32 quantization)\n" + table(rows) + "\n"
          + reported(at_10_hz))
    assert [row.quantity for row in rows if not row.holds and row.quantity != LAG] == []


def test_the_float32_lag_sits_within_its_quantization_of_the_float64_lag(at_10_hz):
    offset = at_10_hz[LAG] - FLOAT64_LAG
    bound = QUANTIZATION["predicted"]["float32_minus_float64_lag_bound_rad"]
    print(f"\nfloat32 lag - float64 lag: {offset:+.4e} rad; the quantization's bound {bound:.3e} rad")
    assert abs(offset) <= bound


def test_the_float32_lag_offset_is_the_same_wherever_the_body_starts_along_the_tube(at_10_hz, monkeypatch):
    def further(frequency=Hz(10.0), dt=s(5e-4)):
        return script_a("script_a", frequency=frequency, dt=dt, start=mm((5.0, 0.0, -1.0)))

    monkeypatch.setattr(script_a_runs, "reproduction_arm", further)
    assert abs(quantities(run(10.0), 10.0)[LAG] - at_10_hz[LAG]) <= 1e-12


def test_solved_for_the_change_in_velocity_the_float32_lag_holds_at_the_declared_tolerance(monkeypatch):
    """The control on the cause: the same equation, ``(M/dt + R) dV = F - R V``, with no small load added to the
    inertial term, takes the offset away, and the float32 lag then holds against the reference at the declared
    tolerance."""
    def change_in_velocity(self, mass_matrix, resistance, twist, load, dt):
        return twist + jnp.linalg.solve(mass_matrix / dt + resistance, load - resistance @ twist)

    monkeypatch.setattr(RigidInertialNode, "velocities", change_in_velocity)
    measured = quantities(run(10.0), 10.0)
    (row,) = compare(measured, REFERENCE["f32"], PREDICTED, {LAG: TOLERANCES[LAG]})
    print(f"\nsolved for the change in velocity: float32 lag - float64 lag {measured[LAG] - FLOAT64_LAG:+.4e} rad\n"
          + table([row]))
    assert abs(measured[LAG] - FLOAT64_LAG) <= 2e-8
    assert row.holds
