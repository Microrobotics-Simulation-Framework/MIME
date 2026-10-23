"""Script A in float32 against the frozen float32 reference, with the differences declared for float32 before the
run (the declaration in force, ``script_a_runs.IN_FORCE``: the f32 reference moved by the emulated difference between
the two schemes; the commands round alike in float32, so theirs is absent). Each line rounds on its own, so the
tolerances are at least ten times the reference's own float32-float64 gap in each kind of quantity. What it bounds is
``test_script_a.py``'s.

**Every quantity is gated, the phase lag included.** 4c2's first run found the lag 2.2e-7 rad off its prediction: the
implicit step, solved as ``(M/dt + R) V' = (M/dt) V + F``, rounded the locked torque at one unit in the last place of
the inertial term ``(I/dt) omega``, the same rounding at every step, so the lock kept it as a fixed lag offset
(MIME-ANO-113). 4c2 reported the lag rather than gating it, post hoc. rigid_inertial 0.2.0 solves the same equation
for the change in velocity (declared change 1), and the lag is gated again at its original tolerance, 1e-7 rad. The
control: 0.1.0's form, planted, takes the lag back beyond that tolerance.
"""

import jax
import pytest
from script_a_runs import DECLARED, GATED, PREDICTED, REFERENCE, compare, quantities, reported, run, table

from mime.core import jnp
from mime.nodes.body import RigidInertialNode

LAG = "phase_lag_field_minus_moment_rad"
TOLERANCES = DECLARED["tolerances"]["f32"]
FLOAT64_LAG = PREDICTED["f64_10hz"][LAG]


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
    rows = compare(at_10_hz, REFERENCE["f32"], PREDICTED["f32_10hz"], TOLERANCES)
    print("\nfloat32, 10 Hz, every quantity gated\n" + table(rows) + "\n" + reported(at_10_hz))
    assert [row.quantity for row in rows if not row.holds] == []


def test_the_float32_lag_lies_on_the_float64_lag(at_10_hz):
    """4c2's diagnostic of the increment form measured +3.97e-9 rad between the float32 and float64 lags; with the
    old form, -2.32e-7."""
    offset = at_10_hz[LAG] - FLOAT64_LAG
    print(f"\nfloat32 lag - float64 lag: {offset:+.4e} rad")
    assert abs(offset) <= 2e-8


def test_with_the_old_form_planted_the_float32_lag_misses_again(monkeypatch):
    """The control: rigid_inertial 0.1.0's ``(M/dt + R) V' = (M/dt) V + F``, which adds the torque to the inertial
    term before the solve, brings the fixed offset back and the gated lag misses its tolerance."""
    def solved_for_the_twist(self, mass_matrix, resistance, twist, load, dt):
        return jnp.linalg.solve(mass_matrix / dt + resistance, mass_matrix / dt @ twist + load)

    monkeypatch.setattr(RigidInertialNode, "velocities", solved_for_the_twist)
    measured = quantities(run(10.0), 10.0)
    (row,) = compare(measured, REFERENCE["f32"], PREDICTED["f32_10hz"], {LAG: TOLERANCES[LAG]})
    print(f"\nthe old form: float32 lag - float64 lag {measured[LAG] - FLOAT64_LAG:+.4e} rad\n" + table([row]))
    assert not row.holds
    assert abs(measured[LAG] - FLOAT64_LAG) > 1e-7
