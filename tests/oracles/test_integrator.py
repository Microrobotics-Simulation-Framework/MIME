"""The rigid inertial body's integrator against closed forms. From rest, under a constant force against a constant
linear resistance, the velocity approaches the terminal value ``F / c`` as ``1 - exp(-c t / m)``; under a constant
torque against a rotational resistance, the angular velocity the same way with the inertia. The implicit scheme's
error against that curve is first order: halving ``dt`` halves it. The quaternion stays unit-norm.

The control: the resistance integrated explicitly, as 0.2 did, planted as a subclass of the node. At a ``dt`` above
twice the resistance's time scale ``m / c`` it diverges, while the implicit scheme relaxes to the terminal value.
"""

import numpy as np
import pytest
import rigid_oracles as oracle
from rigid_oracles import ConstantResistance, ConstantWrench, resistance

from mime import Body, Experiment, State
from mime.dynamics import RigidInertial
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, rad_per_s, s

LEVELS = (50, 100, 200, 400)


@pytest.mark.parametrize("rotation", [False, True], ids=["translation", "rotation"])
def test_the_implicit_scheme_is_first_order_against_the_closed_form(rotation):
    errors = [oracle.velocity_error(steps, rotation=rotation) for steps in LEVELS]
    orders = oracle.first_order(errors)
    print(f"\nerrors {errors}\norders {orders}")
    assert all(0.9 < order < 1.1 for order in orders[1:])
    assert errors[-1] < 1e-2


def test_the_quaternion_stays_unit_norm():
    _, norms = oracle.twists(oracle.HORIZON / 400, 400)
    assert float(np.max(np.abs(norms - 1.0))) < 1e-14


def test_the_scheme_relaxes_where_an_explicit_one_diverges():
    """The control: at ``dt = 2.5 m / c`` the explicit update multiplies the velocity's distance from the terminal
    value by ``|1 - dt c / m| = 1.5`` each step; the implicit one divides it by ``1 + dt c / m``."""
    dt = 2.5 * oracle.TAU
    implicit, _ = oracle.twists(dt, 40)
    explicit, _ = oracle.twists(dt, 40, oracle.ExplicitNode)
    terminal = np.concatenate([oracle.FORCE / oracle.LINEAR, oracle.TORQUE / oracle.ROTATIONAL])
    assert np.allclose(implicit[-1], terminal, rtol=1e-9, atol=0.0)
    assert not np.all(np.isfinite(explicit[-1])) or float(np.max(np.abs(explicit[-1] / terminal))) > 1e3
    small, _ = oracle.twists(0.01 * oracle.TAU_ROTATION, 3)
    explicit_small, _ = oracle.twists(0.01 * oracle.TAU_ROTATION, 3, oracle.ExplicitNode)
    assert np.allclose(small, explicit_small, rtol=0.05), "below the time scale the two schemes agree to first order"


def test_the_same_holds_through_a_built_experiment():
    """The node as an experiment builds it, with the constant wrench and resistance delivered as laws: its twist
    is the node's, step for step."""
    steps, dt = 50, oracle.HORIZON / 50
    experiment = Experiment("terminal", up=(0.0, 0.0, 1.0), gravity=m_per_s2(0.0), dt=s(dt))
    experiment.add_body(Body("probe", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("probe.mass", kg(oracle.MASS), reason="the oracle's body")
    experiment.declare("probe.inertia", kg_m2((oracle.INERTIA,) * 3), reason="the oracle's body")
    experiment.attach(ConstantWrench(value=tuple(np.concatenate([oracle.FORCE, oracle.TORQUE]))), body="probe")
    experiment.attach(ConstantResistance(value=resistance(oracle.LINEAR, oracle.ROTATIONAL)), body="probe")
    history = experiment.build().run(s(steps * dt), history=True).history
    assert history is not None
    expected, _ = oracle.twists(dt, steps)
    assert np.allclose(history["probe"]["twist"], expected, rtol=1e-12, atol=0.0)
