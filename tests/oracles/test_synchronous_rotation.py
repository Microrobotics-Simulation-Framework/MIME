"""The integration oracle: a magnetic body turning with a uniform rotating field.

A body with a permanent moment ``m``, in a field of magnitude ``B`` turning at ``omega``, against a constant rotational
resistance ``R`` (a test-only law), locks to the field in the steady state, lagging it by ``phi = arcsin(R omega /
(m B))``, while ``R omega < m B``; above that it steps out. Built through the experiment: the field provider, its
constant drives, ``Magnetic.Torque`` and ``RigidInertial``.

**The tolerance, stated before the run.** Lag accounting shows one previous-step read in the loop: the torque reads
the body's pose at the start of the step while the field is this step's. In the steady state the body turns by
exactly ``omega dt`` a step, so the torque of step ``n + 1`` is ``m B sin(phi_measured + omega dt)``, where
``phi_measured`` compares the body and the field after the same step; balancing ``R omega`` gives
``phi_measured = phi - omega dt``. So the measured lag must lie within ``omega dt`` (plus 1e-9 rad for the transient,
which has decayed by a factor exp(-77) by the window) of the closed form, and the discrepancy must be that term: within
1e-9 rad of ``-omega dt``. Halving ``dt`` must halve it.

The control: above step-out the body does not lock: it turns at a third of the field's rate, and its lag does not
settle.
"""

import math

import numpy as np
import pytest
from rigid_oracles import ConstantResistance, resistance

from mime import Apparatus, Body, Experiment, State
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import Magnetic
from mime.scene import Scene, Z
from mime.units import A_m2, Hz, T, kg, kg_m2, m, m_per_s, m_per_s2, rad_per_s, s

INERTIA, MOMENT, FIELD, FREQUENCY = 1.0e-10, 1.68e-3, 1.2e-3, 10.0
OMEGA = 2 * math.pi * FREQUENCY
DURATION, WINDOW = 1.0, 0.2
TRANSIENT = 1e-9


def lock_experiment(dt: float, rotational: float) -> Experiment:
    experiment = Experiment("synchronous", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("umr.mass", kg(1.0e-4), reason="the oracle's body")
    experiment.declare("umr.inertia", kg_m2((INERTIA,) * 3), reason="the oracle's body")
    experiment.declare("umr.magnetic_moment", A_m2((MOMENT, 0.0, 0.0)), reason="the oracle's body")
    experiment.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(
        rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))},
        commands={"frequency": Hz(FREQUENCY), "amplitude": T(FIELD)}))
    experiment.attach(Magnetic.Torque(), body="umr", field="field.magnetic")
    experiment.attach(ConstantResistance(value=resistance(1.0e-3, rotational)), body="umr")
    return experiment


def steady_lag(dt: float, rotational: float) -> tuple[np.ndarray, float]:
    """The lag of the body's moment behind the field after each step of the last ``WINDOW`` seconds, and the body's
    mean rotation rate over it relative to the field's."""
    history = lock_experiment(dt, rotational).build().run(s(DURATION), history=True).history
    assert history is not None
    window = round(WINDOW / dt)
    moments = np.array([np.asarray(quaternion.rotate(q, np.array([1.0, 0.0, 0.0])))
                        for q in history["umr"]["pose"][-window:, 3:]])
    fields = history["field.magnetic"]["flux_density_magnetic_torque_umr"][-window:]
    lag = np.arctan2(np.cross(moments, fields)[:, 2], np.sum(moments * fields, axis=1))
    return lag, float(np.mean(history["umr"]["twist"][-window:, 5])) / OMEGA


PHI = 0.5
LOCKED = MOMENT * FIELD * math.sin(PHI) / OMEGA


@pytest.mark.parametrize("dt", [5e-4, 2.5e-4])
def test_the_body_locks_to_the_field_lagging_by_arcsin_r_omega_over_mb_less_one_steps_turn(dt):
    lag, ratio = steady_lag(dt, LOCKED)
    discrepancy = float(np.mean(lag)) - math.asin(LOCKED * OMEGA / (MOMENT * FIELD))
    print(f"\ndt {dt}: lag {np.mean(lag):.12f} rad (spread {np.ptp(lag):.2e}), discrepancy {discrepancy:.3e}, "
          f"omega dt {OMEGA * dt:.3e}, rate ratio {ratio:.15f}")
    assert abs(discrepancy) <= OMEGA * dt + TRANSIENT
    assert abs(discrepancy + OMEGA * dt) <= TRANSIENT
    assert float(np.ptp(lag)) < TRANSIENT and abs(ratio - 1.0) < 1e-9


def test_halving_dt_halves_the_discrepancy():
    coarse = float(np.mean(steady_lag(5e-4, LOCKED)[0])) - PHI
    fine = float(np.mean(steady_lag(2.5e-4, LOCKED)[0])) - PHI
    assert abs(fine) < abs(coarse)
    assert abs(coarse / fine - 2.0) < 1e-6


def test_above_step_out_the_body_does_not_lock():
    """The control: at ``R omega = 1.5 m B`` no lag balances the resistance; the body slips, turning far slower than
    the field, and its lag runs through every angle."""
    lag, ratio = steady_lag(5e-4, 1.5 * MOMENT * FIELD / OMEGA)
    print(f"\nrate ratio {ratio}, lag spread {np.ptp(lag)}")
    assert ratio < 0.5
    assert float(np.ptp(lag)) > math.pi
