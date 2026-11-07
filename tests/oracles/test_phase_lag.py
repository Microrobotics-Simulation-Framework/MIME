"""Gate 7, ``PhaseLag``: the lag of a body's moment behind a uniform field turning below step-out, against the torque
balance ``m B sin(phi) = 2 pi f R``, with ``R`` the rotational resistance.

**The oracle.** In the steady state the body turns with the field at ``omega = 2 pi f``, and the torque the field
exerts on the moment, ``m B sin(phi)``, balances the drag's ``R omega``. The torque reads the body's pose at the start
of the step and the field at its end (the build report's lags), so the angle it acts on is a step behind the one an
observation reads after the step, by the body's turn over the step, ``omega dt``; ``PhaseLag`` is defined net of that
declared lag, and must equal ``arcsin(R omega / (m B))``. Two cases: a constant rotational resistance (a test-only
law, ``R`` exact), within 1e-9 rad (the transient's remainder, as the integration oracle states it); and a 1 mm sphere
in free space on the drag's MRS rung at its defaults, against the exact regularised sphere's rotational resistance at
the rung's blob, within ``tan(phi)`` times the rung's stated discretisation, 4e-4 of the resistance (2.2e-4 rad at
``phi = 0.5``), as committed in ``tests/study/predictions_drives.json`` before this ran.

**The control.** The lag without its correction, the moment as it stands after the step, misses the balance by
``omega dt``, 0.0314 rad at 10 Hz and 0.5 ms: 143 times the MRS case's tolerance and 3e7 times the constant
resistance's, so it fails on every run; and ``PhaseLag`` built without its correction (the mutant, its declared lag
planted at zero steps) fails the gate. The correction is the build's declared lag: the report gives the torque's read of
the pose as the previous step and the observation's as this step, one step apart."""

from __future__ import annotations

import math
from collections.abc import Iterator

import jax
import numpy as np
import pytest
from mrs_cases import drag_node
from regularised_sphere import regularised_rotational_drag
from rigid_oracles import ConstantResistance, resistance

import mime.observations._phase_lag as phase_lag_module
from mime import Apparatus, Body, Experiment, Medium, State
from mime.build import BuildRefused, Lag
from mime.build._maddening import parameter_leaves
from mime.dynamics import RigidInertial
from mime.experiment import BuiltExperiment
from mime.fields import UniformRotating
from mime.interactions import Drag, Magnetic
from mime.observations import PhaseLag
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import A_m2, Hz, T, kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, rad_per_s, s

FREQUENCY, FIELD, PHI = 10.0, 1.2e-3, 0.5
OMEGA = 2.0 * math.pi * FREQUENCY
#: The integration oracle's body against a constant resistance: its inertia and moment (tests/oracles/
#: test_synchronous_rotation.py), the resistance set for a lag of PHI.
INERTIA, MOMENT = 1.0e-10, 1.68e-3
LOCKED = MOMENT * FIELD * math.sin(PHI) / OMEGA
#: The constant resistance's tolerance: the transient's remainder by the window (exp(-77) of its start).
EXACT = 1e-9
#: The MRS sphere: radius, viscosity, and a light body whose lock settles in a tenth of the run.
RADIUS, VISCOSITY, SPHERE_MASS, SPHERE_INERTIA = 1e-3, 1e-3, 1e-6, 1e-13
#: The rung's stated discretisation against the exact regularised sphere (tests/oracles/test_mrs_separated.py), and
#: the lag's tolerance it gives at PHI, with the transient's remainder.
DISCRETISATION = 4e-4
MRS_TOLERANCE = math.tan(PHI) * DISCRETISATION + 1e-6
DURATION, AFTER = s(0.6), s(0.4)
LAG = "phase_lag:ball"


@pytest.fixture(scope="module")
def float64() -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def turned(name: str, *, dt: float, moment: float, mass: float, inertia: float, sphere: bool) -> Experiment:
    """A body with a moment along +x turned about +z by a uniform field at FREQUENCY, against a constant rotational
    resistance or, as a 1 mm sphere in water, the drag's MRS rung; its phase lag observed from AFTER."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    parts = {"shell": Part(Sphere(radius=mm(RADIUS * 1e3)))} if sphere else {}
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), parts=parts, initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("ball.mass", kg(mass), reason="the oracle's body")
    exp.declare("ball.inertia", kg_m2((inertia,) * 3), reason="the oracle's body")
    exp.declare("ball.magnetic_moment", A_m2((moment, 0.0, 0.0)), reason="the oracle's body")
    field = exp.add_apparatus(Apparatus("field", fields={"magnetic": UniformRotating(
        rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))},
        commands={"frequency": Hz(FREQUENCY), "amplitude": T(FIELD)}))
    exp.attach(Magnetic.Torque(), body=ball, field=field.magnetic)
    if sphere:
        water = exp.add_medium(Medium("water"))
        exp.declare("water.viscosity", mPa_s(VISCOSITY * 1e3), reason="water")
        exp.attach(Drag.MRS(wall_kind="free_space"), body=ball, medium=water)
    else:
        exp.attach(ConstantResistance(value=resistance(1.0e-3, LOCKED)), body=ball)
    exp.observe(PhaseLag(ball, field=field.magnetic, about=(0.0, 0.0, 1.0), after=AFTER))
    return exp


def constant(dt: float) -> Experiment:
    return turned("lag_constant", dt=dt, moment=MOMENT, mass=1.0e-4, inertia=INERTIA, sphere=False)


def sphere_resistance() -> float:
    """The exact regularised sphere's rotational resistance at the rung's blob for the 1 mm sphere (N m s)."""
    probe = turned("lag_sphere", dt=5e-4, moment=1e-6, mass=SPHERE_MASS, inertia=SPHERE_INERTIA, sphere=True).build()
    node = drag_node(probe, "ball")
    held = node.body({key: parameter_leaves(probe.graph)["drag:ball"][key] for key in node.body_leaf_names()})
    eps = float(np.sqrt(held.eps_squared)) / RADIUS
    return 8.0 * math.pi * VISCOSITY * RADIUS ** 3 * regularised_rotational_drag(eps)


@pytest.fixture(scope="module")
def sphere(float64: None) -> tuple[BuiltExperiment, float]:
    """The MRS sphere built with its moment set for a lag of PHI against the exact resistance, and that resistance."""
    exact = sphere_resistance()
    moment = exact * OMEGA / (FIELD * math.sin(PHI))
    return turned("lag_sphere", dt=5e-4, moment=moment, mass=SPHERE_MASS, inertia=SPHERE_INERTIA,
                  sphere=True).build(), exact


def lag_and_spin(built: BuiltExperiment) -> tuple[float, float]:
    result = built.run(DURATION)
    return float(result[LAG].mean_lag.value), float(result[LAG].mean_spin.to(Hz))


@pytest.mark.parametrize("dt", [5e-4, 2.5e-4])
def test_the_lag_meets_the_torque_balance_against_a_constant_resistance(dt, float64, record_property):
    lag, spin = lag_and_spin(constant(dt).build())
    miss = lag - PHI
    record_property("miss_rad", miss)
    print(f"\ndt {dt}: lag {lag:.15f} rad, miss {miss:.2e}, omega dt {OMEGA * dt:.4f}, spin {spin:.12f} Hz")
    assert abs(miss) <= EXACT
    assert abs(spin / FREQUENCY - 1.0) < 1e-9, "locked"


def test_the_lag_meets_the_torque_balance_of_the_regularised_sphere_on_the_mrs_rung(sphere, record_property):
    built, exact = sphere
    lag, spin = lag_and_spin(built)
    node = drag_node(built, "ball")
    assert node.sample_count == 338, "the rung's default for a sphere"
    miss = lag - PHI
    record_property("miss_rad", miss)
    print(f"\nMRS sphere: lag {lag:.9f} rad, miss {miss:.3e} against the tolerance {MRS_TOLERANCE:.3e}; omega dt "
          f"{OMEGA * 5e-4:.4f} ({OMEGA * 5e-4 / MRS_TOLERANCE:.0f} times it); spin {spin:.9f} Hz")
    assert abs(miss) <= MRS_TOLERANCE
    assert abs(spin / FREQUENCY - 1.0) < 1e-6, "locked"


def test_control_the_lag_without_its_correction_misses_by_omega_dt(sphere, float64):
    """The moment as it stands after the step, the lag less the body's turn over the step: off by ``omega dt``, far
    past both tolerances."""
    for built, tolerance, dt in ((constant(5e-4).build(), EXACT, 5e-4), (sphere[0], MRS_TOLERANCE, 5e-4)):
        lag, spin = lag_and_spin(built)
        raw = lag - 2.0 * math.pi * spin * dt
        assert abs(abs(raw - PHI) - OMEGA * dt) < 10 * tolerance + 1e-6
        assert abs(raw - PHI) > 10 * tolerance


def test_mutant_phase_lag_without_its_correction_fails_the_gate(float64, monkeypatch):
    monkeypatch.setattr(phase_lag_module, "LAWS_READ_BEHIND", 0)
    lag, _ = lag_and_spin(constant(5e-4).build())
    assert abs(lag - PHI) > 10 * EXACT
    assert abs(abs(lag - PHI) - OMEGA * 5e-4) < 1e-6


def test_the_correction_is_the_lag_the_build_reports(float64):
    """The torque reads the pose a step behind the observation, and the field in the same step: the report's lags."""
    built = constant(5e-4).build()
    lags = {edge.text: edge.lag for edge in built.report.lags}
    assert lags["ball.pose -> magnetic.torque:ball.pose"] is Lag.PREVIOUS_STEP
    assert lags["ball.pose -> phase_lag:ball.pose"] is Lag.THIS_STEP
    field = next(text for text in lags if text.endswith("-> magnetic.torque:ball.flux_density"))
    assert lags[field] is Lag.THIS_STEP
    assert lags[field.replace("magnetic.torque:ball.flux_density", "phase_lag:ball.flux_density")] is Lag.THIS_STEP
    assert phase_lag_module.LAWS_READ_BEHIND == 1


def test_the_observation_reads_the_uniform_field_where_the_torque_reads_it_adding_no_slot(float64):
    """The provider's node is the same with the observation and without it: its reader slots are the torque's alone."""
    with_lag = constant(5e-4).build()
    plain = constant(5e-4)
    plain.remove(LAG)
    without = plain.build()
    fields = [sorted(built.graph.get_node("field.magnetic").initial_state()) for built in (with_lag, without)]
    assert fields[0] == fields[1] and "flux_density_magnetic_torque_ball" in fields[0]
    assert not any("phase_lag" in name for name in fields[0])


def test_an_observation_of_a_field_no_law_reads_is_refused(float64):
    exp = constant(5e-4)
    exp.remove("magnetic.torque:ball")
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    assert "observation.field-read" in refused.value.codes()
    message = next(r.message for r in refused.value.refusals if r.code == "observation.field-read")
    assert "no law reads field.magnetic.flux_density from it" in message
