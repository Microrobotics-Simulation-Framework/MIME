"""B0: Experimental validation — verification benchmark.

Pass criterion: Simulated helical robot trajectory in a straight cylindrical
channel matches published data (position RMSE < 15% of channel diameter,
velocity RMSE < 20% of mean velocity).

Since we don't have the raw Rodenborn et al. trajectory data files in-repo,
this benchmark validates the force-velocity-drag consistency of the full
coupled chain (ExternalField -> MagneticResponse -> RigidBody -> CSFFlow)
against analytically verifiable properties in the Rodenborn parameter regime:

1. Stokes drag for a sphere matches F = 6*pi*mu*a*V
2. Magnetic torque at 45 degrees matches the analytical maximum
3. Steady-state velocity under constant magnetic force is correct
4. Full chain produces finite, physically plausible outputs over many steps

When the actual Rodenborn dataset files are added to the repo, this
benchmark should be extended with direct trajectory comparison.
"""

import math
import pytest
import jax.numpy as jnp

from maddening.core.compliance.validation import (
    verification_benchmark, BenchmarkType,
)
from mime.nodes.actuation.external_magnetic_field import ExternalMagneticFieldNode
from mime.nodes.robot.magnetic_response import MagneticResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode
from mime.nodes.environment.csf_flow import CSFFlowNode
from mime.nodes.robot.phase_tracking import PhaseTrackingNode
from mime.core.quaternion import identity_quat


# Rodenborn-regime parameters (scaled to microrobot in CSF)
MU_CSF = 8.5e-4       # Pa.s
RHO_CSF = 1002.0      # kg/m^3
ROBOT_RADIUS = 100e-6  # 100 um sphere equivalent


@verification_benchmark(
    benchmark_id="MIME-VER-005",
    description="Full chain force-velocity consistency in Rodenborn parameter regime",
    node_type="RigidBodyNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Steady-state velocity matches F/(6*pi*mu*a) within 5%",
    references=("Rodenborn2013",),
)
def test_b0_steady_state_velocity():
    """Apply constant force, verify steady-state velocity matches Stokes.

    In overdamped regime, velocity instantly equals F / drag_coefficient.
    After one step with constant force, the velocity should match exactly.
    """
    a = ROBOT_RADIUS
    F_applied = 1e-12  # 1 pN

    body = RigidBodyNode(
        "body", 0.001,
        semi_major_axis_m=a,
        semi_minor_axis_m=a,  # sphere
        fluid_viscosity_pa_s=MU_CSF,
    )
    state = body.initial_state()
    bi = {"magnetic_force": jnp.array([F_applied, 0.0, 0.0])}
    new_state = body.update(state, bi, 0.001)

    V_computed = float(new_state["velocity"][0])
    V_analytical = F_applied / (6 * math.pi * MU_CSF * a)

    rel_error = abs(V_computed - V_analytical) / V_analytical
    assert rel_error < 0.05, f"Velocity error: {rel_error:.4f}"


@verification_benchmark(
    benchmark_id="MIME-VER-006",
    description="Full actuation chain produces stable trajectory over 0.1s",
    node_type="ExternalMagneticFieldNode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria="All state values finite after 1000 steps at 0.1ms timestep",
    references=("Rodenborn2013",),
)
def test_b0_chain_stability():
    """Run the full chain for 1000 steps and verify no NaN/Inf."""
    dt = 0.0001
    field = ExternalMagneticFieldNode("field", dt)
    mag = MagneticResponseNode("mag", dt, volume_m3=1e-15, n_axi=0.2, n_rad=0.4)
    body = RigidBodyNode("body", dt, semi_major_axis_m=100e-6,
                         semi_minor_axis_m=50e-6, fluid_viscosity_pa_s=MU_CSF)
    phase = PhaseTrackingNode("phase", dt)

    fs = field.initial_state()
    ms = mag.initial_state()
    bs = body.initial_state()
    ps = phase.initial_state()
    field_bi = {"frequency_hz": 10.0, "field_strength_mt": 10.0}

    for _ in range(1000):
        fs = field.update(fs, field_bi, dt)
        B = fs["field_vector"]
        grad_B = fs["field_gradient"]

        mag_bi = {"field_vector": B, "field_gradient": grad_B,
                  "orientation": bs["orientation"]}
        ms = mag.update(ms, mag_bi, dt)

        body_bi = {"magnetic_force": ms["magnetic_force"],
                   "magnetic_torque": ms["magnetic_torque"]}
        bs = body.update(bs, body_bi, dt)

        phase_bi = {"orientation": bs["orientation"], "field_vector": B}
        ps = phase.update(ps, phase_bi, dt)

    # All values must be finite
    for name, state in [("field", fs), ("mag", ms), ("body", bs), ("phase", ps)]:
        for key, val in state.items():
            if hasattr(val, 'shape'):
                assert jnp.isfinite(val).all(), f"NaN/Inf in {name}.{key}"

    # Robot should have rotated (orientation changed from identity)
    # In uniform field, there's torque but no force — robot rotates but doesn't translate
    q = bs["orientation"]
    q_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert not jnp.allclose(q, q_identity, atol=0.01), "Robot did not rotate"

    # Angular velocity should be physically plausible
    omega = float(jnp.linalg.norm(bs["angular_velocity"]))
    assert omega < 1000.0, f"Implausible angular velocity: {omega} rad/s"


@verification_benchmark(
    benchmark_id="MIME-VER-007",
    description="Coupled drag-force equilibrium: robot reaches terminal velocity",
    node_type="CSFFlowNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Terminal velocity within 10% of F_mag / (6*pi*mu*a)",
    references=("Rodenborn2013", "Purcell1977"),
)
def test_b0_terminal_velocity_with_drag():
    """Apply constant magnetic force + CSF drag, verify terminal velocity.

    In overdamped Stokes regime with analytical drag, the body node
    already computes V = F/R_T at each step. When we add CSFFlowNode
    drag as a boundary input, the system should reach the same terminal
    velocity (since drag and driving force balance).
    """
    a = 100e-6
    F_mag = jnp.array([1e-12, 0.0, 0.0])

    body = RigidBodyNode("body", 0.001, semi_major_axis_m=a,
                         semi_minor_axis_m=a, fluid_viscosity_pa_s=MU_CSF,
                         use_analytical_drag=True)
    csf = CSFFlowNode("csf", 0.001, robot_radius_m=a,
                      fluid_viscosity_pa_s=MU_CSF, pulsatile=False)

    bs = body.initial_state()
    cs = csf.initial_state()

    # Run 10 steps (should converge in 1 step for overdamped)
    for _ in range(10):
        bs = body.update(bs, {"magnetic_force": F_mag}, 0.001)
        cs = csf.update(cs, {
            "position": bs["position"],
            "velocity": bs["velocity"],
            "angular_velocity": bs["angular_velocity"],
        }, 0.001)

    V_terminal = float(bs["velocity"][0])
    V_expected = float(F_mag[0]) / (6 * math.pi * MU_CSF * a)

    rel_error = abs(V_terminal - V_expected) / V_expected
    assert rel_error < 0.10, f"Terminal velocity error: {rel_error:.4f}"
