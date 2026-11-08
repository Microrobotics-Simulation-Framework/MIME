"""Gate 6, the lagged step's gain (Bundle 2): ``rho((M/dt + R_d)^-1 R_off)``, each body's step implicit in its own coupled
6 x 6 and the others' loads from the step before, checked each step and at the build. MIME's sphere at its default (338
force points), free space, float64, every cross-block integrated (as the patch spike's question 6 assembled them), the
mass matrix of a uniform sphere with its relaxation time ``tau = m / (6 pi mu a)`` set against the step.

The spike's table (one agent's, float64): the frozen equilateral triangle at a gap of 0.1 a, overdamped (tau/dt 0.001),
gain 1.015, its lagged error 4.6e4 times the velocity after 1,000 steps: **refused on every run**, at the build and by
the step; not refused: a pair at 0.05 a (0.900), the line of three at 0.1 a (0.900), the triangle at 0.25 a (0.927) and
at tau/dt 1 (0.878). Each gain within 0.01 of the table. The control is the check removed: the triangle's bodies
stepped lagged with the node's own blocks, constant forces, diverge from the joint step, where every configuration
the check lets through decays."""

import math

import numpy as np
import pytest
from near_field_cases import (RADIUS, VISCOSITY, assembled, bodies, joint_node, line, precision, spheres, start_inputs,
                              uncoupled_inputs)

from mime.build import BuildRefused
from mime.interactions import Drag
from mime.interactions.hydrodynamic import UncoupledDrag

#: (configuration, gap over the radius, tau over dt, the spike's gain).
TABLE = [("pair", 0.05, 0.001, 0.900), ("line", 0.1, 0.001, 0.900), ("triangle", 0.25, 0.001, 0.927),
         ("triangle", 0.1, 1.0, 0.878), ("triangle", 0.1, 0.001, 1.015)]
DT = 1e-3
NEAR = 0.01


def placed(shape: str, gap: float):
    if shape == "pair":
        return spheres(gap * RADIUS, 2)
    if shape == "line":
        return line(gap * RADIUS, 3)
    return spheres(gap * RADIUS, 3)


def cluster(shape: str, gap: float, ratio: float, law: type = Drag):
    """The configuration at MIME's default sphere, each body's mass set by its relaxation time over the step, its
    moment a uniform sphere's."""
    mass = ratio * 6.0 * math.pi * VISCOSITY * RADIUS * DT
    return bodies("gain", placed(shape, gap), law.MRS(wall_kind="free_space"), mass=mass,
                  inertia=0.4 * mass * RADIUS * RADIUS, dt_ms=DT * 1e3)


def step_of(shape: str, gap: float, ratio: float):
    """The joint node's first step at the configuration, every cross-block integrated: assembled from the bodies'
    per-body nodes (drag 0.4.0), so that a configuration the build refuses at its start is stepped all the same."""
    built = cluster(shape, gap, ratio, UncoupledDrag).build()
    names = sorted(placed(shape, gap))
    node = assembled([built.graph.get_node(f"drag:{body}") for body in names], names, reach=1e9, cross_share=1.0)
    out = node.update(node.initial_state(), uncoupled_inputs(built, names), DT, params=node.params_pytree())
    return built, node, names, {key: np.asarray(value) for key, value in out.items()}


@pytest.mark.parametrize(("shape", "gap", "ratio", "spike"), TABLE)
def test_the_gain_is_the_spikes_and_a_gain_of_one_or_more_stops_the_step(shape, gap, ratio, spike):
    with precision("float64"):
        _, node, _, out = step_of(shape, gap, ratio)
    gain = float(out["gain"])
    print(f"\n{shape} at {gap} a, tau/dt {ratio}: gain {gain:.4f} (the spike's {spike}), relaxation "
          f"{float(out['relaxation_steps']):.3g} steps, flagged {float(out['gain_exceeded'])}")
    assert abs(gain - spike) <= NEAR
    refused = spike > 1.0
    assert float(out["gain_exceeded"]) == float(refused)
    assert np.all(np.isnan(out["resistance_0"])) == refused, "a refused step reaches no body"
    if refused:
        assert node.stop_problem(out).code == "drag.near-field-gain"


def test_the_build_refuses_the_overdamped_triangle_at_its_start_and_lets_the_inertial_one_through():
    with precision("float64"):
        with pytest.raises(BuildRefused) as refused:
            cluster("triangle", 0.1, 0.001).build()
        assert "drag.near-field-gain" in refused.value.codes()
        built = cluster("triangle", 0.1, 1.0).build()
    assert float(joint_node(built).judged_at(start_inputs(built))["gain"]) < 1.0


def lagged_against_joint(own: np.ndarray, off: np.ndarray, masses: np.ndarray, steps: int, dt: float) -> float:
    """The bodies' velocities under constant random forces, each body's step implicit in its own ``R_aa`` and the
    others' loads lagged a step, against the joint step's (one solve of ``(M/dt + R) dV = F - R V``), from rest: the
    lagged error's size over the joint velocity's, after ``steps``."""
    count = own.shape[0]
    forces = np.random.default_rng(7).normal(size=6 * count)
    big = np.zeros((6 * count, 6 * count))
    for i in range(count):
        big[6 * i:6 * i + 6, 6 * i:6 * i + 6] = masses[i] / dt + own[i]
    grand = off.copy()
    for i in range(count):
        grand[6 * i:6 * i + 6, 6 * i:6 * i + 6] = own[i]
    lagged = np.zeros(6 * count)
    joint = np.zeros(6 * count)
    mass = np.zeros((6 * count, 6 * count))
    for i in range(count):
        mass[6 * i:6 * i + 6, 6 * i:6 * i + 6] = masses[i]
    for _ in range(steps):
        lagged = lagged + np.linalg.solve(big, forces - off @ lagged - (big - mass / dt) @ lagged)
        joint = joint + np.linalg.solve(mass / dt + grand, forces - grand @ joint)
    return float(np.linalg.norm(lagged - joint) / np.linalg.norm(joint))


def test_control_without_the_check_the_triangle_it_refuses_diverges_and_the_line_it_lets_through_decays():
    found = {}
    for shape in ("triangle", "line"):
        with precision("float64"):
            built, node, names, out = step_of(shape, 0.1, 0.001)
        grand = np.asarray(out["resistance"], dtype=np.float64)
        count = len(node.members)
        own = np.stack([grand[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(count)])
        off = grand.copy()
        for i in range(count):
            off[6 * i:6 * i + 6, 6 * i:6 * i + 6] = 0.0
        inputs = uncoupled_inputs(built, names)
        masses = np.stack([np.asarray(inputs[f"mass_matrix_{i}"], dtype=np.float64) for i in range(count)])
        found[shape] = lagged_against_joint(own, off, masses, 1000, DT)
    print(f"\nthe lagged error after 1,000 steps over the joint velocity: triangle {found['triangle']:.3g}, line "
          f"{found['line']:.3g}")
    assert found["triangle"] > 1e2 and found["line"] < 1e-3


#: Gate 11's float32 line on the gain: the float32 step's gain against the float64 one, absolute.
FLOAT32_GAIN = 2e-3


@pytest.mark.parametrize(("shape", "gap", "ratio", "spike"), TABLE)
def test_in_float32_the_gain_is_the_float64_ones_and_decides_the_same(shape, gap, ratio, spike):
    with precision("float64"):
        _, _, _, wide = step_of(shape, gap, ratio)
    with precision("float32"):
        _, _, _, narrow = step_of(shape, gap, ratio)
    print(f"\n{shape} at {gap} a, tau/dt {ratio}: the gain {float(narrow['gain']):.5f} in float32, "
          f"{float(wide['gain']):.5f} in float64")
    assert abs(float(narrow["gain"]) - float(wide["gain"])) <= FLOAT32_GAIN
    assert float(narrow["gain_exceeded"]) == float(wide["gain_exceeded"])
