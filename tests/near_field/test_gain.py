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


def step_of(shape: str, gap: float, ratio: float, built=None, **settings):
    """The joint node's first step at the configuration, every cross-block integrated: assembled from the bodies'
    per-body nodes (drag 0.4.0, of ``built`` where given), so that a configuration the build refuses at its start is
    stepped all the same."""
    built = built if built is not None else cluster(shape, gap, ratio, UncoupledDrag).build()
    names = sorted(placed(shape, gap))
    node = assembled([built.graph.get_node(f"drag:{body}") for body in names], names, reach=1e9, cross_share=1.0,
                     **settings)
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


def lagged_against_joint(own: np.ndarray, off: np.ndarray, masses: np.ndarray, steps: int,
                         dt: float) -> tuple[float, float]:
    """The bodies' velocities under constant random loads (each torque on the scale of a force times the radius, and
    each angular velocity judged times the radius, so that no mode's share of the error is set by the units), each
    body's step implicit in its own ``R_aa`` and the others' loads lagged a step, against the joint step's (one solve
    of ``(M/dt + R) dV = F - R V``), from rest: the lagged error's size over the joint velocity's after ``steps``, and
    its growth over the next ``steps``. The growth is the judgement of an unstable lagged step: whatever share of the
    error its growing mode starts with, after ``steps`` it leads, and its size then grows by the iteration's spectral
    radius to the power ``steps``."""
    count = own.shape[0]
    scale = np.tile(np.r_[np.ones(3), np.full(3, RADIUS)], count)
    forces = np.random.default_rng(7).normal(size=6 * count) * scale
    mass = np.zeros((6 * count, 6 * count))
    big = np.zeros((6 * count, 6 * count))
    grand = off.copy()
    for i in range(count):
        mass[6 * i:6 * i + 6, 6 * i:6 * i + 6] = masses[i]
        big[6 * i:6 * i + 6, 6 * i:6 * i + 6] = masses[i] / dt + own[i]
        grand[6 * i:6 * i + 6, 6 * i:6 * i + 6] = own[i]
    lagged = np.zeros(6 * count)
    joint = np.zeros(6 * count)
    sizes = []
    for step in range(2 * steps):
        lagged = lagged + np.linalg.solve(big, forces - off @ lagged - (big - mass / dt) @ lagged)
        joint = joint + np.linalg.solve(mass / dt + grand, forces - grand @ joint)
        if step + 1 in (steps, 2 * steps):
            sizes.append(float(np.linalg.norm((lagged - joint) * scale) / np.linalg.norm(joint * scale)))
    return sizes[0], sizes[1] / sizes[0]


def test_control_without_the_check_the_triangle_it_refuses_grows_and_the_line_it_lets_through_decays():
    """The check removed, the blocks the node built stepped lagged: the triangle the check refuses (gain 1.015) grows
    at its gain's rate, at least a thousandfold over 1,000 steps (1.015 to the 1,000th is about 2e6, less the
    mass's share), where the line it lets through (0.90) has decayed to under a thousandth of the joint velocity."""
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
    print(f"\nthe lagged error over the joint velocity after 1,000 steps, and its growth over the next 1,000: triangle "
          f"{found['triangle'][0]:.3g}, x{found['triangle'][1]:.3g}; line {found['line'][0]:.3g}, "
          f"x{found['line'][1]:.3g}")
    assert found["triangle"][1] > 1e3 and found["line"][0] < 1e-3


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


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_on_the_iterative_route_the_gain_is_the_dense_routes_and_decides_the_same(dtype):
    """Gate 6 past the takeover line (gate 11's dtype arm): the triangle at a tenth of a radius, overdamped (refused)
    and inertial (let through), its gain on GMRES within the solve's tolerance of the dense route's, in each dtype."""
    rows = []
    with precision(dtype):
        for ratio in (0.001, 1.0):
            built, _, _, dense = step_of("triangle", 0.1, ratio)
            rows.append((ratio, dense, step_of("triangle", 0.1, ratio, built, dense_unknowns=0)[3]))
    for ratio, dense, iterative in rows:
        gains = float(dense["gain"]), float(iterative["gain"])
        print(f"\n{dtype}, the triangle at 0.1 a, tau/dt {ratio}: gain {gains[1]:.6f} on GMRES, {gains[0]:.6f} dense "
              f"(report {iterative['solve_report']})")
        assert abs(gains[1] - gains[0]) <= (1e-8 if dtype == "float64" else 1e-4)
        assert float(iterative["gain_exceeded"]) == float(dense["gain_exceeded"]) == float(ratio < 0.01)
        assert float(iterative["solve_flagged"]) == 0.0
