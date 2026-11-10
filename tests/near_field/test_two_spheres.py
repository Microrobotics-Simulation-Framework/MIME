"""Gate 3, two spheres against analytic results (MIME's sphere at its default, 338 force points, float64, the joint
node at its default reach), lambda normalised by the single sphere's MRS drag at the same discretisation (its +0.5 %
divided out):

- the squeeze against Brenner's and Maude's series: within 1 % at h/a 1, 3 % at 0.25, 4 % at 0.1 (S3's pipeline:
  0.59, 2.36, 3.18 %, one agent's);
- off-axis, the line of centres at 45 degrees, the cross mobility against Rotne and Prager's: within 5e-4 at r/a 6 and
  2e-5 at 40, relative in the Frobenius norm (S3: 1.3e-4, 4.1e-6);
- equal velocities against Stimson and Jeffery: within 0.2 %, a far-field check that cannot see the near field (S3: the
  point cross-kernel, 17 to 154 % wrong on the squeeze, passed it to 0.13 %).

Below h/a 0.05 MRS's error is not monotone in the resolution (S3): lubrication's territory (W2), not a gate here. The
oracles' own checks: each series against the method of reflections at large gaps, and a planted slip in their
argument (cosh(alpha) = r/a for d/a) caught. The control: no cross-blocks, its error against each oracle at least twice
the joint node's at every configuration."""

import math

import numpy as np
import pytest
from near_field_cases import (NODE, RADIUS, VISCOSITY, bodies, joint_node, precision, rebuilt, spheres, start_inputs)
from sphere_pair_oracles import approaching, approaching_reflections, reflections, stimson_jeffery

from mime.build._maddening import parameter_leaves
from mime.interactions import Drag
from mime.nodes.hydro.near_field import grand_resistance, joint_matrix
from mime.scene import Sphere
from mime.units import m

STOKES = 6.0 * math.pi * VISCOSITY * RADIUS
SQUEEZE = {1.0: 0.01, 0.25: 0.03, 0.1: 0.04}
EQUAL = 0.002
OFF_AXIS = {6.0: 5e-4, 40.0: 2e-5}


def law():
    return Drag.MRS(wall_kind="free_space")


def single_drag() -> float:
    built = bodies("one", {"s0": (Sphere(radius=m(RADIUS)), (0.0, 0.0, 0.0))}, law()).build()
    node = built.graph.get_node("drag:s0")
    out = node.update(node.initial_state(), {"pose": np.asarray((0, 0, 0, 1.0, 0, 0, 0))}, 1e-3,
                      params=parameter_leaves(built.graph)["drag:s0"])
    return float(np.asarray(out["resistance"])[0, 0])


def grand_at(placed, coupled: bool = True) -> np.ndarray:
    built = bodies("pair", placed, law()).build()
    node = joint_node(built)
    p = node.constants(parameter_leaves(built.graph)[NODE])
    inputs = start_inputs(built)
    poses = np.stack([np.asarray(inputs[f"pose_{i}"]) for i in range(2)])
    if coupled:
        found = np.asarray(node.solved(p, poses)[0])
    else:
        held = [node._held_value(i, p) for i in range(2)]  # noqa: SLF001 - the node's own pieces, its coupling off
        found = np.asarray(grand_resistance(joint_matrix(node._operators, [node._leaves(i, p) for i in range(2)],  # noqa: SLF001
                                                         held, poses, p["viscosity"], node._wall(), {}, p["reach"],  # noqa: SLF001
                                                         coupled=False)))
    return 0.5 * (found + found.T)


def test_the_oracles_meet_the_method_of_reflections_far_apart_and_a_slip_in_their_argument_is_caught():
    for gap in (98.0, 18.0):
        s = 2.0 + gap
        assert abs(stimson_jeffery(gap) - reflections(gap)) * s ** 6 < 10.0
        assert abs(approaching(gap) - approaching_reflections(gap)) * s ** 6 < 10.0
        slipped = stimson_jeffery(2.0 * gap + 2.0)  # cosh(alpha) = r/a, the common slip, is the series at that gap
        assert abs(slipped - reflections(gap)) > 1e3 * abs(stimson_jeffery(gap) - reflections(gap))
    assert approaching(1e-3) * 2.0 * 1e-3 == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("gap", sorted(SQUEEZE, reverse=True))
def test_the_squeeze_and_equal_velocities_against_their_series_and_the_uncoupled_control(gap):
    with precision("float64"):
        single = single_drag()
        placed = spheres(gap * RADIUS, 2)
        rows = {kind: grand_at(placed, coupled=kind == "joint") for kind in ("joint", "uncoupled")}
    errors = {}
    for kind, grand in rows.items():
        squeeze = (grand[0, 0] - grand[0, 6]) / single
        equal = (grand[0, 0] + grand[0, 6]) / single
        errors[kind] = (squeeze / approaching(gap) - 1.0, equal / stimson_jeffery(gap) - 1.0)
    print(f"\nh/a {gap}: the joint node's squeeze {errors['joint'][0]:+.3%} (line {SQUEEZE[gap]:.0%}), equal velocities "
          f"{errors['joint'][1]:+.3%} (line {EQUAL:.1%}, far field only); uncoupled {errors['uncoupled'][0]:+.1%}, "
          f"{errors['uncoupled'][1]:+.1%}")
    assert abs(errors["joint"][0]) <= SQUEEZE[gap] and abs(errors["joint"][1]) <= EQUAL
    assert all(abs(errors["uncoupled"][k]) >= 2.0 * abs(errors["joint"][k]) for k in (0, 1))


def rotne_prager(r: np.ndarray) -> np.ndarray:
    """The cross mobility of two unit spheres at unit viscosity, centres ``r`` apart [recalled]."""
    d = float(np.linalg.norm(r))
    e = r / d
    return ((1.0 + 2.0 / (3.0 * d * d)) * np.eye(3) + (1.0 - 2.0 / (d * d)) * np.outer(e, e)) / (8.0 * math.pi * d)


@pytest.mark.parametrize("distance", sorted(OFF_AXIS))
def test_off_axis_the_cross_mobility_against_rotne_prager_and_the_uncoupled_control(distance):
    direction = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    centre = tuple(distance * RADIUS * direction)
    placed = {"s0": (Sphere(radius=m(RADIUS)), (0.0, 0.0, 0.0)), "s1": (Sphere(radius=m(RADIUS)), centre)}
    with precision("float64"):
        found = {kind: grand_at(placed, coupled=kind == "joint") for kind in ("joint", "uncoupled")}
    oracle = rotne_prager(distance * direction)
    errors = {}
    for kind, grand in found.items():
        mobility = np.linalg.inv(grand) * VISCOSITY * RADIUS
        cross = mobility[0:3, 6:9]
        errors[kind] = float(np.linalg.norm(cross - oracle) / np.linalg.norm(oracle))
    print(f"\nr/a {distance}: cross mobility against Rotne-Prager {errors['joint']:.3g} (line {OFF_AXIS[distance]:g}); "
          f"uncoupled {errors['uncoupled']:.3g}")
    assert errors["joint"] <= OFF_AXIS[distance]
    assert errors["uncoupled"] >= 2.0 * errors["joint"]
