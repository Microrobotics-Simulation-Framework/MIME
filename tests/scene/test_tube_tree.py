"""The tube tree (``mime.scene.TubeTree``): its declaration's refusals, its junctions and which tubes meet, its
centreline's contract (``mime.testing.centreline_problems``), its fillet, its placement, the held plan's guard, and the
walls' resolution report. The union's gates are ``test_exact_union.py``'s."""

import math

import numpy as np
import pytest
from union_oracles import D, loop_tree, rotation, y_tree

from mime.build import ArgumentRefused
from mime.core import jnp
from mime.scene import (
    FILLET_SHARE,
    Helix,
    TubeTree,
    steady_layer,
    stokes_layer,
    wall_resolution,
)
from mime.testing import centreline_problems
from mime.units import mm

#: A kind that labels the centreline harness's lines; the tree's own centreline is what is held.
LABEL = Helix(radius=mm(0.15), pitch=mm(0.6), turns=2.5, filament_radius=mm(0.03))


def parameters_of(tree: TubeTree) -> dict[str, jnp.ndarray]:
    return {name: jnp.asarray(value) for name, value in tree.parameters().items()}


@pytest.mark.parametrize("points, segments, radii, fillet, said", [
    ([(0, 0, 0)], [(0, 0)], [(1e-4, 1e-4)], 0.1, "two or more"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 0)], [(1e-4, 1e-4)], 0.1, "two distinct points"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 2)], [(1e-4, 1e-4)], 0.1, "two distinct points"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 1), (1, 0)], [(1e-4, 1e-4)] * 2, 0.1, "the same two points"),
    ([(0, 0, 0), (1e-3, 0, 0), (2e-3, 0, 0)], [(0, 1)], [(1e-4, 1e-4)], 0.1, "every point"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 1)], [(1e-4, 0.0)], 0.1, "radii are two above zero"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 1)], [(1e-4,)], 0.1, "radii are two above zero"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 1)], [(2e-3, 1e-4)], 0.1, "differ by less than its length"),
    ([(0, 0, 0), (1e-3, 0, 0)], [(0, 1)], [(1e-4, 1e-4)], -0.1, "fillet share"),
    ([(0, 0, 0), (1e-3, 0, math.nan)], [(0, 1)], [(1e-4, 1e-4)], 0.1, "finite"),
])
def test_a_malformed_tree_is_refused_with_its_code(points, segments, radii, fillet, said):
    with pytest.raises(ArgumentRefused) as raised:
        TubeTree(tuple(map(tuple, points)), tuple(map(tuple, segments)), tuple(map(tuple, radii)), fillet)
    refusal = raised.value.refusals[0]
    assert refusal.code == "call.tube_tree" and said in refusal.message


def test_the_ys_junction_and_meeting_tubes():
    tree = y_tree("asymmetric")
    assert [(j.name, j.ends) for j in tree.junctions] == [("j1", ((0, 1), (1, 0), (2, 0)))]
    assert tree.meets() == ((0, 1), (0, 2), (1, 2))
    assert tree.lumen().corners == ((0, 1, 2),)


def test_the_loops_junctions_and_its_tubes_that_meet_without_sharing_a_point():
    tree = loop_tree()
    assert [j.name for j in tree.junctions] == ["j1", "j2", "j3", "j4", "j5", "j6"]
    sharing = {(i, j) for i in range(8) for j in range(i + 1, 8) if set(tree.segments[i]) & set(tree.segments[j])}
    assert sharing <= set(tree.meets())
    assert (1, 5) in set(tree.meets()) - sharing, "the daughter's two ends' balls come within reach across the middle"


def test_tubes_far_apart_do_not_meet():
    tree = TubeTree(((0.0, 0.0, 0.0), (1e-3, 0.0, 0.0), (0.0, 1e-3, 0.0), (1e-3, 1e-3, 0.0)), ((0, 1), (2, 3)),
                    ((1e-4, 1e-4), (1e-4, 1e-4)))
    assert tree.meets() == ()
    assert tree.junctions == ()
    assert float(tree.fillet_radius(tree.parameters())) == 0.0


@pytest.mark.parametrize("tree", [y_tree("asymmetric"), y_tree("spike"), loop_tree()], ids=["y", "spike", "loop"])
def test_the_centreline_keeps_the_contract_through_its_junctions(tree):
    assert centreline_problems(LABEL, tree.parameters(), tree.centreline()) == []


def test_the_centrelines_radius_is_linear_between_its_ends():
    tree = TubeTree(((0.0, 0.0, 0.0), (2e-3, 0.0, 0.0)), ((0, 1),), ((0.4e-3, 0.3e-3),))
    segment = tree.centreline().segments[0]
    radius = np.asarray(segment.radius(parameters_of(tree), jnp.asarray([0.0, 1e-3, 2e-3])))
    assert radius == pytest.approx([0.4e-3, 0.35e-3, 0.3e-3], rel=1e-14)


def test_the_fillets_radius_is_its_share_of_the_smallest_radius_at_a_junction():
    tree = y_tree("asymmetric", FILLET_SHARE)
    assert tree.fillet == 0.1
    assert tree.fillet_radius_value() == pytest.approx(0.1 * 0.3 * D, rel=1e-15)


def test_a_round_cone_is_the_union_of_its_balls():
    """A tapered segment's distance against the least over dense balls along it (its definition), outside it."""
    tree = TubeTree(((0.0, 0.0, 0.0), (2e-3, 0.0, 0.0)), ((0, 1),), ((0.5e-3, 0.2e-3),))
    rng = np.random.default_rng(5)
    points = rng.uniform((-1e-3, -1e-3, -1e-3), (3e-3, 1e-3, 1e-3), (500, 3))
    found = np.asarray(tree.lumen().distance(parameters_of(tree), jnp.asarray(points)))
    s = np.linspace(0.0, 1.0, 20001)
    centres = s[:, None] * np.asarray([2e-3, 0.0, 0.0])
    radii = 0.5e-3 + s * (0.2e-3 - 0.5e-3)
    outside = found > 0.0
    balls = np.min(np.linalg.norm(points[outside][:, None] - centres[None], axis=-1) - radii[None], axis=1)
    assert np.max(np.abs(balls - found[outside])) <= 1e-9


def test_its_normal_is_unit_and_its_nearest_point_on_the_wall():
    tree = y_tree("asymmetric", 0.1)
    lumen = tree.lumen()
    held = parameters_of(tree)
    points = jnp.asarray(np.random.default_rng(6).uniform(-0.8 * D, 0.8 * D, (400, 3)))
    plan = lumen.plan(held, points, margin=0.0)
    distance = lumen.distance(held, points, plan)
    normal = lumen.normal(held, points, plan)
    lengths = np.asarray(jnp.linalg.norm(normal, axis=-1))
    off_medial = np.abs(lengths - 1.0) < 1e-6
    assert off_medial.mean() > 0.9
    nearest = points - distance[:, None] * normal
    assert np.max(np.abs(np.asarray(lumen.distance(held, nearest, lumen.plan(held, nearest, margin=0.0))))
                  [off_medial]) <= 1e-12


def test_a_turned_and_moved_tree_turns_and_moves_its_distance():
    tree = y_tree("asymmetric", 0.1)
    turn, offset = rotation(9), np.asarray([3e-5, -2e-5, 1e-5])
    moved = tree.turned(turn, offset)
    points = np.random.default_rng(7).uniform(-0.8 * D, 0.8 * D, (300, 3))
    here_lumen, there_lumen = tree.lumen(), moved.lumen()
    held, moved_held = parameters_of(tree), parameters_of(moved)
    at_here, at_there = jnp.asarray(points), jnp.asarray(points @ turn.T + offset)
    here = np.asarray(here_lumen.distance(held, at_here, here_lumen.plan(held, at_here, margin=0.0)))
    there = np.asarray(there_lumen.distance(moved_held, at_there, there_lumen.plan(moved_held, at_there, margin=0.0)))
    assert np.max(np.abs(here - there)) <= 1e-15


def test_points_outside_the_plan_that_fail_the_check_are_counted():
    """The plan is held: walls moved further than its margin leave failing points outside it, counted, never silent;
    at the values it was chosen at, none."""
    tree = y_tree("asymmetric")
    lumen = tree.lumen()
    held = parameters_of(tree)
    points = jnp.asarray(np.random.default_rng(8).uniform(-0.8 * D, 0.8 * D, (4000, 3)))
    plan = lumen.plan(held, points, margin=0.0)
    assert int(lumen.signed_distance(held, points, plan).unrefined) == 0
    wider = {**held, "radii": held["radii"] * 1.1}
    assert int(lumen.signed_distance(wider, points, plan).unrefined) > 0
    margined = lumen.plan(held, points, margin=0.2 * D)
    assert int(lumen.signed_distance(wider, points, margined).unrefined) == 0


def test_the_features_the_report_reads():
    tree = y_tree("asymmetric", 0.1)
    features = tree.features()
    assert features.smallest_radius == pytest.approx(0.03 * D)
    assert features.narrowest_lumen == pytest.approx(0.6 * D)
    assert features.narrowest_solid == math.inf
    assert features.sharpest_wedge == pytest.approx(60.0)
    sharp = y_tree("asymmetric", 0.0).features()
    assert sharp.smallest_radius == 0.0


def test_the_walls_resolution_report():
    features = y_tree("asymmetric", 0.1).features()
    h = D / 16.0
    report = wall_resolution(features, h, layer=stokes_layer(0.5 * D, 3.0))
    assert report.entry("smallest wall radius").cells == pytest.approx(0.48)
    assert report.entry("smallest wall radius").below is None
    lumen = report.entry("narrowest lumen")
    assert lumen.cells == pytest.approx(9.6) and lumen.below is False
    wedge = next(e for e in report.entries if e.name.startswith("sharpest wedge"))
    assert wedge.cells == pytest.approx(1.0 / math.tan(math.radians(30.0)))
    layer = report.entry("wall layer, for wall shear within 5 %")
    assert layer.cells == pytest.approx(0.5 * math.sqrt(2.0) / 3.0 * 16.0) and layer.below is False
    assert report.entry("wall layer, for wall shear within 2 %").below is True
    assert any("refine the wall here" in line for line in report.lines())
    assert steady_layer(1e-3, 100.0) == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        wall_resolution(features, 0.0)
