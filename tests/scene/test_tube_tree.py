"""The tube tree (``mime.scene.TubeTree``): its declaration's refusals, its junctions and which tubes meet, its
centreline's contract (``mime.testing.centreline_problems``), its fillet, its placement, the held plan's guard, and the
walls' resolution report. The union's gates are ``test_exact_union.py``'s."""

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from union_oracles import D, hairpin_tree, lattice, loop_tree, rotation, seed_list, y_tree

from mime.build import ArgumentRefused
from mime.core import jnp
from mime.scene import (
    CURVATURE_LINE,
    FILLET_SHARE,
    LAYER_LINES,
    Helix,
    TubeTree,
    steady_layer,
    stokes_layer,
    wall_resolution,
)
from mime.testing import centreline_problems
from mime.units import mm

UNION_PLANT = os.environ.get("MIME_PLANT_UNION")
#: The shift the curvature line is set for, in cells: FA4's line.
CURVATURE_SHIFT = 0.1
#: The sweep's major vertex's radius of curvature, b^2 / a, in diameters.
VERTEX = 0.3 ** 2 / 0.5
#: The shares of the wall shear's error each layer line is set for.
LAYER_SHARES = {"wall shear within 5 %": 0.05, "wall shear within 2 %": 0.02}
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
        TubeTree.si(points, tuple(map(tuple, segments)), radii, fillet)
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
    tree = TubeTree.si(((0.0, 0.0, 0.0), (1e-3, 0.0, 0.0), (0.0, 1e-3, 0.0), (1e-3, 1e-3, 0.0)), ((0, 1), (2, 3)),
                    ((1e-4, 1e-4), (1e-4, 1e-4)))
    assert tree.meets() == ()
    assert tree.junctions == ()
    assert float(tree.fillet_radius(tree.parameters())) == 0.0


@pytest.mark.parametrize("tree", [y_tree("asymmetric"), y_tree("spike"), loop_tree()], ids=["y", "spike", "loop"])
def test_the_centreline_keeps_the_contract_through_its_junctions(tree):
    assert centreline_problems(LABEL, tree.parameters(), tree.centreline()) == []


def test_the_centrelines_radius_is_linear_between_its_ends():
    tree = TubeTree.si(((0.0, 0.0, 0.0), (2e-3, 0.0, 0.0)), ((0, 1),), ((0.4e-3, 0.3e-3),))
    segment = tree.centreline().segments[0]
    radius = np.asarray(segment.radius(parameters_of(tree), jnp.asarray([0.0, 1e-3, 2e-3])))
    assert radius == pytest.approx([0.4e-3, 0.35e-3, 0.3e-3], rel=1e-14)


def test_the_fillets_radius_is_its_share_of_the_smallest_radius_at_a_junction():
    tree = y_tree("asymmetric", FILLET_SHARE)
    assert tree.fillet == 0.1
    assert tree.fillet_radius_value() == pytest.approx(0.1 * 0.3 * D, rel=1e-15)


def test_a_round_cone_is_the_union_of_its_balls():
    """A tapered segment's distance against the least over dense balls along it (its definition), outside it."""
    tree = TubeTree.si(((0.0, 0.0, 0.0), (2e-3, 0.0, 0.0)), ((0, 1),), ((0.5e-3, 0.2e-3),))
    rng = np.random.default_rng(5)
    points = rng.uniform((-1e-3, -1e-3, -1e-3), (3e-3, 1e-3, 1e-3), (500, 3))
    found = np.asarray(tree.lumen().signed_distance(parameters_of(tree), jnp.asarray(points)).distance)
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
    found = lumen.outward_normal(held, points, plan)
    distance, normal = found.distance.distance, found.normal
    lengths = np.asarray(jnp.linalg.norm(normal, axis=-1))
    off_medial = np.abs(lengths - 1.0) < 1e-6
    assert off_medial.mean() > 0.9
    nearest = points - distance[:, None] * normal
    on_wall = lumen.signed_distance(held, nearest, lumen.plan(held, nearest, margin=0.0)).distance
    assert np.max(np.abs(np.asarray(on_wall))[off_medial]) <= 1e-12


def test_a_turned_and_moved_tree_turns_and_moves_its_distance():
    tree = y_tree("asymmetric", 0.1)
    turn, offset = rotation(9), np.asarray([3e-5, -2e-5, 1e-5])
    moved = tree.turned(turn, offset)
    points = np.random.default_rng(7).uniform(-0.8 * D, 0.8 * D, (300, 3))
    here_lumen, there_lumen = tree.lumen(), moved.lumen()
    held, moved_held = parameters_of(tree), parameters_of(moved)
    at_here, at_there = jnp.asarray(points), jnp.asarray(points @ turn.T + offset)
    here = np.asarray(here_lumen.signed_distance(held, at_here, here_lumen.plan(held, at_here, margin=0.0)).distance)
    there = np.asarray(there_lumen.signed_distance(moved_held, at_there,
                                                   there_lumen.plan(moved_held, at_there, margin=0.0)).distance)
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


@pytest.mark.parametrize("seed", seed_list((51,)))
def test_a_study_that_brings_tubes_within_reach_is_counted(seed):
    """A hairpin whose legs do not meet as made (walls 0.2 to 0.4 D apart, turned and moved at random), and a study that
    scales its radii until they come within reach (the fillet's widening included): the pair the union does not hold
    as meeting is counted at the run's values, with the distance and with the normal; at the values it was made with,
    none. **Control** (``MIME_PLANT_UNION=no-unplanned``): the count left out."""
    rng = np.random.default_rng(seed)
    wall = rng.uniform(0.2, 0.4) * D
    tree = hairpin_tree(wall, 0.1).turned(rotation(seed), rng.uniform(-D, D, 3))
    lumen = tree.lumen()
    if UNION_PLANT == "no-unplanned":
        lumen = dataclasses.replace(lumen, unplanned=lambda parameters: 0)
    assert lumen.meets == ((0, 1), (1, 2))
    held = parameters_of(tree)
    middle = 0.25 * np.sum(tree.parameters()["points"], axis=0)
    points = jnp.asarray(lattice(middle, 0.3 * D, D / 16.0))
    assert int(lumen.signed_distance(held, points).unplanned) == 0
    within = (D + wall) / (1.1 * D)
    study = {**held, "radii": held["radii"] * rng.uniform(within + 0.01, within + 0.2)}
    assert int(lumen.signed_distance(study, points).unplanned) == 1
    assert int(lumen.outward_normal(study, points).distance.unplanned) == 1


def test_the_report_names_tubes_the_fillet_joins():
    """A hairpin whose wall is thinner than the fillet's diameter is joined by its closing, and the report says so;
    the narrowest solid is read between tubes sharing no point whether or not they meet (a wall of 0.12 D, which the
    fillet does not fill, is read too); the loop's island between its branches is solid, its touching ends' gap
    lumen."""
    fused = hairpin_tree(0.06 * D, 0.1).wall_features()
    assert fused.fused == ((0, 2),) and fused.narrowest_solid == pytest.approx(0.06 * D, rel=1e-9)
    entries = wall_resolution(fused, D / 16.0).entries
    joined = next(entry for entry in entries if entry.name.startswith("solid the fillet fills"))
    assert joined.cells == pytest.approx(0.96, rel=1e-9) and "joins these tubes" in joined.text()
    apart = hairpin_tree(0.12 * D, 0.1).wall_features()
    assert apart.fused == () and apart.narrowest_solid == pytest.approx(0.12 * D, rel=1e-9)
    loop = loop_tree(0.1).wall_features()
    assert loop.fused == () and 0.0 < loop.narrowest_solid < 0.3 * D


def test_the_layer_lines_are_the_sweeps_worst_arm():
    """``LAYER_LINES`` against the Womersley sweep's rows they are set from: past each line every row's wall shear is
    within its share, and the line is the worst arm's crossing of the share, interpolated between the rows either side
    of it, rounded up to a tenth of a cell."""
    rows = json.loads(Path(__file__).with_name("womersley_sweep_rows.json").read_text())["rows"]
    worst: dict[float, float] = {}
    for row in rows:
        layer = row["layer_cells"]
        worst[layer] = max(worst.get(layer, 0.0), abs(row["wall_shear_amplitude_error"]))
    layers = sorted(worst)
    for name, line in LAYER_LINES.items():
        share = LAYER_SHARES[name]
        assert all(worst[layer] <= share for layer in layers if layer >= line), name
        above = max(layer for layer in layers if worst[layer] > share)
        below = min(layer for layer in layers if layer > above)
        crossing = above + (worst[above] - share) / (worst[above] - worst[below]) * (below - above)
        assert crossing <= line <= math.ceil(crossing * 10.0) / 10.0 + 1e-12, (name, crossing)


def test_the_curvature_line_is_the_sweeps():
    """``CURVATURE_LINE`` against the elliptic section's sweep it is set from (FA4's reading, each semi-axis's shift less
    a round tube's, cells, over turns and sub-cell offsets at resolutions putting the major vertex at 2.9 to 8.6
    cells): every row at or above the line within 0.1 cells, and the line the least swept radius above the last one
    with a row past 0.1, rounded up to a tenth of a cell. The reading is not monotone in the radius (5.04 cells read
    0.117 between 4.32's 0.030 and 5.22's 0.076), so the line is the sweep's, at the radii it swept, not a bound
    between them."""
    held = json.loads(Path(__file__).with_name("curvature_sweep_rows.json").read_text())
    worst: dict[float, float] = {}
    for values in held["rows"]:
        row = dict(zip(held["columns"], values, strict=True))
        radius = VERTEX * row["dh"]
        worst[radius] = max(worst.get(radius, 0.0), abs(row["major"]), abs(row["minor"]))
    radii = sorted(worst)
    assert all(worst[radius] <= CURVATURE_SHIFT for radius in radii if radius >= CURVATURE_LINE)
    above = max(radius for radius in radii if worst[radius] > CURVATURE_SHIFT)
    least = min(radius for radius in radii if radius > above)
    assert least <= CURVATURE_LINE <= math.ceil(least * 10.0) / 10.0 + 1e-12, (above, least)


def test_the_features_the_report_reads():
    tree = y_tree("asymmetric", 0.1)
    features = tree.wall_features()
    assert features.smallest_radius == pytest.approx(0.03 * D)
    assert features.narrowest_lumen == pytest.approx(0.6 * D)
    assert features.narrowest_solid == math.inf
    assert features.sharpest_wedge == pytest.approx(60.0)
    sharp = y_tree("asymmetric", 0.0).wall_features()
    assert sharp.smallest_radius == 0.0


def test_the_walls_resolution_report():
    features = y_tree("asymmetric", 0.1).wall_features()
    h = D / 16.0
    report = wall_resolution(features, h, layer=stokes_layer(0.5 * D, 3.0))
    assert report.entry("smallest wall radius").cells == pytest.approx(0.48)
    assert report.entry("smallest wall radius").line == CURVATURE_LINE
    assert report.entry("smallest wall radius").below is True
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
