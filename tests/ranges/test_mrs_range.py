"""The first consumer of a declared range: ``Drag.MRS`` samples its body over the range of the body's radius, at the
range's corner with the most force points, and keeps the body's own radius as its leaf, so that the chosen samples
move with the shape. On the pulled sphere ([1.0, 1.15] mm at a written spacing of 0.6 mm): 91 force points where the
sphere alone at 1.0 mm has 66, and the build's price report says so. With the default spacing, the spacing is the
family's at the range's smallest body (the finest across the range), at which every body inside has at least its
family's count."""

from __future__ import annotations

import numpy as np
import pytest
from ranged_rigs import RADIUS, REASON, pulled

from mime.interactions import Drag
from mime.units import mm

NODE = "drag:ball"


@pytest.fixture(scope="module")
def ranged():
    return pulled(1.0).build()


@pytest.fixture(scope="module")
def alone():
    return pulled(1.0, ranged=None).build()


def test_the_rung_samples_at_the_ranges_corner_with_the_most_force_points_and_keeps_its_own_radius(ranged, alone):
    node, own = ranged.graph.get_node(NODE), alone.graph.get_node(NODE)
    assert (node.sample_count, own.sample_count) == (91, 66)
    assert float(node.params["shape_radius"]) == float(own.params["shape_radius"]) == 1e-3
    held = np.asarray(node.samples(node.params).points)
    eps = float(np.finfo(held.dtype).eps)
    points = held.astype(np.float64)
    assert np.allclose(np.linalg.norm(points, axis=1), 1e-3, rtol=16 * eps, atol=0.0), "at the body's own radius"
    other = pulled(1.15, ranged=None).build().graph.get_node(NODE)
    corner = np.asarray(other.samples(other.params).points).astype(np.float64)
    assert np.allclose(points * 1.15, corner, rtol=16 * eps, atol=16 * eps * 1.15e-3), "the corner's samples, scaled"


def test_the_price_report_names_the_choice_the_range_moved_and_its_cost(ranged, alone):
    (line,) = ranged.report.ranges
    assert line.startswith(f"{RADIUS} in [0.001, 0.00115] m: ")
    assert "drag:ball's force points 91, chosen at ball.parts.shell.radius 0.00115 m, against 66" in line
    assert "dense solve of 273 unknowns against 198, about 2.6 times the factorisation" in line
    assert "declared ranges, each with the choices it moved" in ranged.report.text()
    assert alone.report.ranges == ()
    (choice,) = [c for c in ranged.plans.range_choices if c.what == "force points"]
    assert choice.own == alone.graph.get_node(NODE).sample_count, "the price's own count is a build alone's"


def test_at_the_corner_it_was_chosen_at_the_ranged_build_is_the_build_alone_bit_for_bit():
    corner, alone = pulled(1.15).build(), pulled(1.15, ranged=None).build()
    a, b = corner.graph.get_node(NODE), alone.graph.get_node(NODE)
    assert a.sample_count == b.sample_count == 91
    for key, value in alone.graph.get_node(NODE).params.items():
        assert np.array_equal(np.asarray(a.params[key]), np.asarray(value)), key
    assert corner.report.ranges[0].endswith("no choice of the build moved from the experiment's own value")


def test_with_the_default_spacing_the_range_takes_the_family_spacing_of_its_smallest_body():
    law = {"blob": 0.5, "quadrature_refinement": 4, "wall_kind": "free_space"}

    def build(radius: float, ranged: tuple[float, float] | None):
        from mrs_cases import free_space

        from mime.scene import Sphere
        from mime.units import kg, kg_m2

        experiment = free_space("defaulted", {"ball": Sphere(radius=mm(radius))}, Drag.MRS(**law), mass=kg(2e-8),
                                inertia=kg_m2((2e-17, 2e-17, 2e-17)))
        if ranged is not None:
            experiment.declare_range(RADIUS, mm(ranged[0]), mm(ranged[1]), reason=REASON)
        return experiment.plan()

    small, large, over = build(1.0, None), build(1.15, None), build(1.15, (1.0, 1.15))
    spacing = {name: next(e.settings["spacing"] for e in plans.entries if e.owner == NODE)
               for name, plans in (("small", small), ("large", large), ("over", over))}
    assert spacing["over"] == spacing["small"] < spacing["large"]
    (moved,) = [c for c in over.range_choices if c.what == "default spacing"]
    assert (moved.chosen, moved.own) == (spacing["small"], spacing["large"]) and not moved.short
