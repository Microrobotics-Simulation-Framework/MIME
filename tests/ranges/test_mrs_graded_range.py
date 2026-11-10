"""A graded body over a declared range: ``Drag.MRS(contact=...)`` samples its body at the range's point with the most
graded force points, chosen by ``GRADED_FORCE_POINTS`` (sampled in every number), and every body inside the range takes
that sampling, moved with its own shape, so that a sweep across the range is one group that compiles once.

**Why sampled, the witness and its control.** A graded count is the uniform one's times the density's mean over the
surface, and between the uniform count's steps a larger body's graded patch is a smaller share of it: the sphere's
contact graded for half of a 0.3 mm spacing gives 106 force points at a radius of 0.5 mm and 104 at 0.525 mm, so a
monotone rule, which judges the corners alone, would take a corner that is not the range's most demanding point. The
witness is deterministic (no draw): it fails a monotone declaration on every run. The uniform sphere's radius is
declared monotone (``tests/ranges/test_mrs_monotone.py``), so the rule is the grading's, not the kind's.

**The gate.** The sphere of radius 1 mm above a plane at a 0.6 mm spacing, its contact graded for half of it, over the
range [1.0, 1.1] mm of its radius: its nine sampled points hold more force points inside the range than at either
corner, the node takes the most, and a sweep of three radii across the range compiles once. Float64."""

from __future__ import annotations

import pytest
from graded_cases import above_plane

from mime.interactions.hydrodynamic import (
    FORCE_POINTS_MONOTONE,
    GRADED_FORCE_POINTS,
    force_points_rule,
    graded_force_points,
)
from mime.registry import ChoiceRule, Hull
from mime.scene import Sphere
from mime.scene._grading import Contact, graded_density
from mime.study import Study, grid
from mime.study._ranges import forget
from mime.units import m, mm, ms

FRAME = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
PATH = "ball.parts.shell.radius"
#: The gate's spacing and range (mm).
SPACING, LOW, HIGH = 0.6e-3, 1.0, 1.1


def hull(radius: float, ranges: dict[str, tuple[float, float]] | None = None) -> Hull:
    return Hull("ball", "shell", Sphere(radius=m(radius)), {"radius": radius}, (0.0, 0.0, 0.0), FRAME, ranges or {})


def graded_count(radius: float, spacing: float) -> float:
    held = hull(radius)
    density = graded_density(held.kind, held.settings, spacing, features=False,
                             contact=Contact((0.0, 0.0, -1.0), 0.5 * spacing))
    return graded_force_points(held, spacing, ("surface",), density)


def ranged(radius: float = LOW, *, over: tuple[float, float] | None = (LOW, HIGH)):
    """The graded sphere of ``radius`` mm, its lowest point a spacing above the plane, with the range ``over`` (mm) of
    its radius declared."""
    experiment = above_plane(Sphere(radius=mm(radius)), (0.0, 0.0, radius * 1e-3 + SPACING), (1.0, 0.0, 0.0, 0.0),
                             blob=0.25, name="graded_ranged", spacing=SPACING, contact=(0.0, 0.0, -1.0),
                             contact_gap=m(0.5 * SPACING))
    if over is not None:
        experiment.declare_range(PATH, mm(over[0]), mm(over[1]), reason="the design range of the ball's radius")
    return experiment


def test_the_graded_count_falls_as_the_sphere_grows_the_witness_its_sampled_rule_rests_on():
    assert (graded_count(0.5e-3, 0.3e-3), graded_count(0.525e-3, 0.3e-3)) == (106.0, 104.0)


def test_control_a_monotone_rule_would_judge_the_corners_and_miss_the_ranges_most_demanding_point():
    """The gate's range sampled at the rule's nine points holds more graded force points inside than at either corner,
    so a monotone rule fails it here on every run; the uniform rule on the same hull is monotone."""
    assert isinstance(GRADED_FORCE_POINTS, ChoiceRule.Sampled) and GRADED_FORCE_POINTS.count == 9
    assert force_points_rule(hull(1e-3, {"radius": (1e-3, 1.1e-3)})) is FORCE_POINTS_MONOTONE
    points = [LOW + (HIGH - LOW) * i / 8 for i in range(9)]
    counts = [graded_count(r * 1e-3, SPACING) for r in points]
    assert max(counts) > max(counts[0], counts[-1]), counts


@pytest.fixture(scope="module")
def swept():
    forget()
    study = Study(ranged()).sweep(grid({PATH: [mm(r) for r in (1.0, 1.05, 1.1)]}), duration=ms(2))
    return study.plan(), study.run()


def test_the_graded_body_takes_the_ranges_most_graded_force_points_and_a_sweep_compiles_once(swept):
    plan, result = swept
    assert [len(group.members) for group in plan.groups] == [3], plan.text(memory=False)
    assert result.compiles == (1,)
    (group,) = plan.groups
    assert group.built is not None
    node = group.built.graph.get_node("drag:ball")
    counts = [graded_count(r * 1e-3, SPACING) for r in (LOW + (HIGH - LOW) * i / 8 for i in range(9))]
    assert node.sample_count == max(counts) > graded_count(LOW * 1e-3, SPACING)
    (choice,) = [c for c in group.built.plans.range_choices if c.what == "force points"]
    assert choice.rule is GRADED_FORCE_POINTS and choice.chosen == max(counts)
    assert plan.range_check is not None and plan.range_check.refusals == ()


def test_the_uniform_sphere_over_the_same_range_keeps_its_monotone_rule():
    experiment = above_plane(Sphere(radius=mm(1.0)), (0.0, 0.0, 1e-3 + SPACING), (1.0, 0.0, 0.0, 0.0), blob=0.25,
                             name="uniform_ranged", spacing=SPACING)
    experiment.declare_range(PATH, mm(LOW), mm(HIGH), reason="the design range of the ball's radius")
    (choice,) = [c for c in experiment.plan().range_choices if c.what == "force points"]
    assert choice.rule is FORCE_POINTS_MONOTONE
