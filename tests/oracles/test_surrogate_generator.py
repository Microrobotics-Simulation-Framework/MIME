"""The generator of a screw surrogate's training data and its held-out protocol (``mime.surrogates.generator``).

- **The domain**: de Jongh's seventeen designs in the four tubes, resting on the floor at contact margins up to
  0.2 mm and no nearer the wall than the swim speed's near-wall edge; every case the protocol draws lies inside it,
  inside the tube table's readable radius and within its axial reach, and what lies outside is named (the whole
  resistance's edge, past the largest margin, among it).
- **The protocol**: the training margins run geometrically from the edge to the largest margin; the held-out sets
  (``between``, the midpoints; ``roll``, the body turned about its axis) share no case with the training set. The
  control: a set drawn at the training margins does share them, and the check finds every one.
- **Through ``Drag.BEM`` in the tube wall kind**: a coarse case's resistance is read inside the table, finite,
  symmetric and positive definite, in the canonical frame, where the body's spin about the tube's axis drives it along
  the axis with the 0.2 line's sign, and the same after a half turn about its own axis (a two-start section's period).
  The data derived through the generator record the shipped table's hash as their ``wall:tube`` input, and are not
  stale; the control: the same record naming another table is.
- **A failed check produces no data**: a case the table cannot read (the body pushed into the wall) is refused by
  ``derive``, on every run. The generator refuses float32.

The tube's cases read the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job)."""

import math
from dataclasses import replace

import jax
import numpy as np
import pytest

from mime import assets
from mime.assets import DerivationInput
from mime.build import BuildRefused
from mime.nodes.hydro.bem._tube import UNIT_TUBE, shipped_unit_tube
from mime.surrogates import current_inputs
from mime.surrogates import generator as g

#: The coarse spacing the tests evaluate at: a few hundred samples, inside the tests' time.
COARSE = g.Discretisation(0.6e-3)
SPACING = 0.15e-3
TABLE = assets.reference(UNIT_TUBE)


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def test_the_domain_is_de_jonghs_seventeen_designs_in_the_four_tubes():
    assert [d.name for d in g.DESIGNS] == [f"FL-{i}" for i in range(1, 12)] + [f"FW-{i}" for i in range(1, 7)]
    assert {d.length for d in g.DESIGNS[:11]} == {7.47e-3}
    assert all((d.radius, d.modulation, d.starts) == (1.56e-3, 0.33, 2) for d in g.DESIGNS)
    assert sorted(g.TUBES.values()) == [2.38125e-3, 3.175e-3, 4.7625e-3, 6.35e-3]
    assert g.DOMAIN.margin == (0.02e-3, 0.20e-3)
    assert g.DOMAIN.edge(SPACING) == g.NEAR_WALL_RATIOS["swim_speed"] * SPACING
    named = " ".join(g.DOMAIN.outside)
    for outside in ("free space", "off the floor", "near-wall edge", "turned about its own axis", "tilted", "flow"):
        assert outside in named
    assert g.DOMAIN.edge(SPACING, "resistance") > g.MARGIN[1] > g.DOMAIN.edge(SPACING)


def every_case() -> list[g.Case]:
    held = g.held_out_cases(g.DOMAIN, SPACING)
    return [*g.training_cases(g.DOMAIN, SPACING), *held["between"], *held["roll"]]


def test_the_training_margins_run_geometrically_from_the_edge_to_the_largest_margin():
    gaps = np.array(g.training_margins(g.DOMAIN, SPACING))
    assert len(gaps) == g.TRAINING_MARGINS and gaps[0] == pytest.approx(max(g.MARGIN[0], g.DOMAIN.edge(SPACING)))
    assert gaps[-1] == pytest.approx(g.MARGIN[1]) and np.allclose(gaps[1:] / gaps[:-1], gaps[1] / gaps[0])
    assert g.training_margins(g.DOMAIN, 1e-3) == (), "an edge past the largest margin leaves no training gap"


def test_every_case_lies_inside_the_domain_the_tables_readable_radius_and_its_axial_reach():
    table = shipped_unit_tube()
    readable, reach = table.axes[0].readable_range[1], table.axes[3].readable_range[1]
    cases = every_case()
    assert len(cases) == 17 * 4 * (g.TRAINING_MARGINS + g.TRAINING_MARGINS - 1 + len(g.ROLLS))
    for case in cases:
        assert g.DOMAIN.problems(case, SPACING) == [], case.key()
        assert case.offset > 0.0 and (case.offset + case.design.outer_radius) / case.tube_radius < readable
        assert case.design.length / case.tube_radius < reach


def test_the_held_out_sets_share_no_case_with_the_training_set():
    training = g.training_cases(g.DOMAIN, SPACING)
    for name, cases in g.held_out_cases(g.DOMAIN, SPACING).items():
        assert cases and g.overlap(cases, training) == [], name


def test_control_a_set_drawn_at_the_training_margins_is_found_to_overlap_in_every_case():
    training = g.training_cases(g.DOMAIN, SPACING)
    planted = tuple(replace(case) for case in training[::5])
    assert len(g.overlap(planted, training)) == len(planted)


def test_control_a_gap_below_the_edge_or_above_the_largest_margin_lies_outside_the_domain():
    design = g.design_named("FL-9")
    for gap in (0.5 * g.DOMAIN.edge(SPACING), 1.5 * g.MARGIN[1]):
        assert g.DOMAIN.problems(g.Case(design, "1/4 in", gap), SPACING)
    assert g.DOMAIN.problems(g.Case(design, "1 in", 0.1e-3), SPACING)


@pytest.fixture(scope="module")
def evaluator() -> g.BEMEvaluator:
    return g.BEMEvaluator(g.design_named("FL-9"), "1/4 in", COARSE)


def test_a_case_through_drag_bem_in_the_tube_wall_kind_passes_its_checks(evaluator: g.BEMEvaluator):
    sample = evaluator(g.Case(evaluator.design, "1/4 in", 0.2e-3))
    print(f"\n{sample.samples} samples; checks {[(c.name, c.measured) for c in sample.checks()]}")
    assert sample.accepted and all(check.passed for check in sample.checks())
    assert sample.swim()[2] > 0.0, "FL-9 spun about +z swims along +z, the sign of the 0.2 line's swim speeds for it"


def test_a_half_turn_about_the_bodys_own_axis_leaves_the_resistance_unchanged(evaluator: g.BEMEvaluator):
    """Turning the body half a turn about its own axis maps a two-start screw's section onto itself, so the samples,
    the pose's frame and the resistance are the same: a check of the pose the evaluator builds from a case's roll."""
    case = g.Case(evaluator.design, "1/4 in", 0.2e-3)
    turned = evaluator(replace(case, roll=math.pi)).resistance
    held = evaluator(case).resistance
    assert np.max(np.abs(turned - held)) <= 1e-6 * np.max(np.abs(held))


def test_the_derived_data_record_the_shipped_tables_hash_and_are_not_stale():
    parameters = {"split": "listed", "discretisation": COARSE.as_json(),
                  "cases": [{"design": "FL-9", "tube": "1/4 in", "gap": 0.2e-3}]}
    content, record = assets.derive("tiny", g.TrainingData(), parameters, {"wall:tube": TABLE}, commit="test",
                                    domain=g.DOMAIN.as_json())
    assert record.input("wall:tube") == DerivationInput("wall:tube", UNIT_TUBE, TABLE.sha256)
    assert assets.stale_inputs(record, current_inputs()) == []
    with np.load(__import__("io").BytesIO(content)) as held:
        assert held["resistance"].shape == (1, 6, 6) and bool(held["accepted"][0])
    planted = replace(record, inputs=(DerivationInput("wall:tube", UNIT_TUBE, "0" * 64),))
    assert assets.stale_inputs(planted, current_inputs())


def test_control_a_case_the_table_cannot_read_produces_no_data():
    """The body pushed 0.2 mm into the wall: its samples lie past the table's readable radius, so the read is refused,
    and with it the case. (A gap of a micrometre is still read at this spacing: the samples miss the ridge's crest by
    more than that, which is why the near-wall edge is measured against the spacing.)"""
    parameters = {"split": "listed", "discretisation": COARSE.as_json(),
                  "cases": [{"design": "FL-9", "tube": "1/2 in", "gap": -0.2e-3}]}
    with pytest.raises(BuildRefused) as refused:
        assets.derive("too_near", g.TrainingData(), parameters, {"wall:tube": TABLE}, commit="test", domain={})
    assert refused.value.codes() == ["asset.derivation-check"] and "wall_accepted" in str(refused.value)


def test_the_generator_refuses_float32():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            g.BEMEvaluator(g.design_named("FL-9"), "1/4 in", COARSE)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_accuracy_is_judged_per_use_on_the_resistance_and_the_swim_speed(evaluator: g.BEMEvaluator):
    reference = [evaluator(g.Case(evaluator.design, "1/4 in", gap)) for gap in (0.1e-3, 0.2e-3)]
    exact = g.accuracy([s.resistance for s in reference], reference, over="two gaps")
    assert [(a.use, a.value) for a in exact] == [("resistance", 0.0), ("swim_speed", 0.0)]
    coupled = []
    for sample in reference:
        held = sample.resistance.copy()
        held[:3, 3:] *= 1.01
        held[3:, :3] *= 1.01
        coupled.append(held)
    found = {a.use: a.value for a in g.accuracy(coupled, reference, over="two gaps")}
    assert found["swim_speed"] == pytest.approx(0.01, rel=1e-6) and found["resistance"] == pytest.approx(0.01, rel=1e-6)
    for each in g.accuracy(coupled, reference, over="two gaps"):
        assert "bounds nothing about the flow" in each.claim
