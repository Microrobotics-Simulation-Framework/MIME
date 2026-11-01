"""The generator of a screw surrogate's training data and its held-out protocol (``mime.surrogates.generator``).

- **The domain**: de Jongh's seventeen designs in the four tubes, resting on the floor at contact margins up to
  0.2 mm; its **near-wall edge is open, the maintainer's decision** (``OPEN_NEAR_WALL_EDGE``): the domain declares it
  open, with the options and the measured edges as data, and the generator **refuses to produce training data while it
  is open**, by name (``generator.near-wall-edge-open``), for every set and for a listed case. The protocol's mechanics
  (margins, held-out sets, the gate) are exercised on a domain with a *test* edge (``DECLARED``, a number for the
  machinery, not a decision).
- **The protocol**: the training margins run geometrically from the edge to the largest margin; the held-out sets
  (``between``, the midpoints; ``roll``, the body turned about its axis) share no case with the training set. The
  control: a set drawn at the training margins does share them, and the check finds every one.
- **The build gate** refuses by name, before any case runs: a case outside the domain (a gap of 0.02 mm below a
  declared edge, one above the largest margin, a design or tube that is not the domain's) and an empty set.
- **The roll**: a body turned about its own axis is floored again at each roll, so that its nearest point, measured on
  the kind's own surface samples, keeps the case's gap from the wall; the control: the pose floored at the unturned
  roll (the review's FL-1 at a quarter turn) is 0.15 mm off the floor by the same measurement.
- **Through ``Drag.MRS`` in the tube wall kind**: a coarse case's resistance is read inside the table, finite,
  symmetric and positive definite, in the canonical frame, where the body's spin about the tube's axis drives it along
  the axis with the 0.2 line's sign, and the same after a half turn about its own axis (a two-start section's period).
  The data derived through the generator record the shipped table's hash as their ``wall:tube`` input, and are not
  stale; the control: the same record naming another table is.
- **A failed check produces no data**: a case the table cannot read (the body pushed into the wall) is refused by
  ``derive``, on every run. The generator refuses float32.

The tube's cases read the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job)."""

import json
import math
from dataclasses import replace

import jax
import numpy as np
import pytest

from mime import assets
from mime.assets import DerivationInput
from mime.build import BuildRefused
from mime.nodes.hydro.mrs._tube import UNIT_TUBE, shipped_unit_tube
import mime.scene
from mime.scene import KEPT, Screw
from mime.surrogates import current_inputs
from mime.surrogates import generator as g
from mime.units import dimensionless, m

#: The coarse spacing the tests evaluate at: a few hundred samples, inside the tests' time.
COARSE = g.Discretisation(0.6e-3)
SPACING = 0.15e-3
TABLE = assets.reference(UNIT_TUBE)
#: The domain the protocol's mechanics are exercised on: a *test* edge, so that margins, held-out sets and the gate have
#: a gap to start from. A number for the machinery, not a decision (the real edge is open).
DECLARED = replace(g.DOMAIN, near_wall=g.DeclaredEdge({"swim_speed": 0.6}))
#: The same for the coarse spacing (0.6 mm), where 0.6 spacings is past the largest margin.
COARSE_DECLARED = replace(g.DOMAIN, near_wall=g.DeclaredEdge({"swim_speed": 0.1}))

#: How far the resistance at a half turn may be from the unturned's, over its largest entry, at the evaluator's coarse
#: discretisation a fifth of a millimetre from the wall: the quasi-uniform sampling's asymmetry, measured 6.1e-2 there
#: (the screw 0.3.0; on its quadrature, the screw 0.2.0, it was rounding). A declaration, that measurement doubled.
HALF_TURN = 0.12


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def test_the_domain_is_de_jonghs_seventeen_designs_in_the_four_tubes_and_its_near_wall_edge_is_open():
    assert [d.name for d in g.DESIGNS] == [f"FL-{i}" for i in range(1, 12)] + [f"FW-{i}" for i in range(1, 7)]
    assert {d.length for d in g.DESIGNS[:11]} == {7.47e-3}
    assert all((d.radius, d.modulation, d.starts) == (1.56e-3, 0.33, 2) for d in g.DESIGNS)
    assert sorted(g.TUBES.values()) == [2.38125e-3, 3.175e-3, 4.7625e-3, 6.35e-3]
    assert g.DOMAIN.margin == (0.02e-3, 0.20e-3)
    assert g.DOMAIN.near_wall is g.OPEN_NEAR_WALL_EDGE and isinstance(g.DOMAIN.near_wall, g.OpenEdge)
    named = " ".join(g.DOMAIN.outside)
    for outside in ("free space", "off the floor", "near-wall edge", "turned about its own axis", "tilted", "flow"):
        assert outside in named
    assert "open" in named and "maintainer" in named


def test_the_open_edge_is_recorded_with_its_options_its_criterion_and_the_measured_edges_as_data():
    record = g.DOMAIN.as_json()["near_wall"]
    assert isinstance(record, dict) and record["status"] == "open" and "maintainer" in str(record["decision"])
    assert "7i-b is blocked" in str(record["decision"])
    assert len(record["options"]) == 4
    assert record["criterion"] == {"quantities": list(g.NEAR_WALL_USES["resistance"]), "tolerance": 0.01,
                                   "committed_at": "fd6519d"} and len(record["criterion"]["quantities"]) == 8
    measured = {(each["design"], each["tube"], each["far_gap_mm"], each["use"]): each["edge_spacings"]
                for each in record["measured"]}
    assert measured[("FL-9", "1/4 in", 0.5, "resistance")] == pytest.approx(2.083, abs=1e-3)
    assert measured[("FL-9", "1/4 in", 0.5, "swim_speed")] == 0.625
    assert measured[("FL-1", "1/4 in", 0.5, "swim_speed")] == 2.5, "the review's FL-1, 4x FL-9's"
    assert (measured[("FL-1", "1/4 in", 0.3, "swim_speed")], measured[("FL-1", "1/4 in", 0.2, "swim_speed")]) == (
        1.579, 0.789)
    assert measured[("FL-9", "3/16 in", 0.5, "resistance")] is None, "the analysis cannot run in that tube"
    json.dumps(g.DOMAIN.as_json(), allow_nan=False)


def test_a_declared_edge_in_the_open_domains_place_is_a_domain_with_margins():
    assert DECLARED.edge(SPACING) == 0.6 * SPACING and g.training_margins(DECLARED, SPACING)
    assert json.loads(g.protocol_json(DECLARED, g.Discretisation(SPACING)))["edges"] == {"swim_speed": 0.6 * SPACING}
    held = json.loads(g.protocol_json(g.DOMAIN, g.Discretisation(SPACING)))
    assert "open" in held["near_wall_edge"] and "training_margins" not in held


def every_case() -> list[g.Case]:
    held = g.held_out_cases(DECLARED, SPACING)
    return [*g.training_cases(DECLARED, SPACING), *held["between"], *held["roll"]]


def test_the_training_margins_run_geometrically_from_the_edge_to_the_largest_margin():
    gaps = np.array(g.training_margins(DECLARED, SPACING))
    assert len(gaps) == g.TRAINING_MARGINS and gaps[0] == pytest.approx(max(g.MARGIN[0], DECLARED.edge(SPACING)))
    assert gaps[-1] == pytest.approx(g.MARGIN[1]) and np.allclose(gaps[1:] / gaps[:-1], gaps[1] / gaps[0])
    assert g.training_margins(DECLARED, 1e-3) == (), "an edge past the largest margin leaves no training gap"


def test_every_case_lies_inside_the_domain_the_tables_readable_radius_and_its_axial_reach():
    table = shipped_unit_tube()
    readable, reach = table.axes[0].readable_range[1], table.axes[3].readable_range[1]
    cases = every_case()
    assert len(cases) == 17 * 4 * (g.TRAINING_MARGINS + g.TRAINING_MARGINS - 1 + len(g.ROLLS))
    for case in cases:
        assert DECLARED.problems(case, SPACING) == [], case.key()
        assert case.offset > 0.0 and (case.tube_radius - case.gap) / case.tube_radius < readable
        assert case.design.length / case.tube_radius < reach


def test_the_held_out_sets_share_no_case_with_the_training_set():
    training = g.training_cases(DECLARED, SPACING)
    for name, cases in g.held_out_cases(DECLARED, SPACING).items():
        assert cases and g.overlap(cases, training) == [], name


def test_control_a_set_drawn_at_the_training_margins_is_found_to_overlap_in_every_case():
    training = g.training_cases(DECLARED, SPACING)
    planted = tuple(replace(case) for case in training[::5])
    assert len(g.overlap(planted, training)) == len(planted)


def test_control_a_gap_below_the_edge_or_above_the_largest_margin_lies_outside_the_domain():
    design = g.design_named("FL-9")
    for gap in (0.5 * DECLARED.edge(SPACING), 1.5 * g.MARGIN[1]):
        assert DECLARED.problems(g.Case(design, "1/4 in", gap), SPACING)
    assert DECLARED.problems(g.Case(design, "1 in", 0.1e-3), SPACING)


def test_control_while_the_edge_is_open_every_case_lies_outside_and_no_edge_is_given():
    design = g.design_named("FL-9")
    found = g.DOMAIN.problems(g.Case(design, "1/4 in", 0.1e-3), SPACING)
    assert len(found) == 1 and "open" in found[0] and "maintainer" in found[0]
    for call in (lambda: g.DOMAIN.edge(SPACING), lambda: g.training_margins(g.DOMAIN, SPACING),
                 lambda: g.training_cases(g.DOMAIN, SPACING), lambda: g.held_out_cases(g.DOMAIN, SPACING)):
        with pytest.raises(BuildRefused) as refused:
            call()
        assert refused.value.codes() == ["generator.near-wall-edge-open"]


# --- the build gate: the open edge, the domain, emptiness ---------------------------------------------------------


def parameters_of(split: str, *cases: dict, spacing: float = SPACING) -> dict:
    held: dict = {"split": split, "discretisation": g.Discretisation(spacing).as_json()}
    if split == "listed":
        held["cases"] = list(cases)
    return held


def refused_codes(builder: g.TrainingData, parameters: dict) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        assets.derive("planted", builder, parameters, {"wall:tube": TABLE}, commit="test", domain={})
    return refused.value.codes()


@pytest.mark.parametrize("split", ["training", "between", "roll", "listed"])
def test_control_the_generator_refuses_by_name_to_produce_data_while_the_edge_is_open(split: str):
    """Every set, and a listed case that is perfectly inside the margin's range, is refused before a case runs."""
    listed = {"design": "FL-9", "tube": "1/4 in", "gap": 0.1e-3}
    parameters = parameters_of(split, listed)
    assert refused_codes(g.TrainingData(), parameters) == ["generator.near-wall-edge-open"]
    with pytest.raises(BuildRefused) as refused:
        g.TrainingData().produce(parameters, {"wall:tube": TABLE})
    message = str(refused.value)
    assert "near-wall edge is open" in message and "maintainer" in message and "7i-b is blocked" in message


@pytest.mark.parametrize("case", [
    {"design": "FL-9", "tube": "1/4 in", "gap": 0.02e-3},
    {"design": "FL-9", "tube": "1/4 in", "gap": 0.05e-3},
    {"design": "FL-9", "tube": "1/4 in", "gap": 0.3e-3},
    {"design": "FL-9", "tube": "1 in", "gap": 0.1e-3},
])
def test_control_a_case_outside_the_domain_is_refused_by_name_before_it_runs(case: dict):
    """A declared edge of 0.6 spacings at 0.15 mm is 0.09 mm: the review's gap of 0.02 mm, a gap of 0.05 mm, one above
    the largest margin and a tube that is not the domain's each lie outside. The refusal comes before any case runs
    (the tube ``1 in`` has no radius to run)."""
    builder = g.TrainingData(domain=DECLARED)
    assert refused_codes(builder, parameters_of("listed", case)) == ["generator.outside-domain"]


def test_control_a_design_that_is_not_the_domains_is_refused_by_name():
    builder = g.TrainingData(domain=replace(DECLARED, designs=g.DESIGNS[1:]))
    codes = refused_codes(builder, parameters_of("listed", {"design": "FL-1", "tube": "1/4 in", "gap": 0.1e-3}))
    assert codes == ["generator.outside-domain"]


@pytest.mark.parametrize("split", ["training", "between", "roll"])
def test_control_an_empty_case_set_is_refused_by_name_not_produced_with_zero_checks(split: str):
    """At a spacing of 1 mm the declared edge lies past the largest margin: no training case, no held-out case."""
    assert refused_codes(g.TrainingData(domain=DECLARED), parameters_of(split, spacing=1e-3)) == ["generator.no-cases"]
    assert refused_codes(g.TrainingData(domain=DECLARED), parameters_of("listed")) == ["generator.no-cases"]


def test_a_case_inside_a_declared_domain_passes_the_gate_and_builds():
    parameters = parameters_of("listed", {"design": "FL-9", "tube": "1/4 in", "gap": 0.2e-3}, spacing=0.6e-3)
    content, record = assets.derive("inside", g.TrainingData(domain=COARSE_DECLARED), parameters,
                                    {"wall:tube": TABLE}, commit="test", domain=COARSE_DECLARED.as_json())
    assert content and record.checks and record.failed() == ()
    assert record.domain["near_wall"] == {"status": "declared", "ratios": {"swim_speed": 0.1}}


# --- the roll: a turned body is floored again --------------------------------------------------------------------


def _surface(design: g.Design) -> np.ndarray:
    """The design's lateral surface from the kind's own samples (``mime.scene.Screw``), a dense grid of points as
    (x, y, z) in the body's frame."""
    kind = Screw(radius=m(design.radius), length=m(design.length), wavenumber=dimensionless(design.wavenumber),
                 modulation=dimensionless(design.modulation), starts=design.starts)
    settings = {"radius": design.radius, "length": design.length, "wavenumber": design.wavenumber,
                "modulation": design.modulation}
    return np.asarray(kind.samples(settings, "lateral", (1440, 500)).points)


@pytest.fixture(scope="module")
def surfaces() -> dict[str, np.ndarray]:
    return {design.name: _surface(design) for design in g.DESIGNS}


def nearest_gap(surfaces: dict[str, np.ndarray], case: g.Case, offset: float) -> float:
    """The nearest distance of the kind's surface to the tube's wall: the body turned by the case's roll about its own
    axis, its centre ``offset`` below the tube's axis."""
    points = surfaces[case.design.name]
    y = points[:, 1] * math.cos(case.roll) - points[:, 2] * math.sin(case.roll)
    z = points[:, 1] * math.sin(case.roll) + points[:, 2] * math.cos(case.roll) - offset
    return case.tube_radius - float(np.max(np.hypot(y, z)))


def test_every_case_rests_with_its_nearest_point_at_its_declared_gap_on_the_kinds_own_samples(
        surfaces: dict[str, np.ndarray]):
    """Every roll case (204) and every fifth of the training and ``between`` cases: the nearest gap equals the declared
    margin to a micrometre, the grid's own resolution being some 0.5 um at worst. The roll set's quarter turns were the
    failures."""
    held = g.held_out_cases(DECLARED, SPACING)
    cases = [*held["roll"], *[*g.training_cases(DECLARED, SPACING), *held["between"]][::5]]
    worst = 0.0
    for case in cases:
        worst = max(worst, abs(nearest_gap(surfaces, case, case.offset) - case.gap))
    print(f"\nthe largest difference between a case's nearest gap and its declared gap: {worst:.3e} m")
    assert worst < 1e-6


def test_the_nearest_gap_the_generator_states_is_the_kinds_too(surfaces: dict[str, np.ndarray]):
    for case in g.held_out_cases(DECLARED, SPACING)["roll"][::7]:
        assert case.nearest_gap == pytest.approx(case.gap, abs=1e-12)
        assert nearest_gap(surfaces, case, case.offset) == pytest.approx(case.nearest_gap, abs=1e-6)


def test_at_roll_zero_the_floored_pose_is_the_outer_radius_one():
    for design in g.DESIGNS:
        for tube in g.TUBES:
            case = g.Case(design, tube, 0.1e-3)
            assert case.offset == pytest.approx(case.tube_radius - design.outer_radius - case.gap, abs=1e-11)


def test_control_the_pose_floored_at_the_unturned_roll_leaves_a_turned_body_off_the_floor(
        surfaces: dict[str, np.ndarray]):
    """The review's case: FL-1 (0.19 turns long) at a quarter turn about its axis, floored by its outer radius, is
    0.288 mm from the wall against a nominal 0.137 mm (and 0.172 at three eighths), measured on the kind's samples. The
    check that every case rests at its gap fails it on every run."""
    design = g.design_named("FL-1")
    for roll, expected in ((math.pi / 2.0, 0.288), (3.0 * math.pi / 4.0, 0.172)):
        case = g.Case(design, "1/4 in", 0.137e-3, roll)
        nominal = case.tube_radius - design.outer_radius - case.gap
        off = nearest_gap(surfaces, case, nominal)
        assert off == pytest.approx(expected * 1e-3, abs=1e-6), (roll, off)
        assert abs(off - case.gap) > 3e-5, "far above the check's micrometre"
        assert case.offset > nominal, "the body goes further towards the floor"
        assert abs(nearest_gap(surfaces, case, case.offset) - case.gap) < 1e-6


@pytest.fixture(scope="module")
def evaluator() -> g.MRSEvaluator:
    return g.MRSEvaluator(g.design_named("FL-9"), "1/4 in", COARSE)


def test_a_case_through_drag_mrs_in_the_tube_wall_kind_passes_its_checks(evaluator: g.MRSEvaluator):
    sample = evaluator(g.Case(evaluator.design, "1/4 in", 0.2e-3))
    print(f"\n{sample.samples} samples; checks {[(c.name, c.measured) for c in sample.checks()]}")
    assert sample.accepted and all(check.passed for check in sample.checks())
    assert sample.swim()[2] > 0.0, "FL-9 spun about +z swims along +z, the sign of the 0.2 line's swim speeds for it"


def test_a_half_turn_about_the_bodys_own_axis_leaves_the_resistance_unchanged_to_the_samplings_asymmetry(
        evaluator: g.MRSEvaluator):
    """Turning the body half a turn about its own axis maps a two-start screw's section onto itself, so the body, the
    pose's frame and the resistance are the same; its samples are not (a quasi-uniform sampling is not the section's
    symmetry, as the quadrature it was chosen from was), so the two resistances agree to the sampling's asymmetry and
    no closer: a check of the pose the evaluator builds from a case's roll, which is applied (they differ) and about
    the body's own axis (they agree to ``HALF_TURN``, where a roll about another axis moves the body at the wall)."""
    case = g.Case(evaluator.design, "1/4 in", 0.2e-3)
    turned = evaluator(replace(case, roll=math.pi)).resistance
    held = evaluator(case).resistance
    apart = float(np.max(np.abs(turned - held)) / np.max(np.abs(held)))
    print(f"\nthe half turn's resistance apart from the unturned's by {apart:.3e} of its largest entry")
    assert 0.0 < apart <= HALF_TURN


def test_on_the_kept_grid_screw_a_half_turn_leaves_the_resistance_unchanged_to_rounding(monkeypatch):
    """The exact check ``HALF_TURN`` replaced, kept on the screw 0.2.0, whose quadrature the half turn maps onto
    itself: under quasi-uniform sampling any roll moves the resistance, so the band passes a roll about another axis
    or by another angle, which this catches."""
    monkeypatch.setattr(mime.scene, "Screw", KEPT[("screw", "0.2.0")][0])
    design = g.design_named("FL-9")
    shell = g.experiment(design, "1/4 in", COARSE).document()["things"]["screw"]["parts"]["shell"]["solid"]
    assert (shell["name"], shell["version"]) == ("screw", "0.2.0")
    evaluator = g.MRSEvaluator(design, "1/4 in", COARSE)
    case = g.Case(design, "1/4 in", 0.2e-3)
    turned, held = evaluator(replace(case, roll=math.pi)).resistance, evaluator(case).resistance
    assert np.max(np.abs(turned - held)) <= 1e-6 * np.max(np.abs(held))


def test_the_derived_data_record_the_shipped_tables_hash_and_are_not_stale():
    parameters = {"split": "listed", "discretisation": COARSE.as_json(),
                  "cases": [{"design": "FL-9", "tube": "1/4 in", "gap": 0.2e-3}]}
    content, record = assets.derive("tiny", g.TrainingData(domain=COARSE_DECLARED), parameters, {"wall:tube": TABLE},
                                    commit="test", domain=COARSE_DECLARED.as_json())
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
    # A domain wide enough to admit the case, so that the table's own refusal is what is tested.
    wide = replace(COARSE_DECLARED, margin=(-1e-3, g.MARGIN[1]), near_wall=g.DeclaredEdge({"swim_speed": -1.0}))
    with pytest.raises(BuildRefused) as refused:
        assets.derive("too_near", g.TrainingData(domain=wide), parameters, {"wall:tube": TABLE}, commit="test",
                      domain={})
    assert refused.value.codes() == ["asset.derivation-check"] and "wall_accepted" in str(refused.value)


def test_the_generator_refuses_float32():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            g.MRSEvaluator(g.design_named("FL-9"), "1/4 in", COARSE)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_accuracy_is_judged_per_use_on_the_resistance_and_the_swim_speed(evaluator: g.MRSEvaluator):
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
