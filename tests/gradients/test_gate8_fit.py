"""The first fit (``Study.fit``) under the fit-loop rule, and gate 8: validity judged at each iterate.

- **The consumer**: the gently pulled sphere's radius fitted from 1.0 mm to a pose target made at 1.08 mm, inside its
  declared range: it ends on the target (to 1.5e-12), where the spike's descent by rebuilding ended 0.79 % short (each
  rebuild re-chose its force points). Its control: the same fit without the range is refused at its first iterate that
  needs another program (``fit.value-keyed``), never ending short, and the error holds the iterates taken. A target
  inside the range, so that the pass is the descent's and not the box's clip (a target at the range's top passes a fit
  whose derivative is wrong by a factor); a target past the range ends at the box, naming it.
- **Gate 8**: the same sphere turned by a torque that puts the rung's declared Reynolds number (0.1) at about 1.05 mm,
  fitted from 1.15 mm towards a target made at 1.0 mm, past the line: an iterate past it is judged there and not
  taken, the line's margin limits the steps, and the fit ends on it, naming it **at the line's own place**: a fresh
  build at the radius it ended at reads the Reynolds number within 1e-4 of the line's limit (:func:`turning_number`,
  independent of the fit's own judgement). **Control**: the same loop through MADDENING's ``sysid.fit`` on the built
  graph (Adam, the loss through the graph's parameters) ends past the line, and its result says nothing of it
  (``MIME_PLANT_FIT=sysid`` runs it in the gate's place, for the kit's rate). A line that cannot be judged (no density
  declared) is not taken as inside: the start is refused, and an iterate whose line cannot be judged ends the fit
  ``unjudged``.
- **The Fisher information at the start**: its scale is the declared noise's (its largest eigenvalue is the squared
  scaled derivative the fit's own steps use), and every axis the residual cannot see is named before the first step.
- **The default stop follows the dtype**: float64's is 1e-9 of the values; float32's is ten times its machine epsilon
  (1.2e-6), where the consumer's fit ends ``converged`` (at 1e-9 it spun to the iteration limit: its steps fall below
  the arithmetic's resolution), and a stop asked for below the floor ends ``stagnated`` once the loss fails to fall in
  three iterates in a row.
- **The fit loop's clauses**, each broken in turn by ``MIME_PLANT_FIT_LOOP`` (``gradient_plants.fit_plant``): the whole
  file must fail on each (the kit's rate)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_plants import FIT_LOOP, fit_plant, from_environment
from gradient_rigs import RADIUS, STEPS, TURNED, gentle, pose_residual, pose_target
from maddening import sysid

import mime.study._fit as fit_module
from mime.build import ArgumentRefused, BuildRefused
from mime.build._watch import WatchPlan
from mime.results._watch import read_watch
from mime.study import Study
from mime.units import kg, mm, ms

NOISE = 1e-4
LINE = "reynolds:drag:ball"
INTERIOR = 1.08
#: How near the line's limit a fit ending on it reads the Reynolds number of a fresh build at its radius (the fit's own
#: reach of a line, ``LINE_REACHED``, is the same share).
ON_THE_LINE = 1e-4


@pytest.fixture(autouse=True, scope="module")
def planted():
    """``MIME_PLANT_FIT_LOOP`` breaks a clause of the loop for the whole module (the gate kit's rate of each)."""
    with pytest.MonkeyPatch.context() as patch:
        fit_plant(patch, from_environment(FIT_LOOP))
        yield


@pytest.fixture(scope="module")
def interior():
    """The pose target at 1.08 mm, made over the range as the fit's iterates are."""
    return pose_target(INTERIOR)


@pytest.fixture(scope="module")
def consumer(interior):
    """The consumer's fit; ``MIME_PLANT_FIT=unranged`` takes the range away (the gate kit's rate of its control)."""
    ranged = None if os.environ.get("MIME_PLANT_FIT") == "unranged" else (1.0, 1.15)
    try:
        return Study(gentle(1.0, ranged=ranged)).fit(pose_residual(interior), start={RADIUS: mm(1.0)}, noise=NOISE,
                                                     duration=ms(STEPS))
    except BuildRefused as refused:  # the plant's refusal, which the test then fails on, never errors on
        return refused


def test_the_fit_ends_on_a_target_inside_the_range(consumer):
    assert not isinstance(consumer, BuildRefused), f"the fit was refused: {consumer}"
    assert consumer.ended == "converged"
    assert abs(float(consumer.values[RADIUS]) / (INTERIOR * 1e-3) - 1.0) < 1e-6
    assert all(line.judged for iterate in consumer.iterates for line in iterate.lines)
    assert int(consumer.fim.rank) == 1 and consumer.insensitive == ()


def float32_fit(**given):
    """The consumer's fit in float32 (the gradient gates run in float64; this one sets its own)."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        return Study(gentle(1.0)).fit(pose_residual(pose_target(INTERIOR)), start={RADIUS: mm(1.0)}, noise=NOISE,
                                      duration=ms(STEPS), **given)
    finally:
        jax.config.update("jax_enable_x64", held)


def test_the_default_stop_follows_the_dtype_and_float32_converges_at_it_where_float64_keeps_its_own(consumer):
    """float64: 1e-9, unchanged (the consumer's reason says so). float32: ten times its epsilon, 1.2e-6, derived from the
    arithmetic's floor; the fit ends ``converged`` within 2e-3 of the target (measured 4.3e-4: the float32 run's own
    rounding), where at 1e-9 it ended ``iterations`` after twenty."""
    assert fit_module.default_tolerance(True) == 1e-9
    assert fit_module.default_tolerance(False) == pytest.approx(10 * float(np.finfo(np.float32).eps))
    assert "1e-09" in consumer.why
    result = float32_fit()
    assert result.ended == "converged", (result.ended, result.why, len(result.iterates))
    assert abs(float(result.values[RADIUS]) / (INTERIOR * 1e-3) - 1.0) < 2e-3


def test_a_stop_below_float32s_floor_ends_stagnated_not_iterations_after_spinning():
    result = float32_fit(tolerance=1e-12, iterations=40)
    assert result.ended == "stagnated" and "did not fall" in result.why
    assert len(result.iterates) < 40 and not result.iterates[-1].taken


def test_the_information_is_over_the_declared_noise(consumer):
    """The Fisher information of a one-axis fit is the square of the scaled derivative the fit's own steps use, times the
    axis's value squared (MADDENING scales a column by its parameter): its largest eigenvalue, against that. Taken with a
    unit noise it would be 1e-8 of it (``fim_noise``)."""
    first = consumer.iterates[0]
    expected = float(np.sum((np.asarray(first.values[RADIUS]) * first.jacobian[RADIUS]) ** 2))
    assert float(consumer.fim.eigvals[-1]) == pytest.approx(expected, rel=1e-4)


def test_control_without_the_range_the_fit_is_refused_where_an_iterate_needs_another_program(interior):
    with pytest.raises(BuildRefused) as raised:
        Study(gentle(1.0, ranged=None, name="unranged")).fit(pose_residual(interior), start={RADIUS: mm(1.0)},
                                                            noise=NOISE, duration=ms(STEPS))
    (refusal,) = raised.value.refusals
    assert refusal.code == "fit.value-keyed" and RADIUS in refusal.message and "judged after 1 taken" in refusal.message
    partial = raised.value.partial
    assert partial.ended == "refused" and len(partial.taken) == 1 and float(partial.values[RADIUS]) == 1.0e-3


def test_a_target_past_the_range_ends_at_the_box_naming_it():
    """The range's top is a line the fit reaches: it ends ``at a line``, not ``converged`` on the clip."""
    target = pose_target(1.3, ranged=None)
    result = Study(gentle(1.0)).fit(pose_residual(target), start={RADIUS: mm(1.0)}, noise=NOISE, duration=ms(STEPS))
    assert result.ended == "at a line" and result.why == f"box:{RADIUS}"
    assert float(result.values[RADIUS]) == pytest.approx(1.15e-3, rel=1e-12)


def sysid_radius(target) -> float:
    """MADDENING's sysid.fit on the built graph, the loss half the scaled residual's squared norm through the graph's
    parameters, the radius alone trained: Adam from 1.15 mm, 150 updates of about 2e-6 m."""
    built = gentle(1.15, pull=TURNED, name="sysid").build()
    graph = built.graph
    residual = pose_residual(target)

    def loss(params):
        graph.reset_state()
        scaled = residual(graph.run_scan(STEPS, params=params)) / NOISE
        return 0.5 * jnp.sum(scaled * scaled)

    mask = jax.tree.map(lambda _: False, graph.params)
    mask["nodes"]["drag:ball"]["shape_radius"] = True
    found = sysid.fit(graph, loss, mask=mask, n_iter=150, lr=2e-6)
    graph.reset_state()
    return float(found.params["nodes"]["drag:ball"]["shape_radius"])


def turning_number(radius_mm: float) -> tuple[float, float]:
    """The Reynolds number a fresh build at ``radius_mm`` reads at its run's end, and the rung's declared bound on it: the
    fit's end judged without the fit's own watch."""
    built = gentle(radius_mm, pull=TURNED, name="watched").build()
    final = {node: {name: np.asarray(value) for name, value in fields.items()}
             for node, fields in built.run(ms(STEPS)).final.items()}
    (plan,) = [held for held in built.plans.watches if isinstance(held, WatchPlan)]
    read = read_watch(plan, final, None)
    assert read.reynolds is not None and read.upper is not None
    return read.reynolds, read.upper


@pytest.fixture(scope="module")
def across():
    target = pose_target(1.0, TURNED)
    if os.environ.get("MIME_PLANT_FIT") == "sysid":
        return None, sysid_radius(target)
    return Study(gentle(1.15, pull=TURNED)).fit(pose_residual(target), start={RADIUS: mm(1.15)}, noise=NOISE,
                                                duration=ms(STEPS)), None


def test_an_iterate_past_a_line_is_judged_there_and_the_fit_ends_on_the_line(across):
    result, control = across
    assert control is None, f"the loop ran through sysid.fit and ended at {control} m, judging no line"
    assert result.ended == "at a line" and result.why == LINE
    last = result.taken[-1]
    (held,) = [line for line in last.lines if line.name == LINE]
    assert 0.0 <= held.margin <= 1e-5
    assert any(LINE in [line.name for line in iterate.past] and not iterate.taken for iterate in result.iterates)
    assert 1.0e-3 < float(result.values[RADIUS]) < 1.15e-3


def test_the_fit_ends_where_a_fresh_build_reads_the_line(across):
    """The independent oracle: the Reynolds number of a fresh build at the radius the fit ended at is within 1e-4 of the
    limit, inside it. A fit that judged each iterate by the start's plan (``stale_watch``) ends 6 % short of
    the line's radius, with every line of its own result in order."""
    result, _ = across
    reynolds, limit = turning_number(float(result.values[RADIUS]) * 1e3)
    assert 0.0 <= (limit - reynolds) / limit <= ON_THE_LINE, (reynolds, limit)


def test_control_the_same_loop_through_sysid_fit_ends_past_the_line_and_says_nothing():
    radius = sysid_radius(pose_target(1.0, TURNED))
    reynolds, limit = turning_number(radius * 1e3)
    assert radius < 1.04e-3, "it went towards the target, through the line"
    assert reynolds > limit, "past the line, which its result does not mention"


def test_a_line_that_cannot_be_judged_is_not_taken_as_inside():
    """No density declared: the Reynolds line reads "not judged" at every iterate. The start is refused, naming the
    line and why (the fit used to end ``converged`` at 1.0091 mm, where a fresh build reads 0.104, past the line)."""
    with pytest.raises(BuildRefused) as raised:
        Study(gentle(1.15, pull=TURNED, density=False, name="nodensity")).fit(
            lambda final: final["ball"]["pose"][:3] * 1e6, start={RADIUS: mm(1.15)}, noise=NOISE, duration=ms(STEPS))
    (refusal,) = raised.value.refusals
    assert refusal.code == "fit.line-unjudged" and LINE in refusal.message and "water.density" in refusal.message


def test_an_iterate_whose_line_cannot_be_judged_ends_the_fit_unjudged(interior, monkeypatch):
    """A line judged at the start and not at the next iterate (planted: a watch whose density goes) ends the fit
    ``unjudged``, naming it, never ``converged`` past it."""
    lines = fit_module._Fit.run_lines  # noqa: SLF001
    calls = []

    def losing(self, tally, final, watches):
        calls.append(1)
        found = lines(self, tally, final, watches)
        if len(calls) > 1:
            found.append(fit_module.LineMargin(LINE, "run", float("nan"), 0.1, float("nan"), "not computed: planted"))
        return found

    monkeypatch.setattr(fit_module._Fit, "run_lines", losing)  # noqa: SLF001
    result = Study(gentle(1.0)).fit(pose_residual(interior), start={RADIUS: mm(1.0)}, noise=NOISE, duration=ms(STEPS))
    assert result.ended == "unjudged" and LINE in result.why and "planted" in result.why
    assert len(result.taken) == 1, "the start alone: an iterate with a line not judged is never taken"


def test_every_axis_the_residual_cannot_see_is_named_before_the_first_step():
    """Two axes, neither seen: the information has rank 0 of 2, and both are named (the weakest direction names one)."""
    result = Study(gentle(1.05)).fit(lambda final: jnp.zeros(3) * final["ball"]["pose"][0] + 1.0,
                                     start={RADIUS: mm(1.05), "ball.mass": kg(2e-8)}, noise=NOISE, duration=ms(STEPS))
    assert int(result.fim.rank) == 0 and set(result.insensitive) == {RADIUS, "ball.mass"}


def test_an_axis_at_exactly_zero_has_no_relative_scale_and_is_not_called_insensitive():
    """MADDENING's ``scale="relative"`` multiplies a column by its parameter: a start of exactly 0.0 vanishes whatever the
    data say, and the report names it ``zero_scaled``. It is not the residual that cannot see it."""
    residual = lambda given: jnp.stack([2.0 * given["seen"], 3.0 * given["zero"]])  # noqa: E731
    report = fit_module.fisher_information(residual, {"seen": jnp.asarray(1.0), "zero": jnp.asarray(0.0)}, np.ones(2))
    assert len(report.zero_scaled) == 1 and "zero" in report.zero_scaled[0]
    assert fit_module._insensitive(report, ("seen", "zero")) == ()  # noqa: SLF001
    blind = fit_module.fisher_information(lambda given: jnp.ones(2) + 0.0 * given["seen"] + 0.0 * given["other"],
                                          {"seen": jnp.asarray(1.0), "other": jnp.asarray(2.0)}, np.ones(2))
    assert set(fit_module._insensitive(blind, ("seen", "other"))) == {"seen", "other"}  # noqa: SLF001


def test_a_fit_with_no_start_or_a_noise_that_is_not_given_or_positive_is_refused():
    with pytest.raises(ArgumentRefused) as raised:
        Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={}, noise=NOISE, duration=ms(STEPS))
    assert [r.code for r in raised.value.refusals] == ["fit.start"]
    for noise in (-1.0, None):
        with pytest.raises(ArgumentRefused) as raised:
            Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={RADIUS: mm(1.05)}, noise=noise,
                                    duration=ms(STEPS))
        assert [r.code for r in raised.value.refusals] == ["fit.noise"], noise


def test_a_fit_starting_outside_its_range_is_refused():
    with pytest.raises(ArgumentRefused) as raised:
        Study(gentle(1.05)).fit(lambda final: final["ball"]["pose"], start={RADIUS: mm(1.2)}, noise=NOISE,
                               duration=ms(STEPS))
    assert [r.code for r in raised.value.refusals] == ["fit.start"]


def box_of(monkeypatch, *, factor, start, at, bounds):
    """The box of an axis writing one bounded leaf through ``factor`` (leaf = at + factor (theta - start))."""
    monkeypatch.setattr(fit_module, "parameter_specs", lambda graph: {"n": {"l": SimpleNamespace(bounds=bounds)}})
    write = SimpleNamespace(leaves=(("n", "l"),), factors={"leaf:n.l": np.asarray(factor, dtype=float)})
    point = SimpleNamespace(tree={"n": {"l": np.asarray(at, dtype=float)}})
    return fit_module._box(SimpleNamespace(path="a"), write, point, np.asarray(start, dtype=float), [],  # noqa: SLF001
                           SimpleNamespace(graph=None), None)


def test_a_leaf_bound_is_read_back_through_the_factor_with_its_sign(monkeypatch):
    """leaf = 2 + f (theta - 1), bounded to (0, 10): theta in (0, 5) for f = 2; the sides turn round for f = -2, (-3, 2);
    a scalar axis writing a leaf of two numbers is inside both of them, (-1, 8); a zero factor binds nothing."""
    low, high = box_of(monkeypatch, factor=[2.0], start=1.0, at=[2.0], bounds=(0.0, 10.0))
    assert (float(low), float(high)) == (0.0, 5.0)
    low, high = box_of(monkeypatch, factor=[-2.0], start=1.0, at=[2.0], bounds=(0.0, 10.0))
    assert (float(low), float(high)) == (-3.0, 2.0)
    low, high = box_of(monkeypatch, factor=[1.0, 1.0], start=1.0, at=[2.0, 3.0], bounds=(0.0, 10.0))
    assert (float(low), float(high)) == (-1.0, 8.0)
    low, high = box_of(monkeypatch, factor=[0.0], start=1.0, at=[2.0], bounds=(0.0, 10.0))
    assert (float(low), float(high)) == (-np.inf, np.inf)


def test_a_leaf_written_through_a_factor_of_another_size_than_the_axis_is_refused_not_skipped(monkeypatch):
    with pytest.raises(ArgumentRefused) as raised:
        box_of(monkeypatch, factor=[1.0, 1.0, 1.0], start=[1.0, 2.0], at=[1.0, 1.0, 1.0], bounds=(0.0, 1.0))
    assert [r.code for r in raised.value.refusals] == ["fit.box"]
