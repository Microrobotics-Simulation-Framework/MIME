"""The plot description and the result protocol it is built from: a result made by hand from plain arrays, units and a
time base (a stand-in for measured data) is described as a simulated run is; the flags as intervals exactly where the
result's track raised them; the validity limits as bands; an uncertainty as a band about its curve. Each control (a flag
shifted by one sample, a band shifted by one sample) must fail on every run, which the tests below count."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rigs_plot import run_of

from mime.build._refusals import ArgumentRefused
from mime.results import Measured, RunData, figure
from mime.results._plot import describe_run, runs_of
from mime.units import Hz, m_per_s, s

SPEEDS = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 2.0, 1.0, 3.0, 3.2, 0.5, 0.2]
RAISED = [False, False, False, True, True, True, False, False, True, True, False, False]
DT = 0.004


def raised_runs(raised):
    """The maximal runs of True, by the test's own loop, as (first, last) step numbers from 1."""
    found, start = [], None
    for i, flag in enumerate([*raised, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            found.append((start + 1, i))
            start = None
    return found


def test_a_flags_intervals_are_exactly_where_its_track_raised_it():
    description = describe_run(RunData(run_of(SPEEDS, DT, raised=RAISED)))
    spans = [(span.first, span.last) for span in description.panels[0].spans if span.flag == "law:p.fired"]
    assert spans == raised_runs(RAISED) == [(4, 6), (9, 10)]
    seconds = [(span.start, span.end) for span in description.panels[0].spans]
    assert seconds == [(4 * DT, 6 * DT), (9 * DT, 10 * DT)]
    assert runs_of(np.asarray(RAISED)) == [(3, 5), (8, 9)]


def test_control_a_flag_shifted_by_one_sample_differs_from_the_track():
    detected = 0
    cases = [RAISED, [True, *[False] * 11], [False] * 4 + [True] * 4 + [False] * 4]
    for raised in cases:
        spans = [(span.first, span.last) for span in describe_run(RunData(run_of(SPEEDS, DT, raised=raised))).panels[0].spans]
        shifted = [(a + 1, b + 1) for a, b in spans]
        detected += shifted != raised_runs(raised)
    assert detected == len(cases)


def test_a_numeric_validity_limit_on_the_plotted_field_is_a_band_and_prose_is_not():
    description = describe_run(RunData(run_of(SPEEDS, DT, limited=True)))
    bands = description.panels[0].bands
    assert [(b.label, b.low, b.high) for b in bands] == [("probe:p.speed_limit limit", -2.5, 2.5)]
    assert bands[0].source == "the speed left its validity range"
    assert not describe_run(RunData(run_of(SPEEDS, DT))).panels[0].bands
    spans = [(s_.flag, s_.first, s_.last) for s_ in description.panels[0].spans if s_.flag == "probe:p.speed_limit"]
    assert spans == [("probe:p.speed_limit", 5, 6), ("probe:p.speed_limit", 9, 10)]  # the steps where |speed| > 2.5


def test_a_result_built_by_hand_plots_as_a_simulated_one_does():
    simulated = describe_run(RunData(run_of(SPEEDS, DT, raised=RAISED)))
    steps = np.arange(1, len(SPEEDS) + 1)
    by_hand = Measured("synthetic", s(steps * DT), {"probe:p.speed": m_per_s(SPEEDS)}, flags={"law:p.fired": RAISED},
                       meanings={"law:p.fired": "the law left its range"})
    description = describe_run(by_hand)
    a, b = simulated.panels[0], description.panels[0]
    assert a.y_label == b.y_label == "value [m/s]" and a.x_label == b.x_label == "time [s]"
    assert np.array_equal(a.traces[0].x, b.traces[0].x) and np.array_equal(a.traces[0].y, b.traces[0].y)
    assert [(x.start, x.end) for x in a.spans] == [(x.start, x.end) for x in b.spans]
    recipe = figure(by_hand, "probe:p.speed", title="a measured trajectory")
    assert recipe.plot().axes[0].get_ylabel() == "value [m/s]"
    assert recipe.plot_interactive().layout.title.text == "a measured trajectory"
    assert json.loads(recipe.description.to_json())["panels"][0]["traces"][0]["label"] == "probe:p.speed"


def test_an_uncertainty_is_a_band_where_the_data_says_and_its_kind_is_marked():
    amount = np.linspace(0.1, 0.5, len(SPEEDS))
    data = Measured("synthetic", s(np.arange(1, len(SPEEDS) + 1) * DT), {"x": m_per_s(SPEEDS)},
                    uncertainty={"x": ("std", m_per_s(amount))})
    description = describe_run(data)
    trace = description.panels[0].traces[0]
    assert trace.uncertainty == "std" and np.array_equal(trace.amount, amount)
    assert any("standard deviation" in note and "estimate" in note for note in description.notes)
    assert any("estimate" in note for note in description.notes)
    bound = describe_run(Measured("b", s(np.arange(1, 4) * DT), {"x": m_per_s([1, 2, 3])},
                                  uncertainty={"x": ("bound", m_per_s([0.1, 0.1, 0.1]))}))
    assert any("stated bound" in note for note in bound.notes)
    figure_ = figure(data, "x").plot()
    assert len(figure_.axes[0].collections) >= 1  # the filled band
    plotted = figure(data, "x").plot_interactive()
    assert any(trace_.fill == "tonexty" for trace_ in plotted.data)


def test_control_an_uncertainty_band_shifted_by_one_sample_differs_from_the_data():
    detected = 0
    for seed in range(5):
        amount = np.random.default_rng(seed).uniform(0.1, 0.9, len(SPEEDS))
        data = Measured("m", s(np.arange(1, len(SPEEDS) + 1) * DT), {"x": m_per_s(SPEEDS)},
                        uncertainty={"x": ("std", m_per_s(amount))})
        trace = describe_run(data).panels[0].traces[0]
        shifted = np.roll(amount, 1)
        detected += not np.array_equal(trace.amount, shifted)
    assert detected == 5


def test_a_result_with_no_series_and_a_time_base_that_is_not_one_are_refused_by_name():
    with pytest.raises(ArgumentRefused) as nothing:
        describe_run(Measured("m", s([0.0, 1.0]), {}))
    assert nothing.value.refusals[0].code == "plot.nothing"
    with pytest.raises(ArgumentRefused) as base:
        Measured("m", s([0.0, 2.0, 1.0]), {"x": m_per_s([1, 2, 3])})
    assert base.value.refusals[0].code == "plot.time-base"
    with pytest.raises(ArgumentRefused) as base2:
        Measured("m", Hz(3.0), {"x": m_per_s([1, 2, 3])})  # type: ignore[arg-type]
    assert base2.value.refusals[0].code == "plot.time-base"
