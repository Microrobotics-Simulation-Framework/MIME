"""The plot description and the result protocol it is built from: a result made by hand from plain arrays, units and a
time base (a stand-in for measured data) is described as a simulated run is; the flags as intervals exactly where the
result's track raised them; the validity limits as bands; an uncertainty as a band about its curve. Each control (a flag
shifted by one sample, a band shifted by one sample) must fail on every run, which the tests below count."""

from __future__ import annotations

import json

import numpy as np
import pytest
from rigs_plot import run_of, study_of

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


# ---------------------------------------------------------------------------------------------------------------
# The caption: what a figure bounds and estimates is drawn on it, not left in its description (the refuting review's M5)


def drawn_text(figure) -> str:
    from matplotlib.text import Text

    return " ".join(t.get_text() for t in figure.findobj(Text)).replace("\n", " ")


def plotly_text(figure) -> str:
    return " ".join(str(a.text) for a in figure.layout.annotations).replace("<br>", " ")


def test_the_drawn_figure_carries_the_bounds_against_estimates_marking_and_the_interpolations_name():
    members = study_of([run_of(SPEEDS, DT), run_of([x + 0.25 for x in SPEEDS], DT)], [10.0, 20.0])
    off_grid = Measured("m", s([0.0015, 0.0045, 0.0075]), {"probe:p.speed": m_per_s([1.0, 2.0, 3.0])})
    comparison = members.compare(off_grid, tolerance=m_per_s(1.0))
    for text in (drawn_text(comparison.plot()), plotly_text(comparison.plot_interactive())):
        assert "bounds the model's error from above only" in text and "linearly interpolated onto the reference's times" in text
    run_text = drawn_text(figure(run_of(SPEEDS, DT), "probe:p.speed").plot())
    assert "what the model computes" in run_text and "estimate" in run_text
    assert "what the model computes" in plotly_text(figure(run_of(SPEEDS, DT), "probe:p.speed").plot_interactive())
    # the page carries it too, as the figure's own annotation
    assert "what the model computes" in figure(run_of(SPEEDS, DT), "probe:p.speed").page().replace("\\u003cbr>", " ")


def test_a_measured_result_is_captioned_as_measured_and_never_as_what_the_model_computes():
    data = Measured("video", s(np.arange(1, 6) * DT), {"x": m_per_s([1, 2, 3, 4, 5])}, uncertainty={"x": ("std", m_per_s([0.1] * 5))})
    recipe = figure(data, "x")
    assert not any("what the model computes" in note for note in recipe.description.notes)
    assert any("measured" in note for note in recipe.description.notes)
    for text in (drawn_text(recipe.plot()), plotly_text(recipe.plot_interactive())):
        assert "measured" in text and "what the model computes" not in text and "standard deviation" in text
    simulated = describe_run(RunData(run_of(SPEEDS, DT)))
    assert any("what the model computes" in note for note in simulated.notes)       # control: a simulated one is
    members = study_of([run_of(SPEEDS, DT)], [10.0])
    against_measurement = members.compare(Measured("video", s(np.arange(1, 5) * DT), {"probe:p.speed": m_per_s(SPEEDS[:4])}),
                                          tolerance=m_per_s(1.0))
    assert "measurement" in against_measurement.description.notes[0] and "another run of the model" not in against_measurement.description.notes[0]


def test_a_measured_results_inputs_are_checked_where_it_is_built():
    t = s(np.arange(1, 6) * DT)
    series = {"x": m_per_s([1, 2, 3, 4, 5])}
    for wrong, code in (({"uncertainty": {"x": ("std", m_per_s([0.1, 0.1]))}}, "plot.uncertainty"),     # the wrong length
                        ({"uncertainty": {"x": ("std", m_per_s([-0.1] * 5))}}, "plot.uncertainty"),     # a negative spread
                        ({"flags": {"f": [True, False]}}, "plot.flag"),                                  # shorter than the time base
                        ({"flags": {"f": [True] * 9}}, "plot.flag"),                                     # and longer
                        ({"limits": [("nothing", -1.0, 1.0, "a limit on no series")]}, "plot.limit")):
        with pytest.raises(ArgumentRefused) as error:
            Measured("m", t, series, **wrong)
        assert error.value.refusals[0].code == code, wrong
    Measured("m", t, series, uncertainty={"x": ("std", m_per_s([0.1] * 5))}, flags={"f": [True] * 5},
             limits=[("x", -1.0, 1.0, "a limit")])    # control: the same inputs, right
