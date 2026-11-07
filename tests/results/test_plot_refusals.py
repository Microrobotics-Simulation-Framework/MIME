"""The refusals of plots, figures, comparisons and the overlay, each by its code and with what to change."""

from __future__ import annotations

import numpy as np
import pytest
from rigs_plot import run_of, study_of

from mime.build._refusals import ArgumentRefused
from mime.results import Comparison, FigureOptions, Measured, Uncertainty
from mime.results._stage import StageRecord
from mime.units import VELOCITY, Quantity, m_per_s, s
from mime.viz._compare import compare_layer

SPEEDS = [1.0, 2.0, 3.0, 4.0]


def study():
    return study_of([run_of(SPEEDS, 0.001), run_of(SPEEDS, 0.001), run_of(SPEEDS, 0.001)], [10.0, 20.0, 30.0])


def code_of(call) -> str:
    with pytest.raises(ArgumentRefused) as error:
        call()
    return error.value.refusals[0].code


def test_a_plot_names_what_it_cannot_draw():
    result = study()
    assert code_of(lambda: result.plot("probe:p.mean", against="no.such.axis")) == "plot.coordinate"
    assert code_of(lambda: result.plot("probe:p.mean", kind="bogus")) == "plot.kind"
    assert code_of(lambda: result.plot("probe:p.nothing")) == "plot.output"
    assert code_of(lambda: result.plot("probe:p.speed", kind="curve")) == "plot.shape"
    assert code_of(lambda: result.plot("probe:p.speed", "probe:p.mean")) == "plot.output"
    assert code_of(lambda: study_of([run_of(SPEEDS, 0.001)], [1.0]).plot(kind="curve")) in {"plot.kind", "plot.output"}
    assert code_of(lambda: run_of(SPEEDS, 0.001).plot("probe:p.absent")) == "plot.output"
    assert code_of(lambda: run_of(SPEEDS, 0.001).plot(kind="overlay")) == "plot.kind"


def test_an_option_that_is_not_one_and_an_uncertainty_that_is_not_a_kind_are_refused():
    assert code_of(lambda: FigureOptions(legend="middle")) == "plot.option"
    assert code_of(lambda: FigureOptions(size=(0.0, 3.0))) == "plot.option"
    assert code_of(lambda: FigureOptions.from_dict({"colour": "red"})) == "plot.option"
    assert code_of(lambda: Uncertainty("variance", np.zeros(2))) == "plot.uncertainty"


def test_plotly_js_that_is_not_the_pinned_file_is_not_inlined_held_to_a_hash(monkeypatch):
    import mime.results._figure as figure

    monkeypatch.setattr(figure, "PLOTLY_JS_SHA256", "0" * 64)
    assert code_of(lambda: figure.library_tag(False)) == "plot.library"
    assert "jsdelivr" in figure.PLOTLY_CDN and figure.library_tag(True)  # the CDN tag needs no installed bundle


def test_a_comparison_names_what_it_cannot_compare():
    result = study()
    assert code_of(lambda: result.compare(7)) == "compare.reference"
    assert code_of(lambda: study_of([run_of(SPEEDS, 0.001)], [1.0]).compare(0)) == "compare.nothing"
    assert code_of(lambda: result.compare(0, window=(s(1.0), s(2.0)))) == "compare.window"
    two_wide = Measured("m", s(np.arange(1, 5) * 0.001), {"probe:p.speed": Quantity.of(np.ones((4, 2)), VELOCITY)})
    assert code_of(lambda: result.compare(two_wide)) == "compare.shape"
    comparison = result.compare(0, tolerance=m_per_s(1.0))
    assert code_of(lambda: comparison.of(9)) == "compare.row"
    assert code_of(lambda: comparison.of(1)) == "compare.row"   # two rows (the speed and its mean): one is asked for
    assert code_of(lambda: Comparison("x", (), "rule").description) == "compare.no-series"


def test_the_overlay_names_the_stage_it_cannot_ghost():
    def record(prim: str, times) -> StageRecord:
        entry = {"prim": prim, "node": "umr", "field": "pose", "index": None, "kind": "pose", "thing": "umr", "name": "umr",
                 "unit": "m", "at": None}
        meta = {"binding": [entry], "up": [0.0, 0.0, 1.0], "meters_per_unit": 1.0, "dt": 0.001, "experiment": "x", "input_hash": "h"}
        return StageRecord(np.asarray(times, dtype=np.int64), {("umr", "pose"): np.zeros((len(times), 7))}, meta)

    def staged(*records: StageRecord):
        runs = [run_of(SPEEDS, 0.001) for _ in records]
        for run, rec in zip(runs, records, strict=True):
            run.stage = rec
        return study_of(runs, [float(i) for i in range(len(runs))])

    assert code_of(lambda: compare_layer(staged(record("/World/a", [0, 5, 10]), record("/World/a", [0, 5, 15])))) == "compare.stage"
    assert code_of(lambda: compare_layer(staged(record("/World/a", [0]), record("/World/a", [0])))) == "compare.stage"
    assert code_of(lambda: compare_layer(staged(record("/World/a", [0, 5]), record("/World/b", [0, 5])))) == "compare.scene"
    assert code_of(lambda: compare_layer(staged(record("/World/a", [0, 5]), record("/World/a", [0, 5])), reference=5)) == "compare.reference"
