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


def test_a_gap_is_called_a_gap_and_the_route_fits_where_the_reference_is_a_member_of_the_same_study():
    import math

    from mime.units import Hz

    t = np.concatenate([np.arange(1, 6) * 0.001, [0.017, 0.018, 0.019, 0.020]])
    gap = Measured("m", s(t), {"probe:p.speed": m_per_s(2000.0 * t)})
    with pytest.raises(ArgumentRefused) as error:
        study().compare(gap, drives=[Hz(50.0)])
    assert "gap of 0.012 s" in error.value.refusals[0].message and "sampled every 0.012" not in error.value.refusals[0].message
    slow = study_of([run_of(SPEEDS, 0.010), run_of(SPEEDS, 0.010)], [1.0, 2.0])
    with pytest.raises(ArgumentRefused) as inside:
        slow.compare(0, drives=[Hz(50.0)])
    assert "every=" in inside.value.refusals[0].action and "reference sampled at least twice" not in inside.value.refusals[0].action
    assert math.isclose(0.010, 0.5 / 50.0)


def test_a_vector_series_is_drawn_by_its_first_component_and_the_comparison_says_so():
    wide = Quantity.of(np.stack([np.arange(1.0, 5.0), np.arange(1.0, 5.0)], axis=1), VELOCITY)
    reference = Measured("m", s(np.arange(1, 5) * 0.001), {"probe:p.speed": wide})
    two = Measured("n", s(np.arange(1, 5) * 0.001), {"probe:p.speed": wide + wide})
    from mime.results._compare import compare

    class One:
        """A study of one hand-built result, to compare two vector series."""

        experiment, axes, points, groups = "x", ("a",), (0, 1), (0, 0)
        coordinates = {"a": Quantity.of(np.asarray([1.0, 2.0]), VELOCITY)}

        def __len__(self) -> int:
            return 2

        def coordinate(self, path: str, index: int) -> object:
            return m_per_s(float(index))

        def member_data(self, index: int):
            return reference if index == 0 else two

    result = compare(One(), 0, tolerance=m_per_s(100.0))     # type: ignore[arg-type]
    assert any("component" in note for note in result.notes)


def test_a_figure_page_says_what_it_needs_where_scripts_cannot_run():
    from mime.results._figure import figure_page

    assert "<noscript>" in figure_page("{}", "t", cdn=True, caption=None, name="n")


def test_the_overlay_draws_every_chosen_members_validity_band_not_only_the_first_ones():
    members = study_of([run_of(SPEEDS, 0.001), run_of(SPEEDS, 0.001, limited=True)], [10.0, 20.0])
    bands = members.figure("probe:p.speed", kind="overlay").description.panels[0].bands
    assert [b.label for b in bands] == ["probe:p.speed_limit limit"]
    first = members.figure("probe:p.speed", kind="overlay", members=[0]).description.panels[0].bands
    assert first == ()       # control: member 0 alone has none


def test_a_folder_written_before_stage_every_names_that_key_and_the_route(copy_of_study):
    import json

    from mime.results import StudyResult

    path = copy_of_study / "study.json"
    document = json.loads(path.read_text())
    del document["run"]["stage_every"]
    path.write_text(json.dumps(document, indent=1))
    with pytest.raises(ArgumentRefused) as error:
        StudyResult.read(copy_of_study)
    refusal = error.value.refusals[0]
    assert refusal.code == "study.manifest" and "stage_every" in refusal.message
    assert "before" in refusal.action and "run_to" in refusal.action and "the MIME that wrote it" not in refusal.action
