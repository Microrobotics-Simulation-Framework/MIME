"""A study's members against a named reference: each miss equals a hand-computed miss, exchanging the reference and a
member is detected, the tolerance is a rule or a stated quantity and never a fixed absolute, a comparison is made on the
reference's time base with a stated interpolation, and it is refused where the reference's base cannot resolve a declared
drive frequency."""

from __future__ import annotations

import numpy as np
import pytest
from rigs_plot import run_of, study_of

from mime.build._refusals import ArgumentRefused
from mime.results import TOLERANCE_EPS, Measured
from mime.units import Hz, m_per_s, s

DT = 0.001
STEPS = np.arange(1, 11)
REFERENCE = 2.0 * STEPS            # the reference's speed: 2 k
MEMBER_A = 2.0 * STEPS + 0.25      # a constant offset 0.25
MEMBER_B = 2.0 * STEPS + 0.1 * STEPS   # a drift 0.1 k: 1.0 at the end, the largest at the end


def study():
    return study_of([run_of(REFERENCE, DT), run_of(MEMBER_A, DT), run_of(MEMBER_B, DT)], [10.0, 20.0, 30.0])


def test_each_miss_equals_the_hand_computed_miss():
    comparison = study().compare(0, tolerance={"probe:p.speed": m_per_s(0.5)})
    a, b = comparison.of(1, "probe:p.speed"), comparison.of(2, "probe:p.speed")
    assert (a.end, a.largest) == (0.25, 0.25) and a.at == pytest.approx(0.001) and a.verdict == "within"
    assert b.end == pytest.approx(1.0) and b.largest == pytest.approx(1.0) and b.at == pytest.approx(10 * DT)
    assert b.verdict == "exceeds" and b.tolerance == 0.5
    window = study().compare(0, window=(s(0.0), s(0.005)), tolerance={"probe:p.speed": m_per_s(0.5)})
    assert window.of(2, "probe:p.speed").largest == pytest.approx(0.5) and window.of(2, "probe:p.speed").end == pytest.approx(0.5)
    mean = comparison.of(2, "probe:p.mean")
    assert mean.end == pytest.approx(MEMBER_B.mean() - REFERENCE.mean()) and mean.at is None
    assert mean.verdict == "not judged" and "It bounds the model's error from above only against a measurement" in comparison.text()


def test_control_the_reference_and_a_member_exchanged_is_detected():
    hand = {1: 0.25, 2: 1.0}  # member minus reference, signed, at the end
    detected = trials = 0
    for reference, member in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        miss = study().compare(reference, tolerance={"probe:p.speed": m_per_s(10.0)}).of(member, "probe:p.speed").end
        exchanged = study().compare(member, tolerance={"probe:p.speed": m_per_s(10.0)}).of(reference, "probe:p.speed").end
        ends = {0: REFERENCE[-1], 1: MEMBER_A[-1], 2: MEMBER_B[-1]}
        assert miss == pytest.approx(ends[member] - ends[reference])
        trials += 1
        detected += not np.isclose(exchanged, miss)
    assert detected == trials == 6 and hand[1] == 0.25


def test_the_tolerance_is_the_studys_rule_between_its_members_and_never_a_fixed_absolute():
    comparison = study().compare(0)
    a = comparison.of(1, "probe:p.speed")
    assert a.tolerance == pytest.approx(TOLERANCE_EPS["float64"] * np.finfo(np.float64).eps * REFERENCE.max())
    assert a.verdict == "exceeds" and "machine epsilons" in comparison.rule
    scaled = study_of([run_of(REFERENCE * 1e6, DT), run_of(MEMBER_A * 1e6, DT)], [10.0, 20.0]).compare(0)
    assert scaled.of(1, "probe:p.speed").tolerance == pytest.approx(1e6 * a.tolerance)  # it follows the scale
    f32 = study_of([run_of(REFERENCE, DT, dtype=np.float32), run_of(REFERENCE, DT, dtype=np.float32)], [1.0, 2.0]).compare(0)
    assert f32.of(1, "probe:p.speed").tolerance == pytest.approx(
        TOLERANCE_EPS["float32"] * np.finfo(np.float32).eps * REFERENCE.max()) and f32.of(1, "probe:p.speed").verdict == "within"


def test_against_a_result_that_is_not_a_member_a_miss_without_a_stated_tolerance_is_not_judged():
    reference = Measured("m", s(STEPS * DT), {"probe:p.speed": m_per_s(REFERENCE)}, readings={"probe:p.mean": m_per_s(REFERENCE.mean())})
    comparison = study().compare(reference)
    assert {row.verdict for row in comparison.rows} == {"not judged"}
    assert all("no tolerance is stated" in row.why for row in comparison.rows)
    stated = study().compare(reference, tolerance=m_per_s(0.5))
    assert stated.of(0, "probe:p.speed").verdict == "within" and stated.of(2, "probe:p.speed").verdict == "exceeds"
    assert stated.of(1, "probe:p.mean").tolerance == 0.5


def test_a_flag_raised_in_the_pair_leaves_the_trajectory_unjudged_and_the_reading_judged():
    raised = [False] * 5 + [True] * 5
    flagged = study_of([run_of(REFERENCE, DT), run_of(REFERENCE, DT, raised=raised)], [1.0, 2.0]).compare(0)
    series = flagged.of(1, "probe:p.speed")
    assert series.verdict == "not judged" and "flag is raised" in series.why
    assert flagged.of(1, "probe:p.mean").verdict == "within"


def test_across_two_time_bases_the_comparison_is_on_the_references_with_a_stated_interpolation():
    # A reference every 4 ms (a sampled measurement) against runs every 1 ms: the reference's times are 4, 8, ...; the
    # member is linear in time, so the interpolation is exact and the hand-computed miss is the member's value there.
    coarse_t = np.arange(1, 4) * 0.004
    reference = Measured("m", s(coarse_t), {"probe:p.speed": m_per_s(2000.0 * coarse_t)})
    member = run_of(2000.0 * np.arange(1, 13) * DT + 0.5, DT)   # k dt -> speed 2000 t + 0.5, exactly linear
    comparison = study_of([member, run_of(2000.0 * np.arange(1, 13) * DT, DT)], [1.0, 2.0]).compare(reference, tolerance=m_per_s(1.0))
    row = comparison.of(0, "probe:p.speed")
    assert row.end == pytest.approx(0.5) and row.largest == pytest.approx(0.5) and row.verdict == "within"
    assert "none: the member has a sample at every one of the reference's times" == row.base   # 4, 8, 12 ms are on the run's grid
    off_grid = Measured("m", s([0.0015, 0.0045, 0.0075]), {"probe:p.speed": m_per_s(2000.0 * np.asarray([0.0015, 0.0045, 0.0075]))})
    miss = study_of([member, member], [1.0, 2.0]).compare(off_grid, tolerance=m_per_s(1.0)).of(0, "probe:p.speed")
    assert miss.end == pytest.approx(0.5) and miss.base == "the member linearly interpolated onto the reference's times"
    assert "linearly interpolated" in study_of([member, member], [1.0, 2.0]).compare(off_grid).description.notes[1]


def test_a_reference_that_cannot_resolve_a_declared_drive_frequency_is_refused_with_the_route():
    coarse = Measured("m", s(np.arange(1, 6) * 0.010), {"probe:p.speed": m_per_s(np.arange(1, 6.0))})   # 10 ms
    members = study_of([run_of(REFERENCE, DT), run_of(MEMBER_A, DT)], [1.0, 2.0])
    with pytest.raises(ArgumentRefused) as error:
        members.compare(coarse, drives=[Hz(50.0)])
    refusal = error.value.refusals[0]
    assert refusal.code == "compare.time-base" and "every 0.01 s" in refusal.message and "50 Hz" in refusal.message
    assert "sampled at least twice as often" in refusal.action
    fine = Measured("m", s(np.arange(1, 6) * 0.001), {"probe:p.speed": m_per_s(np.arange(1, 6.0))})
    members.compare(fine, drives=[Hz(50.0)])      # 1 ms against 50 Hz resolves it


def test_outputs_that_are_not_the_same_are_refused_and_the_figure_marks_what_a_miss_says():
    other = Measured("m", s(STEPS * DT), {"another": m_per_s(REFERENCE)})
    with pytest.raises(ArgumentRefused) as error:
        study().compare(other)
    assert error.value.refusals[0].code == "compare.outputs"
    description = study().compare(0, tolerance=m_per_s(0.5)).description
    assert "bounds the model's error from above only against a measurement" in description.notes[0]
    assert [p.title for p in description.panels] == ["probe:p.speed", "probe:p.speed: miss"]
    assert description.panels[1].bands[0].high == 0.5
