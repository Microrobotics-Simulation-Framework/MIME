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
from mime.units import Hz, m_per_s, rad_per_s, s

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


LOCKED = np.full(10, 0.5)                 # a lag of 0.5 rad behind the field, under 90 degrees at every step
STEPPED_OUT = np.linspace(0.0, 2.0, 10)   # a lag that passes 90 degrees (pi / 2 = 1.57)


def same_point(members, *, reference_lag=LOCKED, lags=None, dtype=np.float64, at=10.0, raised=None):
    """Members and a reference at one point (one frequency): a re-run, which is the case S2b-1's rule is for."""
    lags = lags if lags is not None else [LOCKED] * len(members)
    runs = [run_of(m, DT, lag=lag, at=at, dtype=dtype, raised=raised) for m, lag in zip(members, lags, strict=True)]
    return study_of(runs, [at] * len(runs)), run_of(REFERENCE, DT, lag=reference_lag, at=at, dtype=dtype)


def test_members_at_different_coordinates_report_their_misses_and_give_no_verdict():
    comparison = study().compare(0)
    assert {row.verdict for row in comparison.rows} == {"not judged"} and all(row.tolerance is None for row in comparison.rows)
    a = comparison.of(1, "probe:p.speed")
    assert a.largest == pytest.approx(0.25) and "different coordinates" in a.why and "field.commands.frequency" in a.why
    assert "state a tolerance" in a.why
    assert "exceeds" not in comparison.text()
    stated = study().compare(0, tolerance=m_per_s(0.5))   # a stated tolerance gives its verdict whatever the coordinates
    assert stated.of(1, "probe:p.speed").verdict == "within" and stated.of(2, "probe:p.speed").verdict == "exceeds"


def test_script_a_at_two_frequencies_gives_misses_and_no_verdict(study):
    comparison = study.compare(0, drives=[Hz(30.0)])    # 10, 20 and 30 Hz, a real study read from its folder
    assert len(comparison.rows) and {row.verdict for row in comparison.rows} == {"not judged"}
    assert any(row.largest > 0.0 for row in comparison.rows if row.output.endswith("speed"))
    assert all("different coordinates" in row.why for row in comparison.rows)


def test_at_the_same_coordinates_s2b1s_rule_gives_the_verdict_on_the_scale_over_the_run():
    members, reference = same_point([REFERENCE + 1e-12, REFERENCE + 0.25])
    comparison = members.compare(reference)
    near, far = comparison.of(0, "probe:p.speed"), comparison.of(1, "probe:p.speed")
    bound = TOLERANCE_EPS["float64"] * np.finfo(np.float64).eps * REFERENCE.max()
    assert near.tolerance == pytest.approx(bound) and near.verdict == "within" and far.verdict == "exceeds"
    mean = comparison.of(0, "probe:p.mean")
    # the reading's scale is its quantity's largest magnitude over the run (S2b-1's amendment), not its own value: the
    # mean is 11, the speed peaks at 20, and a miss of 1e-12 is within 300 eps of 20 and outside 300 eps of 11
    assert mean.tolerance == pytest.approx(bound) and mean.verdict == "within"
    assert comparison.of(1, "probe:p.mean").verdict == "exceeds"
    assert "numerical-equivalence" in comparison.rule


def test_the_rule_follows_the_scale_and_the_float_type():
    members, reference = same_point([REFERENCE * 1e6 + 1e-6])
    scaled = members.compare(run_of(REFERENCE * 1e6, DT, lag=LOCKED, at=10.0))
    assert scaled.of(0, "probe:p.speed").tolerance == pytest.approx(
        TOLERANCE_EPS["float64"] * np.finfo(np.float64).eps * 1e6 * REFERENCE.max())
    members32, reference32 = same_point([REFERENCE + 1e-4, REFERENCE + 0.25], dtype=np.float32)
    f32 = members32.compare(reference32)
    assert f32.of(0, "probe:p.speed").tolerance == pytest.approx(
        TOLERANCE_EPS["float32"] * np.finfo(np.float32).eps * REFERENCE.max())
    assert [f32.of(i, "probe:p.speed").verdict for i in (0, 1)] == ["within", "exceeds"]
    assert f32.of(0, "probe:p.mean").tolerance == pytest.approx(f32.of(0, "probe:p.speed").tolerance)   # the reading's type too


def test_where_the_regime_is_not_locked_or_cannot_be_read_there_is_no_verdict_and_the_result_says_why():
    stepped_out, reference = same_point([REFERENCE + 1e-12], lags=[STEPPED_OUT])
    out = stepped_out.compare(reference)
    assert {r.verdict for r in out.rows if r.output.startswith("probe")} == {"not judged"}
    assert all("not locked" in r.why for r in out.rows if r.output.startswith("probe"))
    unread = study_of([run_of(REFERENCE + 1e-12, DT, at=10.0)], [10.0]).compare(run_of(REFERENCE, DT, at=10.0))
    assert {r.verdict for r in unread.rows} == {"not judged"} and all("cannot be read" in r.why for r in unread.rows)
    control, ref = same_point([REFERENCE + 1e-12])      # the same pair, locked: the verdict is given
    assert control.compare(ref).of(0, "probe:p.speed").verdict == "within"


def test_a_flag_raised_in_the_pair_leaves_the_trajectory_unjudged_and_the_reading_judged():
    raised = [False] * 5 + [True] * 5
    members, reference = same_point([REFERENCE], raised=raised)
    flagged = members.compare(reference)
    series = flagged.of(0, "probe:p.speed")
    assert series.verdict == "not judged" and "flag is raised" in series.why
    assert flagged.of(0, "probe:p.mean").verdict == "within"


def test_against_a_result_that_is_not_a_member_a_miss_without_a_stated_tolerance_is_not_judged():
    reference = Measured("m", s(STEPS * DT), {"probe:p.speed": m_per_s(REFERENCE)}, readings={"probe:p.mean": m_per_s(REFERENCE.mean())})
    comparison = study().compare(reference)
    assert {row.verdict for row in comparison.rows} == {"not judged"}
    assert all("coordinates are not known" in row.why and "state a tolerance" in row.why for row in comparison.rows)
    stated = study().compare(reference, tolerance=m_per_s(0.5))
    assert stated.of(0, "probe:p.speed").verdict == "within" and stated.of(2, "probe:p.speed").verdict == "exceeds"
    assert stated.of(1, "probe:p.mean").tolerance == 0.5


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
    assert miss.end == pytest.approx(0.5) and miss.base.startswith("the member linearly interpolated onto the reference's times")
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


# ---------------------------------------------------------------------------------------------------------------
# The time bases, the drives and the folder (the refuting review's M2, M3, M6)


def code_of(call) -> str:
    with pytest.raises(ArgumentRefused) as error:
        call()
    return error.value.refusals[0].code


def test_a_member_coarser_than_the_reference_is_refused_as_a_reference_that_is_coarse_is():
    reference = Measured("m", s(np.arange(1, 101) * 0.001), {"probe:p.speed": m_per_s(2.0 * np.arange(1, 101))})   # 1 ms
    coarse = study_of([run_of(REFERENCE, 0.010), run_of(MEMBER_A, 0.010)], [1.0, 2.0])                          # 10 ms
    with pytest.raises(ArgumentRefused) as error:
        coarse.compare(reference, drives=[Hz(50.0)])
    refusal = error.value.refusals[0]
    assert refusal.code == "compare.time-base" and "member" in refusal.message and "every 0.01 s" in refusal.message
    assert "record" in refusal.action and "every=" in refusal.action        # the route is to record the member more often
    fine = study_of([run_of(REFERENCE, 0.001), run_of(MEMBER_A, 0.001)], [1.0, 2.0])
    fine.compare(reference, drives=[Hz(50.0)])                           # control: both 1 ms resolve the drive


def test_the_coarser_of_the_two_decides_and_the_interpolation_is_named_with_the_sides_spacings():
    reference = Measured("m", s(0.0025 + np.arange(10) * 0.004), {"probe:p.speed": m_per_s(np.arange(1, 11.0))})   # 4 ms, off the grid
    members = study_of([run_of(REFERENCE, 0.001), run_of(MEMBER_A, 0.001)], [1.0, 2.0])                      # 1 ms
    comparison = members.compare(reference, tolerance=m_per_s(100.0), drives=[Hz(50.0)])      # 4 ms resolves 50 Hz (10 ms half)
    base = comparison.of(0, "probe:p.speed").base
    assert "member" in base and "reference" in base and "0.001" in base and "0.004" in base
    on_grid = Measured("m", s(np.arange(1, 11) * 0.004), {"probe:p.speed": m_per_s(np.arange(1, 11.0))})
    coincident = study_of([run_of(REFERENCE, 0.004)], [1.0]).compare(on_grid, tolerance=m_per_s(100.0))
    assert coincident.of(0, "probe:p.speed").base.startswith("none")
    assert "linearly interpolated" in " ".join(comparison.description.notes)


def test_a_folders_result_without_drives_is_refused_with_the_route_and_passing_them_is_the_way(study):
    assert code_of(lambda: study.compare(0)) == "compare.drives-unknown"
    with pytest.raises(ArgumentRefused) as error:
        study.compare(0)
    assert "drives=[Hz(" in error.value.refusals[0].action
    study.compare(0, drives=[Hz(30.0)])                # stated: compared
    study.compare(0, drives=[])                        # stated as none: compared
    members = study_of([run_of(REFERENCE, DT), run_of(MEMBER_A, DT)], [1.0, 2.0])
    members.compare(0)                                  # control: a result in memory kept its drives (none declared)


def test_drives_are_frequencies_in_hertz_and_anything_else_is_refused_naming_the_unit():
    coarse = Measured("m", s(np.arange(1, 6) * 0.010), {"probe:p.speed": m_per_s(np.arange(1, 6.0))})
    members = study_of([run_of(REFERENCE, DT), run_of(MEMBER_A, DT)], [1.0, 2.0])
    for wrong in (s(0.02), 50.0, rad_per_s(2.0 * np.pi * 50.0), m_per_s(50.0)):
        with pytest.raises(ArgumentRefused) as error:
            members.compare(coarse, drives=[wrong])
        refusal = error.value.refusals[0]
        assert refusal.code == "compare.drives" and "Hz" in refusal.message + refusal.action, wrong
        with pytest.raises(ArgumentRefused) as measured:
            Measured("m", s([0.0, 1.0]), {"x": m_per_s([1.0, 2.0])}, drives=[wrong])
        assert measured.value.refusals[0].code == "compare.drives"
    with pytest.raises(ArgumentRefused):
        members.compare(coarse, drives=[Hz(50.0)])         # control: the same call with a frequency is refused for the 10 ms base
