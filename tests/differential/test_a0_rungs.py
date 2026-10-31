"""The kinematic rung against the dynamic on A0, the measurement the kinematic rung's validity line will cite (design
#99), and A0 with its magnet's mass, both as data in ``references/a0_rungs_measured.json`` (``measure_a0_rungs.py``):
these tests hold the record to what the declaration stated before it was measured (its conditions, its thresholds and
its operating points) and to its own rows (each crossing follows from the runs it is fitted to). The validity line's
form is a1's; this does not build it."""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RECORD = json.loads((HERE / "references" / "a0_rungs_measured.json").read_text())
DECLARED = json.loads((HERE / "a0_dynamic_expected_differences.json").read_text())


def test_the_record_holds_the_declared_conditions_operating_points_and_thresholds():
    stated = DECLARED["kinematic_against_dynamic"]
    assert RECORD["conditions"] == stated["conditions"]
    assert [row["frequency_hz"] for row in RECORD["runs"]] == [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    for name, found in RECORD["crossing"].items():
        assert found["threshold"] == DECLARED["tolerances"]["values"][name], name
    for row in RECORD["runs"]:
        for rung in ("kinematic", "dynamic"):
            assert all(math.isfinite(value) for value in row[rung].values()), (row["frequency_hz"], rung)


def test_each_crossing_follows_from_the_runs_it_is_fitted_to():
    """A line through zero, the rung difference against the window's largest commanded joint speed over 20 ms, crosses
    the threshold at the joint speed the record states (the declared rule; the record says why it is read with the
    runs), and the frequencies listed above the threshold are those whose difference exceeds it."""
    speeds = np.array([max(row["commanded"]["speed_over_20_ms_max_window_rad_s"]) for row in RECORD["runs"]])
    for name, found in RECORD["crossing"].items():
        moved = np.array([row["difference"][name] for row in RECORD["runs"]])
        slope = float(np.sum(speeds * moved) / np.sum(speeds ** 2))
        assert slope == found["slope_per_rad_s"], name
        if slope:
            assert found["crossing_joint_speed_rad_s"] == found["threshold"] / abs(slope), name
        assert found["above_the_threshold_at"] == [row["frequency_hz"] for row in RECORD["runs"]
                                                   if abs(row["difference"][name]) > found["threshold"]], name


def test_the_magnets_mass_is_reported_for_both_spins_and_both_frequencies_and_never_compared():
    runs = RECORD["with_the_magnets_mass"]["runs"]
    assert {(run["frequency_hz"], run["case"]) for run in runs} == {
        (frequency, case) for frequency in (3.0, 10.0)
        for case in ("spun about a diameter (declared)", "spun about its own axis (symmetric)")}
    for run in runs:
        assert "predicted" not in run and all(math.isfinite(value) for value in run["quantities"].values())
    assert "inference" in RECORD["with_the_magnets_mass"]["inferences"]
