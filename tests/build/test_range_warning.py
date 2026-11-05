"""A warning before the run when the operating point the experiment declares lies past a law's declared range and the
build can compute it (the second walk's friction 1): the turning Reynolds number from the field's declared rate, the
body's hull and the medium, against the number the drag's rung is declared up to.

It uses the validity line's existing bound (``ReynoldsRange``) and the formula of the post-run watch, so the two cannot
disagree; it is a warning in every profile, never a refusal; the build report says what it was computed from.

**Controls.** The same experiment inside the range does not warn (the warning is an error in that arm, so a warning
raised anyway fails it); an operating point 10% either side of the bound falls either side of the warning; an
experiment with no field that turns the body has nothing to compute, and says nothing; and the pre-run number equals the
post-run watch's, read from a state turning at the declared rate."""

import math
import warnings

import numpy as np
import pytest
from range_cases import TURNING, rotating_screw

from mime import Experiment
from mime.results._watch import read_watch


def turning_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(each.message) for each in caught if TURNING in str(each.message)]


def built_with_warnings(exp: Experiment):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built = exp.build()
    return built, turning_warnings(caught), caught


def test_the_walkers_setup_warns_before_it_runs_at_the_line_that_built_it():
    with pytest.warns(UserWarning, match=TURNING) as caught:
        built = rotating_screw().build()
    # Judged on the turning warning itself: JAX may warn too, before it, and on some Pythons it does.
    (record,) = [each for each in caught if TURNING in str(each.message)]
    text = str(record.message)
    assert record.filename == __file__, "the warning is at the script's own line, not in the library"
    (watch,) = built.plans.watches
    assert watch.upper == 0.1 and watch.turning_rate == pytest.approx(2.0 * math.pi * 5.0)
    expected = 998.0 * (2.0 * math.pi * 5.0) * watch.diameter**2 / (2.0 * 1e-3)
    assert watch.turning_reynolds() == pytest.approx(expected, rel=1e-12) and expected > 100.0 * watch.upper
    for part in ("declared up to Reynolds number 0.1", "field.frequency is declared 5 Hz", "density 998 kg/m^3",
                 "viscosity 0.001 Pa s", f"{expected:.3g}", "A warning, not a refusal"):
        assert part in text, part
    assert "at the drive's rate" in text and "can turn faster than its drive" in text
    assert "the most it turns at" not in text, "the number is the drive's, not an upper bound for a body held by contact"


def test_the_build_report_says_what_it_was_computed_from():
    built = built_with_warnings(rotating_screw())[0]
    (line,) = built.report.range_warning_lines()
    assert line.startswith("warning before the run: ") and "field.frequency is declared 5 Hz" in line
    assert line in built.report.text() and line in built.report.text(details=True)
    assert "before the run, the declared rate (field.frequency is declared 5 Hz" in built.report.text()


def test_the_number_before_the_run_is_the_number_the_watch_reads_after_a_body_turning_at_that_rate():
    built = built_with_warnings(rotating_screw())[0]
    (watch,) = built.plans.watches
    final = {watch.node: {watch.field: np.array([0.0, 0.0, 0.0, watch.turning_rate, 0.0, 0.0])}}
    assert read_watch(watch, final, None).turning == watch.turning_reynolds()


def test_the_control_inside_the_range_it_does_not_warn():
    """Glycerol at 1 Hz, the regime the screw example runs in; a turning warning raised here is an error."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{TURNING}.*")
        built = rotating_screw(hz=1.0, viscosity_mpa_s=1400.0, density=1260.0).build()
    assert built.report.range_warnings() == []
    (watch,) = built.plans.watches
    assert watch.turning_reynolds() < watch.upper, "it was computed, and inside"


def test_the_warning_turns_at_the_bound_a_tenth_either_side_of_it():
    (watch,) = built_with_warnings(rotating_screw())[0].plans.watches
    at_the_bound = watch.upper * 2.0 * 1e-3 / (998.0 * watch.diameter**2) / (2.0 * math.pi)  # the Hz that gives it
    below = built_with_warnings(rotating_screw(hz=0.9 * at_the_bound))
    above = built_with_warnings(rotating_screw(hz=1.1 * at_the_bound))
    assert below[1] == [] and len(above[1]) == 1


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_it_is_a_warning_in_every_profile_never_a_refusal(profile):
    built, found, _ = built_with_warnings(rotating_screw(profile))
    assert len(found) == 1 and built.report.refusals == ()


def test_an_experiment_whose_field_turns_nothing_declared_has_no_number_to_compute_and_says_nothing():
    from profile_cases import settling_ball

    built, found, _ = built_with_warnings(settling_ball())
    assert found == [] and built.report.range_warnings() == []
    assert all(watch.turning_rate is None and watch.turning_reynolds() is None for watch in built.plans.watches)
