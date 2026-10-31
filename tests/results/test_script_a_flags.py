"""Script A's flag, today's one (design #184 item 9): the surrogate's ``clamp_fired``, raised where the body's offset
from the tube's axis lies beyond the clamp near the wall, on every window a result reads and in its folder's manifest.
The control: from where script A starts, inside the clamp, the same runs raise nothing, so what is reported is what
the node raised."""

import json

import pytest
from script_a import script_a

from mime.results import RUN_MANIFEST, RunResult
from mime.units import mm, ms, s

#: Past the clamp (about 1.05 mm from the axis, for FL-9 in the 1/4" tube): the drag reads the pose at the start of
#: the step, so the first step is clamped before the contact projects the body back inside its margin.
PAST_THE_CLAMP = mm((10.0, 0.0, -1.06))
CLAMP = "drag:umr.clamp_fired"
#: The surrogate's two flags for a tilt it cannot see (7l-c fix round): declared, and not raised by script A, whose body
#: lies along the tube.
TILT_FLAGS = ["drag:umr.tilt_fired", "drag:umr.bore_fired"]


@pytest.fixture(scope="module")
def clamped():
    return script_a(start=PAST_THE_CLAMP).build()


def test_the_clamp_raised_in_the_first_step_is_on_every_window_that_holds_it_and_in_the_manifest(clamped, tmp_path):
    written = clamped.run_to(tmp_path / "run", ms(5), record=["swim_speed:umr.speed"])
    result = written.result
    (read,) = result.flags  # the clamp alone: the body lies along the tube, so neither tilt flag is raised
    assert (read.path, read.raised, read.steps, read.first) == (CLAMP, 1, 10, pytest.approx(5e-4))
    speed = result["swim_speed:umr"].speed
    assert [r.path for r in speed.flags] == [CLAMP] and speed.between(ms(0.5)).flags == ()
    assert CLAMP in result["swim_speed:umr"].mean_speed.text() and CLAMP in result.table()["flags"][0]
    manifest = json.loads((tmp_path / "run" / RUN_MANIFEST).read_text())
    assert [f["path"] for f in manifest["flags"]] == [CLAMP, *TILT_FLAGS] and manifest["run_flags"][0]["raised_steps"] == 1
    for output in manifest["observations"][0]["outputs"]:
        if output["kind"] == "at_end":
            assert [f["flag"] for f in output["flags"]] == [CLAMP], output["name"]
    later = clamped.run(ms(5))
    (read, *others) = later.flags  # the clamp, and the two tilt flags, none of them known before the last step
    assert [r.path for r in others] == TILT_FLAGS and all(r.raised is None for r in others)
    assert read.raised is None and read.was_raised is None and "unknown, not recorded per step" in str(later)
    assert "not raised" in str(later) and "before it is unknown" in str(later)
    assert later.manifest()["run_flags"][0]["was_raised"] is None


def test_a_run_that_records_nothing_reports_a_flag_raised_at_its_last_step_as_raised(tmp_path):
    """Through ``run()``'s own path: one step from past the clamp, the only step is the last, and it clamps. The
    control, from script A's own start, reads the same step as clear at the last step and unknown before it."""
    fresh = script_a(start=PAST_THE_CLAMP).build()
    written = fresh.run_to(tmp_path / "run", ms(0.5))
    for result in (written.result, RunResult.read(tmp_path / "run")):
        read = next(r for r in result.flags if r.path == CLAMP)
        assert (read.at_last, read.was_raised) == (True, True) and "at the run's last step it was raised" in str(result)
    assert json.loads((tmp_path / "run" / RUN_MANIFEST).read_text())["flags"][0]["raised_at_last_step"] is True
    own = script_a().build().run(ms(0.5))
    assert (own.flags[0].path, own.flags[0].at_last, own.flags[0].was_raised) == (CLAMP, False, None)


def test_from_script_as_own_start_the_same_runs_raise_nothing():
    built = script_a().build()
    assert built.run(ms(5), record=["swim_speed:umr.speed"]).flags == ()
    assert built.layout.flags[0].path == CLAMP, "the flag is declared, and not raised"
    assert "flags: none raised" in str(built.run(s(0.005), record=["swim_speed:umr.speed"])["swim_speed:umr"].speed)
