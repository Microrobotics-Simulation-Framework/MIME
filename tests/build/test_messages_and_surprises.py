"""Usability bundle 5: messages that name the problem once, and small surprises said where they happen.

- F6b: a part whose own refusal was made (a solid's radius below zero) is one problem; what follows from the body having
  no such part (its mass derived from parts, a drag's need of its hull) is not refused as a problem of its own, and the
  part's refusal says so. A problem of the experiment that is not a consequence is still refused (the control).
- F6c: a run of zero steps, or of less than one, is named as such, and the "do:" does not offer the same run twice.
- F8: the docstring of ``run`` and the README say that a run continues the last.
- F9: an observation whose window starts at or after the run's end is warned of at ``run()``, before the run's steps.
- F15: a reason that is only a placeholder word is refused with "did you mean ``pending()``".
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))  # profile_cases, beside the tests

from profile_cases import Paperwork, settling_ball

from mime import Body, Experiment, Medium, State
from mime.build import BuildRefused
from mime.dynamics import RigidInertial
from mime.build._refusals import ArgumentRefused
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, mPa_s, ms, rad_per_s, s, um


def ball(radius=um(10.0), *, after=ms(2.0), dt=ms(0.1), viscosity=mPa_s(1.0)) -> Experiment:
    """The README's settling ball, its radius and its observation's window given."""
    exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=dt)
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0))
    exp.declare("water.viscosity", viscosity)
    body = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=radius))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0))
    exp.attach(Drag.MRS(), body=body, medium=water)
    exp.attach(Gravity(), body=body, medium=water)
    exp.observe(SwimSpeed(body, along=-Z, after=after))
    return exp


# --- F6b ---------------------------------------------------------------------------------------------------------------


def test_a_refused_solid_is_one_problem_and_says_what_follows_from_it_is_not_listed_apart():
    with pytest.raises(BuildRefused) as refused:
        ball(um(-10.0)).build()
    (held,) = refused.value.refusals
    assert held.code == "setting.value" and "radius" in held.message and "ball" in held.message
    assert held.why is not None and "what follows from it is not listed as a problem of its own" in held.why
    assert "ball's mass, inertia and hull come from its parts" in held.why


def test_control_a_problem_that_is_no_consequence_of_the_refused_solid_is_still_refused():
    """The medium's viscosity is below zero besides: that is not a consequence of the radius, and stays a problem."""
    with pytest.raises(BuildRefused) as refused:
        ball(um(-10.0), viscosity=mPa_s(-1.0)).build()
    codes = refused.value.codes()
    assert "setting.value" in codes and "value.not-positive" in codes, codes
    assert "drag.no-samples" not in codes and "parts.no-bulk" not in codes, codes


def test_control_without_a_refused_part_nothing_is_suppressed_and_no_note_is_made():
    """A hull the drag cannot sample (a body with no part) is still its own problem: only a refused part silences it."""
    exp = Experiment("no_parts", scene=Scene.new(up=Z, units="m"), dt=ms(0.1))
    water = exp.add_medium(Medium("water"))
    exp.declare("water.density", kg_per_m3(998.0))
    exp.declare("water.viscosity", mPa_s(1.0))
    body = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.attach(Drag.MRS(), body=body, medium=water)
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    assert "drag.no-samples" in refused.value.codes()
    assert all(r.why is None or "what follows from it" not in r.why for r in refused.value.refusals)


# --- F6c ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("duration", "named"), [(ms(0.0), "a run of zero steps"),
                                                 (ms(-1.0), "a negative length of time"),
                                                 (ms(0.01), "less than one base step")])
def test_a_run_of_less_than_one_step_is_named_so_and_offers_one_run_not_the_same_twice(duration, named):
    built = ball().build()
    with pytest.raises(ArgumentRefused) as refused:
        built.run(duration)
    (held,) = refused.value.refusals
    assert held.code == "run.duration" and named in held.message and "at least one base step" in held.message
    assert held.action.count("run(s(0.0001))") == 1, held.action
    assert "not a whole number of base steps" not in held.message


def test_control_a_duration_that_is_between_whole_steps_still_gives_the_two_runs_either_side():
    with pytest.raises(ArgumentRefused) as refused:
        ball().build().run(ms(0.25))
    (held,) = refused.value.refusals
    assert "not a whole number of base steps" in held.message
    assert "run(s(0.0002)) or run(s(0.0003" in held.action


# --- F8 ----------------------------------------------------------------------------------------------------------------

RUN_CONTINUES = "A run continues from where the last one ended"


def test_the_docstring_of_run_and_the_readme_say_that_a_run_continues_the_last():
    from mime.experiment import BuiltExperiment

    assert RUN_CONTINUES in (BuiltExperiment.run.__doc__ or "")
    readme = (Path(__file__).parent.parent.parent / "README.md").read_text(encoding="utf-8")
    assert RUN_CONTINUES in readme


def test_control_a_second_run_does_continue_from_the_first_so_the_sentence_is_true():
    built = ball(after=ms(1.0)).build()
    first = built.run(ms(2.0))
    second = built.run(ms(2.0))
    assert (first.start_step, first.end_step, second.start_step, second.end_step) == (0, 20, 20, 40)


# --- F9 ----------------------------------------------------------------------------------------------------------------


def test_a_window_that_starts_after_the_runs_end_is_warned_of_at_run_before_it_is_spent():
    built = ball(after=ms(2.0)).build()
    with pytest.warns(UserWarning, match=r"swim_speed:ball's window starts at 0\.002 s, at or after the end of this run "
                                         r"at 0\.001 s"):
        built.run(ms(1.0))


def test_a_run_that_continues_into_a_window_is_not_warned_of():
    """The control: the window is reached by the second run, which continues the first, so the second is not warned."""
    built = ball(after=ms(2.0)).build()
    with pytest.warns(UserWarning):
        built.run(ms(1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        built.run(ms(2.0))


def test_control_a_window_inside_the_run_is_never_warned_of():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ball(after=ms(2.0)).build().run(ms(3.0))


# --- F15 ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["pending", "TODO", "todo", " Tbd. ", "FIXME"])
def test_a_reason_that_is_only_a_placeholder_word_is_refused_with_the_call_it_meant(reason):
    with pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(profile="research", density=reason)).plan()
    held = [r for r in refused.value.refusals if r.code == "value.reason"]
    assert held and "a placeholder and not a reason" in held[0].message and "did you mean `pending()`" in held[0].message


@pytest.mark.parametrize("reason", ["x", "a steel ball's density", "pending the vendor's sheet", "to do with steel"])
def test_control_any_other_text_is_still_a_reason(reason):
    with pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(profile="research", density=reason)).plan()
    assert not [r for r in refused.value.refusals if r.code == "value.reason"], refused.value.refusals
