"""Controller settings as leaves (round 3, item 6): a setting a study's axis names (``controller.<setting>``) is a leaf
of the controller's node inside the compiled step where its step reads it, the law made again from the leaves inside
the traced step; one its start alone reads reaches the run through each point's own start; one the law reads through
``self`` is refused exactly (``controller.setting-unread``: one trace with the setting as an argument). Released
programs keep their constants: nothing becomes a leaf unless an axis names it.

The rig: ``tests/control_cases.py``'s spun magnet, closed inside the compiled step by a follower that reads its gain and
smoothing through its context (``planted_gradients.TracedFollower``); and A0's tracking controller, whose numbers
(smoothing, gain, damping) it now reads through its context too."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = str(Path(__file__).resolve().parents[1] / "study")
if STUDY not in sys.path:
    sys.path.append(STUDY)

from chain_cases import tracked_rpm, tracking  # noqa: E402
from control_cases import spun_magnet  # noqa: E402
from planted_gradients import TracedFollower  # noqa: E402

import mime.control._run as run_module  # noqa: E402
from mime.build import ArgumentRefused, BuildRefused  # noqa: E402
from mime.study._axes import candidates, resolve  # noqa: E402
from mime.study import Study, along  # noqa: E402
from mime.units import ms  # noqa: E402

GAINS = [40.0, 50.0, 60.0]
H = 1e-4


def tilt(final):
    return final["umr"]["pose"][4]


def followed(name: str = "followed", **settings):
    return spun_magnet(name, controller=TracedFollower(**settings), mode="graph")


@pytest.fixture(scope="module")
def swept():
    """The gain's sweep; ``MIME_PLANT_CONTROLLER=self`` has the law read its gain through ``self`` (the gate kit's
    rate of the exact check's control)."""
    through_self = os.environ.get("MIME_PLANT_CONTROLLER") == "self"
    study = Study(followed(through_self=through_self)).sweep(along({"controller.gain": GAINS}), duration=ms(30))
    try:
        return study, study.run()
    except BuildRefused as refused:  # the plant's refusal, which the test then fails on, never errors on
        return study, refused


def test_a_sweep_of_a_setting_the_step_reads_is_one_group_one_compile_and_each_member_its_own_build(swept):
    study, result = swept
    assert not isinstance(result, BuildRefused), f"the sweep was refused: {result}"
    plan = study.plan()
    assert len(plan.groups) == 1 and plan.compiles == 1
    leaves = plan.groups[0].built.graph.params["nodes"]["controller"]
    assert set(leaves) == {"setting_gain"}
    for index, gain in enumerate(GAINS):
        own = followed(f"own_{index}", gain=gain).build().run(ms(30)).final["umr"]["pose"]
        found = np.asarray(result.member(index).final["umr"]["pose"])
        assert np.allclose(found, np.asarray(own), rtol=1e-12, atol=1e-15), (gain, found - np.asarray(own))


def test_without_an_axis_the_controller_holds_no_leaf():
    assert dict(followed("plain").build().graph.params["nodes"]["controller"]) == {}


def test_the_derivative_in_a_controller_setting_is_the_rebuilt_central_difference(swept):
    """The MRS gradient oracle's protocol (``tests/oracles/test_mrs_gradients.py``, check 3): the central difference
    that rebuilds at ``h`` and ``h / 2``, its truncation estimated by their difference and its rounding as 64 eps of
    the value over the step; the derivative within ``1e-6`` of the finer plus four times both. The closed loop's
    objective bends more in the gain than gate 9's consumers in theirs, so gate 9's fixed line is not this rig's."""
    study, _ = swept
    found = float(study.gradient(tilt).member(1).gradient["controller.gain"])

    def at(gain: float) -> float:
        return float(tilt(followed("rebuilt", gain=gain).build().run(ms(30)).final))

    def difference(h: float) -> float:
        return (at(50.0 * (1 + h)) - at(50.0 * (1 - h))) / (2 * h * 50.0)

    coarse, fine = difference(H), difference(H / 2)
    rounding = 64 * np.finfo(float).eps * abs(at(50.0)) / (H * 50.0)
    assert abs(found - fine) <= 1e-6 * abs(fine) + 4 * (abs(coarse - fine) + rounding), (found, coarse, fine)


def test_a_setting_only_the_start_reads_is_carried_by_each_points_start():
    """No leaf: the node's start, which its description holds, is each point's own (so the points group apart, as a
    controller's start always has)."""
    plan = Study(followed("started")).sweep(along({"controller.start": [0.0, 0.1]}), duration=ms(30)).plan()
    assert not plan.refused
    assert all(dict(group.built.graph.params["nodes"]["controller"]) == {} for group in plan.groups)
    assert [float(point.state["controller"]["state_filtered"]) for point in plan.points] == [0.0, 0.1]


def test_control_a_setting_read_through_self_is_refused_exactly():
    plan = Study(followed("selfread", through_self=True)).sweep(along({"controller.gain": GAINS[:2]}),
                                                               duration=ms(30)).plan()
    assert [[refusal.code for refusal in point.refusals] for point in plan.points] == \
        [["controller.setting-unread"]] * 2
    with pytest.raises(BuildRefused):
        Study(followed("selfread_run", through_self=True)).sweep(along({"controller.gain": GAINS[:2]}),
                                                                 duration=ms(30)).run()


def test_the_tracking_controllers_numbers_read_through_its_context_sweep_as_one_group():
    """A0's tracking controller inside the compiled step: its smoothing, which it read through ``self`` before, is now
    read through its context, so a sweep of it is one group (one a value before)."""
    plan = Study(tracked_rpm("tracked_graph", mode="graph")).sweep(
        along({"controller.smoothing": [0.005, 0.006]}), duration=ms(4)).plan()
    assert not plan.refused and len(plan.groups) == 1
    assert set(plan.groups[0].built.graph.params["nodes"]["controller"]) == {"setting_smoothing"}


def test_a_fit_through_a_callback_controller_is_refused():
    """A fit steps on its derivative, and a controller called on the host is a cut path: refused, with the route to the
    compiled step."""
    from mime.units import A_m2

    exp = spun_magnet("fitted_callback", controller=TracedFollower(), mode="callback")
    with pytest.raises(BuildRefused) as raised:
        Study(exp).fit(tilt, start={"spin.magnetic_moment": A_m2(18.89)}, noise=1.0, duration=ms(30))
    assert [refusal.code for refusal in raised.value.refusals] == ["fit.callback"]


INTEGERS = {"gain": 2, "smoothing": 1, "damping": 0}


@pytest.fixture(autouse=True)
def planted_float_only(monkeypatch):
    """``MIME_PLANT_CONTROLLER=float_only`` has the context's bare numbers the fields holding a float alone, as the first
    form of this stage took them (the gate kit's rate of the integer tests)."""
    if os.environ.get("MIME_PLANT_CONTROLLER") == "float_only":
        monkeypatch.setattr(run_module, "numeric_settings", lambda entry: {
            field.name: float(getattr(entry, field.name)) for field in dataclasses.fields(entry)
            if isinstance(getattr(entry, field.name), float)})


def test_an_integer_written_for_a_float_setting_is_that_float_and_a_count_or_a_switch_is_not_a_setting():
    """``ArmTracking(gain=2)`` type-checks (an integer is a float for a type checker) and built before the law read its
    numbers through its context: the number the context holds is the float, whatever was written."""
    written = run_module.numeric_settings(tracking(**INTEGERS))
    floats = run_module.numeric_settings(tracking(**{name: float(value) for name, value in INTEGERS.items()}))
    assert written == floats and all(type(value) is float for value in written.values())
    assert {"gain", "smoothing", "damping"} <= set(written) and not {"iterations", "orientation_feedback"} & set(written)
    assert "orientation_feedback" not in run_module.numeric_settings(tracking(orientation_feedback=True))


@pytest.mark.parametrize("mode", ["callback", "graph"])
def test_a_controller_written_with_integers_for_its_float_settings_builds_in_both_modes(mode):
    """The blocker of the review of this stage: on the release these builds (an integer for a float field), and the first
    form of the context's numbers refused them as an entry defect."""
    built = tracked_rpm(f"integers_{mode}", mode=mode, controller=tracking(**INTEGERS)).build()
    assert built.plans.controller is not None


READ_AS_BUILT = ("iterations", "orientation_feedback", "end_effector", "axis", "description", "orientation")


@pytest.mark.parametrize("setting", READ_AS_BUILT)
def test_a_setting_read_as_built_is_no_axis_inside_the_compiled_step_and_the_refusal_names_those_that_are(setting):
    """A count, a switch, a name, a direction or a file is read as built, so inside the compiled step no leaf can hold
    it: ``controller.iterations`` was offered and crashed uncoded when planned."""
    _, declarations, _ = tracked_rpm("not_a_leaf", mode="graph").declared_as()
    with pytest.raises(ArgumentRefused) as raised:
        resolve(f"controller.{setting}", declarations)
    (refusal,) = raised.value.refusals
    assert refusal.code == "controller.setting-not-leaf"
    assert all(f"controller.{name}" in refusal.action for name in ("gain", "smoothing", "damping", "standoff"))
    offered = [path for path in candidates(declarations) if path.startswith("controller.")]
    assert f"controller.{setting}" not in offered and "controller.gain" in offered


def test_as_a_callback_every_setting_is_a_members_own_law_and_so_an_axis():
    _, declarations, _ = tracked_rpm("callback_axes", mode="callback").declared_as()
    assert resolve("controller.iterations", declarations).name == "iterations"
    assert "controller.iterations" in candidates(declarations)
