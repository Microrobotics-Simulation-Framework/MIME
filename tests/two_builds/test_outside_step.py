"""MIME-ANO-146's detection: a callback controller's setting (A0's tracking controller's ``home``, ``tracked_rpm``)
reaches the run through the controller's held writes, the graph's external inputs, and not through the step's
parameters. The trace form sees the step only, so it is quiet; the run form, which writes the leaves that differ into
the base's program but feeds it the base's own inputs, fires; and the check says where the value went
(``inputs_moved``) and gives the verdict ``outside-step``. Its control: the same experiment built twice fires neither
form (and says it moved nothing). The settings the compiled step reads are leaves where a study's axis names them
(``tests/gradients/test_controller_leaves.py``), which resolves the anomaly for those; the callback mode stays outside
the step's parameters by design, and this is how the check reports it, on every run.

**What is not resolved**: a setting only the law's *start* reads (``home``: the controller node's initial state and its
first write) in the graph mode (``mode="graph"``) is neither a leaf nor an input the check sees. Two graph-mode builds
with ``home`` moved give the verdict ``quiet`` (the trace is the same, no leaf moved, no held input moved, the run form
reads no change in the base's program) while their runs differ by 3.2e-4 relative at 1e-4 of ``home``
(:func:`test_a_start_only_setting_in_the_graph_mode_is_quiet_to_the_check_while_the_run_moves`): the check reports two
runs as agreeing that do not, the anomaly's hazard, open for this case.

The ladder: ``home`` moved by a relative 1e-6 (the spike's), 1e-5 and 1e-4, every joint at once. The controller holds
its pose in float32 (its writes are the arm's joint positions in float32), so a move below about 6e-8 does not reach
the run at all, and the check then rightly finds nothing moved."""

from __future__ import annotations

import numpy as np
import pytest
from chain_cases import AR4, AR4_BASE, AR4_HOME, tracked_rpm

from mime.control import ArmTracking
from mime.scene import X, Z, asset
from mime.testing import two_builds
from mime.units import m, ms, rad


def tracking_at(home: tuple[float, ...]) -> ArmTracking:
    """``chain_cases.tracking``'s controller with its ``home`` written."""
    return ArmTracking(description=asset(AR4), end_effector="link_6", home=rad(home), axis=X, up=Z, at=m(AR4_BASE))


@pytest.fixture(scope="module")
def base():
    return tracked_rpm().build()


@pytest.mark.parametrize("move", [1e-6, 1e-5, 1e-4])
def test_a_callback_controllers_setting_is_reported_outside_the_step(base, move: float):
    other = tracked_rpm(controller=tracking_at(tuple(h * (1.0 + move) for h in AR4_HOME))).build()
    found = two_builds(base, other)
    assert not found.trace_fired, "the step's program is the same"
    assert not found.moved, "no leaf moved"
    assert found.inputs_moved, "the controller's held writes moved"
    assert found.verdict == "outside-step"
    assert found.run is not None and found.run.ratio >= 4.0, "four times k resolutions above the run form's line"


def test_control_the_same_controller_built_twice_fires_neither_form(base):
    found = two_builds(base, tracked_rpm().build())
    assert not found.fired and not found.inputs_moved and found.verdict == "not-discriminating"
    assert found.run is not None and found.run.ratio == 0.0


def test_a_start_only_setting_in_the_graph_mode_is_quiet_to_the_check_while_the_run_moves():
    """MIME-ANO-146, open for this case: ``home`` is read only by the controller node's start, so inside the compiled
    step it is no leaf and no input, and the check says ``quiet`` of two builds whose runs differ (3.2e-4 relative at a
    1e-4 move of ``home``, measured on the review of the PR that made the other settings leaves)."""
    base = tracked_rpm("home_graph_base", mode="graph").build()
    other = tracked_rpm("home_graph_moved", mode="graph",
                        controller=tracking_at(tuple(h * (1.0 + 1e-4) for h in AR4_HOME))).build()
    found = two_builds(base, other)
    assert found.verdict == "quiet" and not found.fired and not found.inputs_moved
    first, second = (built.run(ms(8)).final for built in (base, other))
    worst = max(float(np.max(np.abs(np.asarray(first[node][name], float) - np.asarray(second[node][name], float))) /
                      (np.max(np.abs(np.asarray(first[node][name], float))) + 1e-300))
                for node in first for name in first[node]
                if np.asarray(first[node][name]).shape == np.asarray(second[node][name]).shape
                and np.asarray(first[node][name]).size)
    assert worst > 1e-5, "the two runs differ, which the check did not say"
