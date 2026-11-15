"""The iterative path through the real path: a built controller node inside the compiled step and as a callback, a
study's sweep and ``Study.gradient``, with its run flag laid out, raised and counted.

The rigs are A0's arm moved by :class:`iterative_cases.ArmFollower` (a nonlinear model, the iterative path by its
declaration) and the coil sphere of ``believed_cases`` with the exact path's size patched below its three coils
(``iterative_cases.iterative_coils``), whose exact path is the oracle. A node's step is traced once a build, so each
patch is in force before the build it acts on.

**Its plants** (``ITERATIVE_PLANT``): ``silent``, the flag never raised, which the flag test must catch on every run;
``unpolished``, the solve not polished past its tolerance, which the follower's derivative test must catch. The
claim's plants (an early stop, a loosened tolerance) are ``test_iterative_allocation``'s, whose draws hold active
bounds that take more than two iterations to find; the sphere's force is met inside its limits in one Newton step."""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest
from believed_cases import believed_sphere
from iterative_cases import ArmFollower, follower_rpm, iterative_coils

import mime.control._iterative as module
import mime.control._models as models
from mime.build import BuildRefused
from mime.units import ms

PLANT = os.environ.get("ITERATIVE_PLANT", "")
#: link_6's index among the AR4's frames.
FLANGE = 24


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def history(experiment: Any, steps: int = 20) -> dict[str, dict[str, np.ndarray]]:
    built = experiment.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return built.run(ms(steps * float(built.report.dt) * 1e3), history=True).history


def built_and_run(experiment: Any, duration: Any) -> tuple[Any, Any]:
    built = experiment.build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = built.run(duration, history=True)
    return result, [str(w.message) for w in caught]


# -- the arm, a nonlinear model ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_follower_is_one_function_in_both_modes_and_holds_the_flange_above_the_body(dtype):
    """The follower's joint positions, written inside the compiled step and on the host, agree to the dtype's rounding
    over the run, and its flange stands its standoff above where the body was read."""
    with in_dtype(dtype):
        graph = history(follower_rpm("follower_graph", mode="graph"))
        callback = history(follower_rpm("follower_callback", mode="callback"))
    joints = np.asarray(graph["rpm.arm"]["joint_positions"])
    eps = float(np.finfo(dtype).eps)
    assert np.max(np.abs(joints - np.asarray(callback["rpm.arm"]["joint_positions"]))) <= 1e4 * eps
    flange = np.asarray(graph["rpm.arm"]["frame_poses"])[-1, FLANGE, :3]
    body = np.asarray(graph["umr"]["pose"])[-1, :3]
    # The last write was made a few base steps before the run's end, from the body's pose then; the body moves
    # micrometres in that time.
    assert np.linalg.norm(flange - body - np.asarray((0.0, 0.0, 0.15))) <= 1e-4


def test_the_follower_on_the_iterative_path_says_so_in_its_report_and_lays_out_its_flag():
    with in_dtype("float64"):
        built = follower_rpm("follower_report").build()
    lines = "\n".join(built.report.controller)
    assert "iterative (its model is not affine in joint_positions), 6 commands of rpm.arm's model" in lines
    assert "allocation_unconverged (warns)" in lines
    assert [flag.path for flag in built.layout.flags if flag.owner == "controller"] == \
        ["controller.allocation_unconverged"]


def test_an_exact_path_controller_declares_no_flag_and_builds_no_holder():
    """The coil sphere on the exact path: no flag laid out, no holder node in the callback mode, and the report names
    the exact path."""
    with in_dtype("float64"):
        for mode in ("graph", "callback"):
            built = believed_sphere(f"exact_{mode}", mode=mode).build()
            assert not [flag for flag in built.layout.flags if flag.owner == "controller"]
            assert "controller.flags" not in built.graph.node_names
            assert "exact, 3 commands of coils.magnetic's model" in "\n".join(built.report.controller)


def test_an_unregularised_follower_is_refused_three_rows_not_fixing_six_joints():
    with in_dtype("float64"), pytest.raises(BuildRefused) as caught:
        follower_rpm("follower_unregularised", effort=0.0).build()
    assert caught.value.codes() == ["allocation.objective"] and "3 rows of 6 commands" in str(caught.value)


@pytest.mark.parametrize("mode", ["graph", "callback"])
def test_an_unconverged_solve_raises_its_flag_counted_and_warned_and_a_converged_one_none(mode, monkeypatch):
    """Capped at one iteration, with its flange asked 2 m above the body (beyond the arm's reach), every tick's solve
    stops short: the run raises allocation_unconverged at every step, counts it and warns of it, its value the
    stationarity. The same rig uncapped and reachable raises none."""
    with in_dtype("float64"):
        clean, said = built_and_run(follower_rpm(f"clean_{mode}", mode=mode), ms(10))
        assert not [read for read in clean.flags if read.flag.name == "allocation_unconverged"] and not \
            [text for text in said if "allocation_unconverged" in text]
        monkeypatch.setattr(module, "ITERATIONS", 1)
        if PLANT == "silent":
            from mime.nodes.control import _flags

            monkeypatch.setattr(_flags, "raise_flag", lambda name, value: None)
            monkeypatch.setattr(models, "raise_flag", lambda name, value: None)
        capped, said = built_and_run(follower_rpm(f"capped_{mode}", mode=mode, standoff=2.0), ms(10))
    raised = [read for read in capped.flags if read.flag.name == "allocation_unconverged"]
    assert raised and raised[0].raised == raised[0].steps, raised
    assert any("allocation_unconverged" in text for text in said), said
    holder = "controller" if mode == "graph" else "controller.flags"
    values = np.asarray(capped.history[holder]["flag_allocation_unconverged"])
    assert np.all(values > module.TOLERANCE[np.dtype("float64")])


# -- the coil sphere, against the exact path ---------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("mode", ["graph", "callback"])
def test_the_coil_sphere_on_the_iterative_path_writes_the_exact_paths_currents(dtype, mode, monkeypatch):
    """The R2 sphere's PD force allocated on the iterative path writes the exact path's currents, step by step over
    the closed loop, within the dtype's rounding of the 25 A limit amplified by the loop (1e4 units in the last place),
    and raises no flag."""
    with in_dtype(dtype):
        exact = history(believed_sphere(f"exact_{mode}", mode=mode), 60)
        with iterative_coils(monkeypatch):
            built = believed_sphere(f"iterative_{mode}", mode=mode).build()
            assert "iterative (3 commands, more than EXACT_COMMANDS = 2)" in "\n".join(built.report.controller)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = built.run(ms(60), history=True)
    raised = [read for read in result.flags if read.flag.name == "allocation_unconverged"]
    assert not raised and not [w for w in caught if "allocation_unconverged" in str(w.message)]
    eps = float(np.finfo(dtype).eps)
    ours = np.asarray(result.history["ports:coils"]["written_currents"])
    theirs = np.asarray(exact["ports:coils"]["written_currents"])
    assert np.max(np.abs(ours - theirs)) <= 1e4 * eps * 25.0, np.max(np.abs(ours - theirs))


# -- a study, and its derivative --------------------------------------------------------------------------------

def test_a_study_sweeps_a_setting_of_a_controller_on_the_iterative_path():
    from mime.study import Study, grid

    with in_dtype("float64"):
        done = Study(follower_rpm("study_follower")).sweep(grid({"controller.standoff": [0.14, 0.16]}),
                                                           duration=ms(10)).run()
    arrays = done.arrays()
    low, high = (np.asarray(arrays[f"m{i}/final/rpm.arm/joint_positions"]) for i in range(2))
    assert np.isfinite(low).all() and np.isfinite(high).all() and not np.array_equal(low, high)


def test_the_derivative_through_the_iterative_path_is_central_differences(monkeypatch):
    """``Study.gradient`` in ``controller.kp`` of the coil sphere's final x, its currents allocated on the iterative
    path (the implicit derivative of each tick's solve), against central differences of the sweep, float64: as the
    exact path's test, to 1e-3 of the derivative, the truncation of the difference over the loop's bound-switching
    kinks."""
    from mime.study import Study, grid

    kp, h = 2500.0, 2.5
    with in_dtype("float64"), iterative_coils(monkeypatch):
        study = Study(believed_sphere("study_sphere_iterative"))
        result = study.sweep(grid({"controller.kp": [kp]}), duration=ms(30)).gradient(
            lambda final: final["sphere"]["pose"][0], wrt=["controller.kp"])
        swept = Study(believed_sphere("study_sphere_iterative")).sweep(
            grid({"controller.kp": [kp - h, kp + h]}), duration=ms(30)).run().arrays()
    low, high = (float(np.asarray(swept[f"m{i}/final/sphere/pose"])[0]) for i in range(2))
    derivative = float(np.asarray(result.members[0].gradient["controller.kp"]))
    assert abs(derivative - (high - low) / (2 * h)) <= 1e-3 * abs(derivative), (derivative, (high - low) / (2 * h))
    assert not [flag for flag in result.members[0].flags if "allocation_unconverged" in str(flag)]


def test_the_derivative_of_the_follower_in_its_standoff_is_central_differences(monkeypatch):
    """``Study.gradient`` in ``controller.standoff`` of the arm's final second joint, through five ticks of the
    follower's solves (a nonlinear model, the exact Hessian's tangent) and the body's contact, against central
    differences of the sweep, float64, within 1e-6 (measured 1e-8 at this step; the differences' truncation at 0.1 mm
    is 6e-8). Each tick's solve starts at the joints it reads, a step or two from the minimiser, on a weakly regularised
    problem (cond 1e8), where the derivative is sensitive to the point it is taken at: without polishing, the
    tolerance left the commands 4e-8 rad from the minimiser and this derivative 7.6e-4 off (ITERATIVE_PLANT=unpolished,
    this test's control)."""
    from mime.study import Study, grid

    standoff, h = 0.15, 1e-4
    if PLANT == "unpolished":
        monkeypatch.setattr(module, "POLISHING_STEPS", 0)
    with in_dtype("float64"):
        result = Study(follower_rpm("study_follower_gradient")).sweep(grid({"controller.standoff": [standoff]}),
                                                                      duration=ms(10)).gradient(
            lambda final: final["rpm.arm"]["joint_positions"][1], wrt=["controller.standoff"])
        swept = Study(follower_rpm("study_follower_gradient")).sweep(
            grid({"controller.standoff": [standoff - h, standoff + h]}), duration=ms(10)).run().arrays()
    low, high = (float(np.asarray(swept[f"m{i}/final/rpm.arm/joint_positions"])[1]) for i in range(2))
    derivative = float(np.asarray(result.members[0].gradient["controller.standoff"]))
    assert abs(derivative - (high - low) / (2 * h)) <= 1e-6 * abs(derivative), (derivative, (high - low) / (2 * h))


__all__: list[str] = ["ArmFollower"]
