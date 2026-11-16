"""A controller's believed models (``set_controller(believed=...)``, ``ctx.believed``): declared data, the truth where
none is declared, their numbers leaves of the controller's node inside the compiled step.

- **The belief that equals the truth** gives the run of no belief, bit for bit, in both dtypes; **one number moved**
  (the believed array placed 1 um off) changes the commands: the leaves are read, not a copy of the truth's.
- **A0's planted fault, on the coils**: the true array moves 2 mm while the belief stays put. The first tick's commands
  are those of the unmoved run (the controller follows its belief), and the field delivered at the sphere is the
  belief's at the sphere's point less the move (the truth's). The control: an allocator planted to read the truth
  moves the first tick's commands, caught on every run.
- **Keys**: an apparatus's name alone, where it holds one part, is saved and reported in full (``coils.magnetic``);
  where it holds several it is refused (``controller.believed``) naming each full key; so are a key naming nothing, a
  belief taking other commands, and a belief of an observation that is not a camera; each refusal's control, the
  belief keyed or made right, plans.
- **The saved form**: a belief is written under ``controller.believed`` by its full key and read back, building the
  derived hash of the experiment it was saved from.
- **A camera's model** is ``Camera.project`` with its nine numbers as the controller's: bit for bit at the declared
  numbers, moved by a changed focal length; read through a plan, by the camera's id."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from believed_cases import PIXEL_SCALE, DEPTH_SCALE, PROJECTED, camera_sphere, believed_sphere, coils, model_of
from chain_cases import TRACKING_READS, TRACKING_WRITES, kinematic_rpm, tracking

from mime.build import BuildRefused
from mime.control import _models
from mime.assets import asset
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.fields import CalibratedMultipole, PointDipole
from mime.units import ms

jnp: Any = jax.numpy
#: The true array's move in the planted fault.
MOVE = (2e-3, 0.0, 0.0)
#: The plant a control's detection rate is measured with (``.github/probe/detection_rate.py --plant-var AP3_PLANT``):
#: ``truth``, the allocator reading the truth whatever the belief; ``leaf``, the equal belief one number off.
PLANT = os.environ.get("AP3_PLANT", "")


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def history(experiment: Experiment, steps: int = 30) -> dict[str, dict[str, np.ndarray]]:
    held = experiment.build().run(ms(steps), history=True).history
    return {node: {field: np.asarray(value) for field, value in fields.items()} for node, fields in held.items()}


def codes(experiment: Experiment) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    return refused.value.codes()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_belief_equal_to_the_truth_is_the_run_of_none_and_one_number_moved_changes_it(dtype):
    with in_dtype(dtype):
        none = history(believed_sphere())
        same = history(believed_sphere(believed={"coils": coils(at=(1e-9, 0.0, 0.0) if PLANT == "leaf" else
                                                                (0.0, 0.0, 0.0))}))
        moved = history(believed_sphere(believed={"coils": coils(at=(1e-6, 0.0, 0.0))}))
    assert none.keys() == same.keys()
    for node, fields in none.items():
        for field, value in fields.items():
            assert np.array_equal(value, same[node][field]), (node, field)
    assert not np.array_equal(none["controller"]["write_currents"], moved["controller"]["write_currents"])


def test_the_beliefs_numbers_are_leaves_of_the_controllers_node_and_the_report_says_whose():
    built = believed_sphere().build()
    params = built.graph.get_node("controller").params
    names = sorted(name for name in params if name.startswith("believed_"))
    # those its step reads: an allocation of force reads no background, which has no gradient
    assert names == [f"believed_coils__magnetic__{leaf}" for leaf in
                     ("at", "coefficients", "orientation", "source_directions", "source_positions")]
    assert any("coils.magnetic: believed = true" in line for line in built.report.controller)
    declared = believed_sphere(believed={"coils": coils(at=(1e-6, 0.0, 0.0))}).build()
    assert any("coils.magnetic: believed calibrated_multipole 0.1.0, declared" in line
               for line in declared.report.controller)


@contextlib.contextmanager
def truth_planted() -> Iterator[None]:
    """The control: the controller's models read the truth whatever belief is declared."""
    held = _models.ControlModels._declared  # noqa: SLF001
    _models.ControlModels._declared = lambda self, key: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        yield
    finally:
        _models.ControlModels._declared = held  # type: ignore[method-assign]  # noqa: SLF001


@pytest.mark.parametrize("planted", [False, True], ids=["as_built", "control_reads_the_truth"])
def test_the_allocation_follows_the_belief_when_the_truth_moves(planted):
    """A0's planted fault on the coils: the truth 2 mm off, the belief where the truth was."""
    with in_dtype("float64"):
        unmoved = history(believed_sphere())
        with truth_planted() if planted or PLANT == "truth" else contextlib.nullcontext():
            moved = history(believed_sphere(truth=coils(at=MOVE), believed={"coils": coils()}))
        first = slice(1, 6)  # the first tick's writes act from step 1 (latency one step) to the next tick's
        same = np.array_equal(unmoved["ports:coils"]["currents"][first], moved["ports:coils"]["currents"][first])
        assert same is not planted
        if planted:
            return
        belief = model_of(believed_sphere(), "coils.magnetic")
        node = moved["coils.magnetic"]
        slot = next(name[len("flux_density_"):] for name in node if name.startswith("flux_density_"))
        poses, delivered = moved["sphere"]["pose"], moved["ports:coils"]["currents"]
        for step in range(1, 20):
            point = poses[step - 1][:3] - np.asarray(MOVE)
            made = belief({"currents": jnp.asarray(delivered[step])}, {"point": jnp.asarray(point)})
            assert np.allclose(np.asarray(made["flux_density"]), node[f"flux_density_{slot}"][step], rtol=1e-12,
                               atol=1e-15)
            unshifted = belief({"currents": jnp.asarray(delivered[step])}, {"point": jnp.asarray(poses[step - 1][:3])})
            assert not np.allclose(np.asarray(unshifted["flux_density"]), node[f"flux_density_{slot}"][step],
                                   rtol=1e-6)


# -- keys ---------------------------------------------------------------------------------------------------------

def test_a_bare_apparatus_key_is_saved_and_reported_in_full(tmp_path: Path):
    """The document holds the belief by its full key and reads it back: the experiment it declares builds the derived
    hash of the one it was written from (and the belief that moved the array is read, so the hash is not that of no
    belief)."""
    from mime.saved import dump_document

    built = believed_sphere(believed={"coils": coils(at=(1e-6, 0.0, 0.0))}).build()
    document = built.record.document
    assert document is not None and document["controller"]["believed"] == {
        "coils.magnetic": {"name": "calibrated_multipole", "version": "0.1.0",
                           "settings": {"calibration": {"asset": "navion_2", "sha256": asset("navion_2").sha256},
                                        "at": {"value": [1e-6, 0.0, 0.0], "unit": "m"}}}}
    assert any(line.startswith("    coils.magnetic: believed") for line in built.report.controller)
    path = tmp_path / "experiment.mime.yaml"
    path.write_text(dump_document(document), encoding="utf-8")
    loaded = Experiment.from_document(path).build()
    assert loaded.report.derived_hash == built.report.derived_hash
    assert loaded.report.derived_hash != believed_sphere().build().report.derived_hash


def test_a_bare_key_of_an_apparatus_of_several_parts_is_refused_naming_each():
    def declared(believed: dict[str, Any]) -> Experiment:
        exp = kinematic_rpm(controlled=("arm.joint_positions",))
        exp.set_controller(tracking(), reads=TRACKING_READS, writes=TRACKING_WRITES, mode="callback", rate=ms(2),
                           believed=believed)
        return exp

    with pytest.raises(BuildRefused) as refused:
        declared({"rpm": PointDipole(part="rotor_magnet")}).plan()
    assert refused.value.codes() == ["controller.believed"]
    assert all(key in str(refused.value) for key in ("rpm.arm", "rpm.motor", "rpm.magnetic"))
    with pytest.raises(BuildRefused) as keyed:  # the control: keyed in full, the key is read (a point dipole's belief
        declared({"rpm.magnetic": PointDipole(part="rotor_magnet")}).plan()  # is the truth's own, which is refused)
    assert "cannot differ from the truth" in str(keyed.value) and "rpm.arm" not in str(keyed.value)


@pytest.mark.parametrize("belief, code", [
    ({"nothing.magnetic": None}, "controller.believed"),
    ({"coils.magnetic": "dipole"}, "controller.believed"),
    ({"coils.magnetic": "rigid"}, "entry.unregistered"),
], ids=["a_key_naming_nothing", "a_belief_of_other_commands", "a_belief_of_another_kind"])
def test_a_belief_that_cannot_be_the_parts_is_refused_and_its_control_plans(belief, code):
    made = {key: {None: coils(), "dipole": PointDipole(part="magnet"), "rigid": RigidInertial()}[value]
            for key, value in belief.items()}
    with pytest.raises(BuildRefused) as refused:
        believed_sphere(believed=made).plan()
    assert code in refused.value.codes(), refused.value
    believed_sphere(believed={"coils.magnetic": coils()}).plan()


def test_a_belief_that_can_only_be_the_truth_is_refused_naming_the_route_and_a_belief_that_can_differ_plans():
    """A point dipole's settings name its part alone: its moment is its apparatus's and its placement its part's, so a
    ``PointDipole`` belief holds the truth's numbers whatever it says. It is refused, naming where to vary them; the
    control, a coil array's belief (its calibration, placement and background are settings), plans."""
    exp = kinematic_rpm(controlled=("arm.joint_positions",))
    exp.set_controller(tracking(), reads=TRACKING_READS, writes=TRACKING_WRITES, mode="callback", rate=ms(2),
                       believed={"rpm.magnetic": PointDipole(part="rotor_magnet")})
    with pytest.raises(BuildRefused) as refused:
        exp.plan()
    text = str(refused.value)
    assert refused.value.codes() == ["controller.believed"] and "cannot differ from the truth" in text
    assert "study leaves" in text and "leave rpm.magnetic out of believed=" in text
    believed_sphere(believed={"coils.magnetic": coils()}).plan()


def test_a_belief_of_an_observation_that_is_not_a_camera_is_refused():
    from mime.observations import SwimSpeed
    from mime.scene import X

    experiment = believed_sphere(believed={"swim_speed:sphere": SwimSpeed("sphere", along=X)})
    experiment.observe(SwimSpeed("sphere", along=X))
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert refused.value.codes() == ["controller.believed"] and "camera" in str(refused.value)


# -- a camera's model ---------------------------------------------------------------------------------------------

def test_a_cameras_model_is_its_projection_with_the_controllers_numbers():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sensing"))
    from in_loop_cases import ballistic

    plans = ballistic(ms(3)).plan()
    models = _models.ControlModels(plans)
    model = models.believed("camera")
    camera = next(p.entry for p in plans.entries if p.owner == "camera:")
    points = jnp.asarray([[0.001, -0.002, 0.0], [0.0, 0.0, 0.0], [0.002, 0.001, -0.001]])
    for found, expected in zip(model.project(points), camera.project(points), strict=True):
        assert np.array_equal(np.asarray(found), np.asarray(expected))
    moved = model.bound({"fx": model.theta["fx"] * 1.01})
    assert not np.array_equal(np.asarray(moved.project(points)[0]), np.asarray(camera.project(points)[0]))
    assert models.read == ["camera:"]


def test_a_belief_made_of_the_truths_declaration_holds_its_numbers_leaf_for_leaf():
    plans = believed_sphere(believed={"coils.magnetic": CalibratedMultipole(calibration=asset("navion_2"))}).plan()
    truth = _models.ControlModels(plans)
    held = truth.leaves(["coils.magnetic"])
    plain = _models.ControlModels(believed_sphere().plan()).leaves(["coils.magnetic"])
    assert held.keys() == plain.keys() and all(np.array_equal(held[k], plain[k]) for k in held)


# -- a camera's belief in a controller that is built ---------------------------------------------------------------
#
# A camera's key is its observation's id with a colon (``camera:``). Its numbers are leaves of the controller's node in
# graph mode, under names parsed back where the law is made again; the tests above call ``ControlModels`` directly and
# never built the node, which split ``believed:camera::fx`` into too many parts (``error.uncoded``, every graph build).

@pytest.mark.parametrize("key, parameter", [("coils.magnetic", "coefficients"), ("rpm", "moment"), ("camera:", "fx"),
                                            ("swim_speed:sphere", "k3")],
                         ids=["apparatus_part", "apparatus_alone_is_not_a_key", "observation_with_a_colon",
                              "observation_bound_to_a_body"])
def test_a_beliefs_leaf_name_round_trips_and_its_leaf_holds_no_colon(key, parameter):
    from mime.nodes.control._controller import leaf_param, split_believed_name

    name = _models.leaf_name(key, parameter)
    assert split_believed_name(name) == (key, parameter)
    assert ":" not in leaf_param(name) and leaf_param(name).startswith("believed_")


def test_the_leaves_of_a_rigs_beliefs_are_distinct_whatever_their_key_forms():
    from mime.nodes.control._controller import leaf_param

    keys = ("coils.magnetic", "coils.magnetic_2", "camera:", "camera", "swim_speed:sphere")
    names = [leaf_param(_models.leaf_name(key, parameter)) for key in keys for parameter in ("fx", "k3", "at")]
    assert len(set(names)) == len(names)


def camera_written(mode: str, believed: Any) -> tuple[np.ndarray, Any]:
    """The currents the camera probe writes late in a 30 ms run, in float64, and the experiment's plan."""
    exp = camera_sphere(believed=believed, mode=mode)
    with in_dtype("float64"):
        written = history(exp, 30)["ports:coils"]["written_currents"][-1]
    return written, exp


@pytest.mark.parametrize("mode", ["graph", "callback"])
@pytest.mark.parametrize("declared", [False, True], ids=["no_belief_declared", "belief_declared"])
def test_a_controller_built_in_either_mode_reads_the_believed_cameras_projection(mode, declared):
    """The law projects a world point through ``ctx.believed("camera")``; built and run, in the compiled step (graph) and
    on the host (callback), with the belief declared (a camera 1 % off in focal length) or not (the true camera), it
    writes the pixel and depth of the camera it believes, and the two cameras' differ (the belief is read, not the
    truth)."""
    from mime.observations import Camera, Detector

    from believed_cases import CAMERA_DETECTOR

    true = camera_sphere()
    held = next(p.entry for p in true.plan().entries if p.owner == "camera:")
    off = dict(CAMERA_DETECTOR, fx=CAMERA_DETECTOR["fx"] * 1.01)
    believed_camera = Camera(detector=Detector(**off), mount=held.mount, frame_period=held.frame_period)
    written, _ = camera_written(mode, {"camera": believed_camera} if declared else None)
    point = np.asarray([PROJECTED])
    with in_dtype("float64"):
        (pixel,), (depth,) = (np.asarray(v) for v in (believed_camera if declared else held).project(point))
        (true_pixel,), (_,) = (np.asarray(v) for v in held.project(point))
    expected = np.asarray([PIXEL_SCALE * pixel[0], PIXEL_SCALE * pixel[1], DEPTH_SCALE * depth])
    assert np.max(np.abs(written - expected)) <= 1e-12 * np.max(np.abs(expected)), (written, expected)
    assert bool(abs(written[0] / PIXEL_SCALE - true_pixel[0]) > 0.1) is declared


# -- a study over a believing controller -----------------------------------------------------------------------------
#
# A graph controller holding a belief holds the belief's numbers as leaves beside its settings' (``believed_...``, not
# ``setting_...``); the structure key built each leaf's name as a setting's, so any study over it raised a ``KeyError``.
# A belief's leaf is no study axis (an axis names a setting, a command, an initial value or a declared value): what a
# study differentiates in a believing controller is its settings, through the believed model.

def swept_final_x(values: list[float]) -> list[float]:
    from mime.study import Study, grid

    done = Study(believed_sphere("study_sphere")).sweep(grid({"controller.kp": values}), duration=ms(30)).run()
    arrays = done.arrays()
    return [float(np.asarray(arrays[f"m{i}/final/sphere/pose"])[0]) for i in range(len(values))]


def test_a_study_sweeps_a_setting_of_a_graph_controller_that_allocates_through_a_belief():
    """The sweep over ``controller.kp`` plans and runs (it raised ``KeyError: 'setting_believed:coils.magnetic:...'``),
    and two gains give two runs."""
    with in_dtype("float64"):
        low, high = swept_final_x([2400.0, 2600.0])
    assert low != high and np.isfinite([low, high]).all()


def test_a_study_over_a_graph_controller_reading_a_cameras_belief_plans():
    """The camera's belief key holds a colon (``camera:``); its leaves are in the structure key too."""
    from mime.study import Study, grid
    from mime.units import m

    with in_dtype("float64"):
        Study(camera_sphere()).sweep(grid({"sphere.initial.position": [m((1e-4, 0.0, 0.0))]}), duration=ms(10)).plan()


def test_the_derivative_in_a_controller_setting_through_a_belief_is_central_differences():
    """``Study.gradient`` in ``controller.kp`` of the sphere's final x, through a controller allocating by a belief
    (the coils' belief leaves held beside the setting), against central differences of the sweep, float64. The run's
    allocation has bound-switching kinks, so the match is to the difference's truncation: 1e-3 of the derivative."""
    from mime.study import Study, grid

    kp, h = 2500.0, 2.5
    with in_dtype("float64"):
        result = Study(believed_sphere("study_sphere")).sweep(grid({"controller.kp": [kp]}), duration=ms(30)).gradient(
            lambda final: final["sphere"]["pose"][0], wrt=["controller.kp"])
        low, high = swept_final_x([kp - h, kp + h])
    derivative = float(np.asarray(result.members[0].gradient["controller.kp"]))
    assert abs(derivative - (high - low) / (2 * h)) <= 1e-3 * abs(derivative), (derivative, (high - low) / (2 * h))
