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

from believed_cases import believed_sphere, coils, model_of
from chain_cases import TRACKING_READS, TRACKING_WRITES, kinematic_rpm, tracking

from mime.build import BuildRefused
from mime.control import _models
from mime.data import asset
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
