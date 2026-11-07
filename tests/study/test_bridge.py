"""The bridge from a study to an experiment, ``result.member(i).as_experiment(name)``: gate 6 of S2c-1.

The gate: script A's reproduction arm at 20 and 30 Hz (both locked, the base at 10 Hz: the points and the run of the
predictions' amendment 1, with its reason), the study written to its folder;
each member made an experiment of its own, built and run alone, is that member's run, within the tolerance of S2b-1's
gate 1 per field in eps of the field's scale over the run (``predictions_s2c1.json``, committed before the run), in float32
and in float64; and its saved folder's ``based_on`` names the base. The controls: a member made an experiment with its
override dropped (it is the base, at 10 Hz) and a member made an experiment and compared with another member must each
fail that tolerance, by orders of magnitude, on every trial. The tolerance is the locked regime's, so the regime is
checked on every run: each member's run alone keeps the lag of the body's moment behind the field within 90 degrees at
every step (measured before the amendment: at most 50.5 and 80.8 degrees). What the comparison does not claim: it is the
sweep's acceptance (a batch against the same point alone), not an accuracy claim about the model."""

from __future__ import annotations

import json
import warnings

import jax
import numpy as np
import pytest
import yaml
from compare import TOLERANCE_EPS, misses, worst
from folder_rigs import SPEED
from script_a import reproduction_arm

from mime import Study
from mime.build._refusals import BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.study import grid
from mime.units import Hz, kg, s

DURATION = s(0.2)
FREQUENCIES = (20.0, 30.0)


@pytest.fixture(scope="module", params=["float32", "float64"])
def gate(request, tmp_path_factory):
    """The study written to its folder in one precision, each member made an experiment and run alone (keeping every
    step, so that each field's scale over the run is known), and the base run alone at its own 10 Hz: the control's
    experiment, which is a member with its override dropped."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        folder = tmp_path_factory.mktemp(request.param) / "study"
        exp = reproduction_arm()
        field = exp.handle("field")
        result = Study(exp).sweep(grid({field.commands.frequency: [Hz(f) for f in FREQUENCIES]}),  # type: ignore[union-attr]
                                  duration=DURATION, record=[SPEED]).run_to(folder)
        bridged, runs = {}, {}
        for i in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bridged[i] = result.member(i).as_experiment(f"bridge_{i}")
            runs[i] = alone(bridged[i])
        dropped = alone(Experiment.based_on(folder / "experiment", name="dropped"))
        yield {"dtype": request.param, "result": result, "folder": folder, "bridged": bridged, "runs": runs,
               "dropped": dropped}
    finally:
        jax.config.update("jax_enable_x64", held)


def alone(experiment: Experiment):
    """The experiment run alone, keeping every step."""
    return experiment.build().run(DURATION, history=True)


def lag(run) -> np.ndarray:
    """The lag of the body's moment (its body y) behind the field about the tube's axis (x), at every step, in rad."""
    from mime.core import quaternion

    history = run.history
    moments = np.asarray(jax.vmap(lambda q: quaternion.rotate(q, jax.numpy.array([0.0, 1.0, 0.0])))(
        jax.numpy.asarray(history["umr"]["pose"][:, 3:])))
    fields = np.asarray(history["field.magnetic"]["flux_density_magnetic_torque_umr"])
    return np.arctan2(np.cross(moments, fields)[:, 0], np.sum(moments * fields, axis=1))


def test_both_members_are_locked_so_that_the_tolerance_is_the_locked_regimes(gate):
    worst_lag = {f: float(np.degrees(np.max(np.abs(lag(gate["runs"][i]))))) for i, f in enumerate(FREQUENCIES)}
    print("largest lag behind the field, degrees", gate["dtype"], worst_lag)
    assert all(value < 90.0 for value in worst_lag.values()), worst_lag


def test_a_member_made_an_experiment_is_the_members_own_run(gate):
    found = {i: worst(misses(gate["runs"][i], gate["result"].member(i), [SPEED])) for i in range(2)}
    print("bridge misses in eps", gate["dtype"], found, "tolerance", TOLERANCE_EPS[gate["dtype"]])
    assert all(miss <= TOLERANCE_EPS[gate["dtype"]] for _, miss in found.values()), found


def test_the_controls_a_dropped_override_and_two_members_exchanged_fail_by_orders_of_magnitude(gate):
    result, tolerance = gate["result"], TOLERANCE_EPS[gate["dtype"]]
    seen = []
    for i in range(2):
        seen.append(("dropped", i, worst(misses(gate["dropped"], result.member(i), [SPEED]))[1]))
        seen.append(("exchanged", i, worst(misses(gate["runs"][1 - i], result.member(i), [SPEED]))[1]))
    print("control misses in eps", gate["dtype"], seen)
    assert sum(miss > 100 * tolerance for _, _, miss in seen) == 4, seen


def test_the_saved_experiment_records_the_base_it_came_from_and_the_member_that_replaced_it(gate, tmp_path):
    experiment = gate["bridged"][1]
    experiment.save(tmp_path / "bridge_saved")
    document = yaml.safe_load((tmp_path / "bridge_saved" / DOCUMENT_FILE).read_text())
    base_lock = json.loads((gate["folder"] / "experiment" / LOCK_FILE).read_text())
    assert document["based_on"]["experiment"] == "script_a"
    assert document["based_on"]["input_hash"] == base_lock["hashes"]["input"]
    assert "30" in json.dumps(document["apparatus"] if "apparatus" in document else document)
    assert experiment.plan().replaced, "the declarations the member's values replaced are in plan().replaced"


def test_a_value_axis_member_declares_its_value_with_the_reason_from_the_study(tmp_path):
    exp = reproduction_arm()
    study = Study(exp).sweep(grid({"umr.mass": [kg(8.2e-5), kg(8.8e-5)]}), duration=s(0.02),
                             reasons={"umr.mass": "the two ends of the printed body's mass range"})
    result = study.run_to(tmp_path / "study")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = result.member(1).as_experiment("heavier")
    declared = [d for d in experiment.declared_as()[1] if str(getattr(d, "path", "")) == "umr.mass"]
    assert len(declared) == 1 and declared[0].value.value == pytest.approx(8.8e-5)
    assert declared[0].reason == f"from study {result.record['id']}, member 1: the two ends of the printed body's mass range"


def test_a_run_that_is_no_member_and_a_member_with_no_saved_base_are_refused_and_a_named_base_is_taken(gate):
    result = gate["result"]
    plain = gate["dropped"]
    with pytest.raises(BuildRefused) as refused:
        plain.as_experiment("x")
    assert refused.value.codes() == ["study.not-a-member"]
    memory = Study(reproduction_arm()).sweep(grid({"umr.mass": [kg(8.2e-5), kg(8.8e-5)]}), duration=s(0.01)).run()
    with pytest.raises(BuildRefused) as refused:
        memory.member(0).as_experiment("x")
    assert refused.value.codes() == ["study.no-base"] and "run_to" in refused.value.refusals[0].action
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        named = memory.member(0).as_experiment("x", base=result.base)  # the base named
    assert named.name == "x"


def test_a_member_that_ran_in_a_precision_the_study_varied_cannot_be_made_an_experiment(tmp_path):
    from folder_rigs import toy
    from mime.study import PRECISION

    result = Study(toy()).sweep(grid({PRECISION: ["float32", "float64"]}), duration=s(0.1)).run_to(tmp_path / "study")
    with pytest.raises(BuildRefused) as refused:
        result.member(1).as_experiment("p")
    assert refused.value.codes() == ["study.bridge-precision"] and "JAX_ENABLE_X64=1" in refused.value.refusals[0].action
