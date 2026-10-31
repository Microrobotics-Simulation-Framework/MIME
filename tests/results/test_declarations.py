"""What an entry declares for a run's result (design #184 items 6 and 9): an observation's outputs, each with its
dimension, per step or at the end, and the flags an entry's node raises. Each is refused at registration when it is
malformed, and at the build when the node it names does not hold it; each control is the same declaration made well,
which registers and builds."""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from mime.build import BuildRefused
from mime.cli import main as mime_main
from mime.gradients import GradientStatus
from mime.registry import (
    Flag,
    Observation,
    Output,
    OutputKind,
    Read,
    Reduction,
    RoleKind,
    register_observation,
)
from mime.results import RUN_MANIFEST, RunResult
from mime.testing.toys import POSITION, ToyRecorder, coupled_toy
from mime.testing.toys._nodes import ToyRecorderNode
from mime.units import LENGTH, VELOCITY, s
from mime.vocabulary import register_namespace

register_namespace("declaring", "0.0.1")


def recorder(name: str, outputs: tuple[Output, ...], *, raises: tuple[Flag, ...] = ()) -> type[Observation]:
    """An observation of the toy body built as the toy's recorder is, declaring ``outputs`` and ``raises``."""
    namespace = {"__module__": __name__, "__qualname__": f"Declaring_{name}", "__annotations__": {"body": Any},
                 "body": None, "name": f"declaring.{name}", "version": "0.0.1",
                 "differentiability": GradientStatus.EXACT, "builds": {"": (ToyRecorderNode,)},
                 "roles": {"body": RoleKind.BODY}, "reads": {"position": Read("body", POSITION.name)},
                 "outputs": outputs, "raises": raises, "build": ToyRecorder.build}
    return dataclass(frozen=True, kw_only=True)(type(f"Declaring_{name}", (Observation,), namespace))


LAST = Output("last", LENGTH, "the position", shape=(3,))
SAMPLES = Output("samples", LENGTH, "a count, declared as a length for the test", kind=OutputKind.AT_END)


@pytest.mark.parametrize(("outputs", "refusal"), [
    ((), "outputs: a non-empty tuple of mime.registry.Output"),
    ((LAST, LAST), "'last' is declared twice"),
    ((Output("Last", LENGTH, "a position", shape=(3,)),), "its name is a lower-case word"),
    ((LAST, Output("mean", LENGTH, "a mean", kind=OutputKind.AT_END, reduces="lost", rule=Reduction.MEAN)),
     "it reduces 'lost', which is to be a per-step output of its node"),
    ((LAST, Output("mean", LENGTH, "a mean", kind=OutputKind.AT_END, reduces="last")),
     "names the per-step output it reduces and its rule"),
    ((LAST, Output("mean", LENGTH, "a mean", kind=OutputKind.AT_END, after="body")),
     "its window starts at the setting 'body', which is to be a setting of dimension time"),
])
def test_a_malformed_output_declaration_is_refused_at_registration_naming_what_is_wrong(outputs, refusal):
    with pytest.raises(TypeError, match=refusal):
        register_observation(recorder("malformed", outputs))


def test_a_well_formed_declaration_registers_and_a_flag_on_no_rung_of_the_entry_is_refused():
    register_observation(recorder("well_formed", (LAST, Output(
        "mean", LENGTH, "a mean", kind=OutputKind.AT_END, reduces="last", rule=Reduction.MEAN)),
        raises=(Flag("samples", "a flag, for the test"),)))
    with pytest.raises(TypeError, match="on rung 'fast', which is not one of its rungs"):
        register_observation(recorder("flagged", (LAST,), raises=(Flag("samples", "a flag", rung="fast"),)))
    with pytest.raises(TypeError, match="with what it means when it is raised"):
        register_observation(recorder("silent", (LAST,), raises=(Flag("samples", " "),)))


def test_an_output_or_a_flag_its_node_does_not_hold_is_refused_at_the_build_and_one_it_holds_builds():
    holds = register_observation(recorder("holds", (LAST, SAMPLES), raises=(Flag("samples", "a flag"),)))
    lost = register_observation(recorder("lost", (LAST, Output("lost", VELOCITY, "not held", kind=OutputKind.AT_END)),
                                         raises=(Flag("gone", "a flag it does not hold"),)))
    shaped = register_observation(recorder("shaped", (Output("last", LENGTH, "the position, as a scalar"),)))
    for cls, problems in ((lost, ["the output 'lost'", "the flag 'gone'"]), (shaped, ["with shape ()"])):
        experiment = coupled_toy()
        experiment.observe(cls(body="probe"))
        with pytest.raises(BuildRefused) as refused:
            experiment.build()
        assert set(refused.value.codes()) == {"entry.defect"}
        assert all(any(problem in r.message for r in refused.value.refusals) for problem in problems), \
            refused.value
    experiment = coupled_toy()
    handle = experiment.observe(holds(body="probe"))
    result = experiment.build().run(s(0.1), record=[handle.last])
    assert result[handle].samples.value == 5.0 and [f.path for f in result.layout.flags] == [
        "declaring.holds:probe.samples"]


def test_mime_run_records_what_record_names_one_every_every_and_writes_the_manifest(tmp_path):
    from mime.testing.toys import full_toy

    full_toy().save(tmp_path / "saved")
    out = tmp_path / "run"
    assert mime_main(["run", str(tmp_path / "saved"), "--duration", "0.24", "--out", str(out),
                      "--record", "toy.recorder:probe.last", "--every", "0.04"]) == 0
    result = RunResult.read(out)
    assert result.recorded == ("toy.recorder:probe.last",) and result.every == 2
    assert result["toy.recorder:probe"].last.time.to(s) == pytest.approx([0.04, 0.08, 0.12, 0.16, 0.2, 0.24])
    assert json.loads((out / RUN_MANIFEST).read_text())["recorded"] == ["toy.recorder:probe.last"]
    assert mime_main(["run", str(tmp_path / "saved"), "--duration", "0.24", "--out", str(tmp_path / "bad"),
                      "--record", "toy.recorder:probe.samples"]) == 2
    assert not (tmp_path / "bad").exists()
