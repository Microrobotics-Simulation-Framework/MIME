"""The run's length is saved with the experiment (the second walk's friction 5; saved form version 5): a script that
declares ``Experiment(..., duration=s(3))`` saves it as ``run.duration``, and ``mime run`` on the folder needs no
``--duration`` (it may still override).

Version 5 added the key, so a document of an earlier form holds none and a document of an earlier form that does is
refused, never read as declaring it. A folder written before it (version 4, and a version 1 folder with no duration)
loads as it did and still needs ``--duration``.

**Controls.** The duration is part of the hashed input: a saved duration edited after its lock is refused
(``load.input-hash``), so it is not decoration. A run with no duration declared and none given is refused with the
code that names both ways out. An override beats the saved length. A duration that is not a whole number of base steps
is refused when the experiment is built, never saved."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from profile_cases import Paperwork, settling_ball

from mime.build import BuildRefused
from mime.cli import main as mime_main
from mime.results import RunResult
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.saved._folder import LoadRefused
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, DURATION_ADDED, document_in_form, upgraded_document
from mime.units import ms

DT_MS = 1.0
FIVE_STEPS = ms(5.0)
V4 = Path(__file__).parent / "stage_writers" / "chain_screw"
V1 = Path(__file__).parent / "previous" / "magnetic_ball"


def saved_ball(tmp_path: Path, duration=FIVE_STEPS) -> Path:
    settling_ball(Paperwork(density="steel", acceptance="Newtonian", duration=duration)).save(tmp_path / "saved")
    return tmp_path / "saved"


def run_steps(folder: Path, out: Path, *extra: str) -> tuple[int, int | None]:
    code = mime_main(["run", str(folder), "--out", str(out), *extra])
    return code, (json.loads((out / "outputs.json").read_text())["steps"] if (out / "outputs.json").is_file() else None)


def test_the_duration_is_saved_as_run_duration_in_the_current_form(tmp_path: Path):
    folder = saved_ball(tmp_path)
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
    assert document["schema"] == DOCUMENT_SCHEMAS[CURRENT] == "mime.document/6"
    assert document["run"]["duration"] == {"value": 0.005, "unit": "s"}
    assert json.loads((folder / LOCK_FILE).read_text())["schema"] == "mime.lock/6"


def test_a_saved_folder_runs_as_long_as_it_declared_with_no_duration_given(tmp_path: Path, capfd):
    code, steps = run_steps(saved_ball(tmp_path), tmp_path / "run")
    assert (code, steps) == (0, 5), capfd.readouterr()
    assert RunResult.read(tmp_path / "run").layout.dt == pytest.approx(DT_MS * 1e-3)


def test_an_override_beats_the_saved_length(tmp_path: Path, capfd):
    code, steps = run_steps(saved_ball(tmp_path), tmp_path / "run", "--duration", "2ms")
    assert (code, steps) == (0, 2), capfd.readouterr()


def test_the_built_experiment_reads_the_length_it_declared(tmp_path: Path):
    from mime import Experiment

    built = Experiment.load(saved_ball(tmp_path)).build()
    assert built.duration is not None and float(built.duration.value) == pytest.approx(0.005)
    assert settling_ball(Paperwork(density="steel", acceptance="Newtonian")).build().duration is None


def test_the_control_a_saved_duration_edited_after_its_lock_is_refused(tmp_path: Path):
    folder = saved_ball(tmp_path)
    text = (folder / DOCUMENT_FILE).read_text()
    assert "value: 0.005" in text
    (folder / DOCUMENT_FILE).write_text(text.replace("value: 0.005", "value: 0.007"))
    from mime import Experiment

    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_the_control_a_folder_with_no_duration_still_needs_one(tmp_path: Path, capfd):
    folder = saved_ball(tmp_path, duration=None)
    assert "duration" not in yaml.safe_load((folder / DOCUMENT_FILE).read_text())["run"]
    code, _ = run_steps(folder, tmp_path / "run")
    shown = capfd.readouterr()
    assert code == 2 and "run.duration" in shown.err + shown.out
    assert "declares no run length" in shown.err + shown.out and "Experiment(..., duration=s(3))" in shown.err + shown.out
    assert run_steps(folder, tmp_path / "run2", "--duration", "3ms") == (0, 3)


def test_an_earlier_form_loads_without_a_duration_and_still_needs_one(tmp_path: Path, capfd):
    from mime import Experiment

    folder = tmp_path / "chain_screw"
    shutil.copytree(V4, folder)
    assert yaml.safe_load((folder / DOCUMENT_FILE).read_text())["schema"] == "mime.document/4"
    experiment = Experiment.load(folder)
    assert experiment.load_report is not None and experiment.load_report.version == 4
    assert experiment._settings.duration is None
    assert "declares no licence" not in experiment.load_report.text(), "version 4 declares terms, 5 only added a run length"
    code, _ = run_steps(folder, tmp_path / "run")
    assert code == 2 and "run.duration" in capfd.readouterr().err


def test_the_control_an_earlier_form_holding_the_key_of_version_five_is_refused(tmp_path: Path):
    from mime import Experiment

    for source in (V4, V1):
        folder = tmp_path / source.name
        shutil.copytree(source, folder)
        document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
        document["run"]["duration"] = {"value": 0.005, "unit": "s"}
        (folder / DOCUMENT_FILE).write_text(yaml.safe_dump(document, sort_keys=False))
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(folder)
        assert "load.document-shape" in refused.value.codes()
        assert any("run.duration" in each.message and f"version {CURRENT}" in each.message
                   for each in refused.value.refusals), [each.message for each in refused.value.refusals]


def test_the_upgrade_and_the_form_before_it_are_inverse_for_a_duration():
    document = yaml.safe_load((V4 / DOCUMENT_FILE).read_text())
    upgraded, version, _ = upgraded_document(document)
    assert version == 4 and upgraded["schema"] == DOCUMENT_SCHEMAS[CURRENT]
    assert document_in_form(upgraded, 4) == document
    with_duration = {**upgraded, "run": {**upgraded["run"], "duration": {"value": 0.005, "unit": "s"}}}
    assert "duration" not in document_in_form(with_duration, DURATION_ADDED - 1)["run"]
    assert "duration" in document_in_form(with_duration, CURRENT)["run"]


@pytest.mark.parametrize(("duration", "why"), [(ms(2.5), "not a whole number of base steps"),
                                                (ms(0.0), "not a finite time above zero"),
                                                (ms(-3.0), "not a finite time above zero")])
def test_the_control_a_duration_a_run_would_refuse_is_refused_at_the_build_and_never_saved(duration, why, tmp_path):
    exp = settling_ball(Paperwork(density="steel", acceptance="Newtonian", duration=duration))
    with pytest.raises(BuildRefused) as refused:
        exp.save(tmp_path / "saved")
    assert "run.duration" in refused.value.codes() and any(why in each.message for each in refused.value.refusals)
    assert not (tmp_path / "saved" / DOCUMENT_FILE).exists()


#: Lengths a run holds to the rule of whole base steps (dt is 1 ms): the build's check of a declared duration and a run's
#: ``steps_in`` read one function, so each is accepted by both or refused by both.
LENGTHS = [ms(1.0), ms(5.0), ms(3.0 * (1.0 + 1e-10)), ms(3.0 * (1.0 + 1e-6)), ms(2.5), ms(0.4), ms(1e-300), ms(1e9)]


@pytest.mark.parametrize("length", LENGTHS, ids=[f"{each.value!r}" for each in LENGTHS])
def test_the_build_and_a_run_hold_a_length_to_one_rule(length, tmp_path: Path):
    ball = settling_ball(Paperwork(density="steel", acceptance="Newtonian"))
    built = ball.build()
    try:
        built.steps_in(length)
        run_accepts = True
    except Exception:  # noqa: BLE001 - the refusal is the other arm of the comparison
        run_accepts = False
    try:
        settling_ball(Paperwork(density="steel", acceptance="Newtonian", duration=length)).build()
        build_accepts = True
    except BuildRefused:
        build_accepts = False
    assert build_accepts == run_accepts


def test_the_control_the_rule_accepts_and_refuses_in_this_table_so_the_comparison_can_tell_the_two_apart():
    built = settling_ball(Paperwork(density="steel", acceptance="Newtonian")).build()
    accepted = []
    for length in LENGTHS:
        try:
            built.steps_in(length)
            accepted.append(True)
        except Exception:  # noqa: BLE001
            accepted.append(False)
    assert any(accepted) and not all(accepted)


def test_a_duration_of_several_times_is_refused_for_what_it_is_not_for_being_non_finite():
    with pytest.raises(BuildRefused) as refused:
        settling_ball(Paperwork(density="steel", acceptance="Newtonian", duration=ms((1.0, 2.0)))).build()
    (message,) = [each.message for each in refused.value.refusals if each.code == "run.duration"]
    assert "holds 2 times" in message and "a run has one length" in message and "finite" not in message
