"""``RunResult.to_mat`` (MATLAB v5, through ``scipy.io.savemat``): the declared outputs, the series with their time
axes,
the units, the flags and the name map, read back by ``scipy.io.loadmat``. MATLAB itself has not been run on a file
written here, so these tests bound what scipy's reader sees and say nothing of MATLAB's: they are not an acceptance
test against MATLAB.

Controls: each refusal (a name that cannot be mapped, a variable past the limit) is beside the same call with a name
or a size that can; a unit and a flag dropped from the file are each caught, planted in the production path."""

import json

import numpy as np
import pytest
from rigs import FLAG, SPEED, falling_bead, synthetic_run
from scipy.io import loadmat

import mime.results._mat as mat_module
from mime.results import Layout, ObservationLayout, RunResult
from mime.results._mat import LONGEST_NAME, matlab_name, names_of
from mime.units import ms, s

DT = 1e-3


def load(result: RunResult, tmp_path, name: str = "run.mat") -> dict:
    return loadmat(result.to_mat(tmp_path / name), simplify_cells=True)


def text(value) -> str:
    """A text read back: ``loadmat`` reads an empty text (a dimensionless output's unit) as an empty array."""
    return "" if np.size(value) == 0 else str(value)


def unit_problems(result: RunResult, tmp_path) -> list[str]:
    """Every output of the file whose unit or dimension is not the manifest's."""
    held = load(result, tmp_path)
    found = []
    for observation in result.manifest()["observations"]:
        for output in observation["outputs"]:
            there = held[matlab_name(observation["id"])][matlab_name(output["name"])]
            if text(there["unit"]) != output["unit"] or text(there["dimension"]) != output["dimension"]:
                found.append(output["path"])
    return found


def test_a_result_is_in_the_file_with_its_units_time_axes_flags_and_the_names_it_was_mapped_by(tmp_path):
    raised = np.zeros(12, dtype=bool)
    raised[4] = True
    run = synthetic_run(12, DT, raised=raised, every=3)
    held = load(run, tmp_path)
    assert {"run", "manifest", "flags", "name_map", "probe_p"} <= set(held)
    speed = held["probe_p"]["speed"]
    assert speed["path"] == "probe:p.speed" and speed["unit"] == "m/s" and speed["dimension"] == "velocity (m/s)"
    assert speed["time"].tolist() == pytest.approx([0.003, 0.006, 0.009, 0.012]) and speed["time_unit"] == "s"
    assert speed["values"].tolist() == [3.0, 6.0, 9.0, 12.0] and speed["steps"].tolist() == [3, 6, 9, 12]
    assert speed["every"] == 3 and speed["window"].tolist() == pytest.approx([0.0, 0.012])
    assert held["probe_p"]["where"]["values"].shape == (4, 3) and held["probe_p"]["where"]["unit"] == "m"
    mean = held["probe_p"]["mean"]
    assert mean["value"] == pytest.approx(float(np.mean(np.arange(1, 13)))) and mean["unit"] == "m/s"
    assert mean["rule"] == "mean" and mean["reduces"] == "speed"
    flag = held["flags"]["law_p_fired"]
    assert flag["path"] == FLAG.path and flag["meaning"] == FLAG.meaning and flag["was_raised"] == "true"
    assert flag["raised_steps"] == 5 and flag["raised_times"] == pytest.approx(0.005)
    assert speed["flags"]["law_p_fired"]["raised_steps"] == 1 and "raised at 1 of 12 steps" in speed["flags_text"]
    assert held["name_map"]["observations"] == {"probe_p": "probe:p"}
    assert held["name_map"]["outputs"]["probe_p"]["speed"] == "probe:p.speed"
    assert held["name_map"]["flags"] == {"law_p_fired": FLAG.path}
    assert json.loads(held["manifest"]) == json.loads(json.dumps(run.manifest(), sort_keys=True))
    assert held["run"]["experiment"] == "synthetic" and held["run"]["dt"] == DT and held["run"]["steps"] == 12


def test_a_flag_a_run_that_recorded_nothing_could_not_see_is_unknown_in_the_file_and_never_false(tmp_path):
    run = synthetic_run(8, DT, raised=np.zeros(8, dtype=bool), recorded=False)
    held = load(run, tmp_path)
    flag = held["flags"]["law_p_fired"]
    assert (flag["was_raised"], flag["raised_at_last_step"], flag["recorded"]) == ("unknown", "false", "the last step")
    assert "raised_steps" not in flag
    mean = held["probe_p"]["mean"]
    assert mean["flags"]["law_p_fired"]["was_raised"] == "unknown" and np.isnan(mean["flags"]["law_p_fired"]["steps"])
    assert held["probe_p"]["speed"]["recorded"] == 0 and "values" not in held["probe_p"]["speed"]
    recorded = load(synthetic_run(8, DT, raised=np.zeros(8, dtype=bool)), tmp_path, "recorded.mat")
    assert recorded["flags"]["law_p_fired"]["was_raised"] == "false", "the control: every step seen, none raised"


def test_a_result_read_back_from_a_run_folder_writes_the_same_file_as_the_run(tmp_path):
    experiment, swim = falling_bead(DT, 0.0105, 2500.0, 1e-9, 0.0)
    built = experiment.build()
    written = built.run_to(tmp_path / "run", s(0.012), record=[swim.speed], every=ms(3))
    live = load(written.result, tmp_path, "live.mat")
    again = load(RunResult.read(tmp_path / "run"), tmp_path, "read.mat")
    assert set(live) == set(again)
    for name in ("run", "name_map", "manifest"):
        assert live[name] == again[name] if name != "manifest" else json.loads(live[name]) == json.loads(again[name])
    live_speed, again_speed = live["swim_speed_bead"]["speed"], again["swim_speed_bead"]["speed"]
    assert np.array_equal(live_speed["values"], again_speed["values"]) and live_speed["unit"] == "m/s"
    assert np.array_equal(live_speed["time"], again_speed["time"])
    assert live["swim_speed_bead"]["mean_speed"]["value"] == pytest.approx(
        written.result[swim].mean_speed.value)


@pytest.mark.parametrize(("original", "mapped"), [
    ("swim_speed:umr.speed", "swim_speed_umr_speed"), ("drag:umr.clamp_fired", "drag_umr_clamp_fired"),
    ("1st", "x1st"), ("_a", "x_a"), ("a-b c", "a_b_c"), ("plain", "plain")])
def test_a_name_is_mapped_to_letters_digits_and_underscores_with_a_letter_first(original, mapped):
    assert matlab_name(original) == mapped
    assert names_of("output", [original]) == {original: mapped}


def test_names_that_collide_or_are_too_long_or_are_the_files_own_are_refused_by_name_and_clean_ones_are_not():
    with pytest.raises(ValueError, match=r"'a:b' and 'a\.b' map to the same MATLAB name 'a_b'.*NPZ"):
        names_of("observation", ["a:b", "a.b"])
    long = "o" * (LONGEST_NAME + 1)
    with pytest.raises(ValueError, match=rf"{long!r} maps to the MATLAB name .*{LONGEST_NAME + 1} characters.*NPZ"):
        names_of("observation", [long])
    with pytest.raises(ValueError, match="one of the file's own variables"):
        names_of("observation", ["run"], taken=mat_module.RESERVED)
    assert names_of("observation", ["o" * LONGEST_NAME, "a:b", "a:c"], taken=mat_module.RESERVED)


def test_a_result_whose_observations_collide_is_refused_when_written_and_one_whose_do_not_is_written(tmp_path):
    run = synthetic_run(4, DT)
    colliding = Layout(run.layout.experiment, DT, (run.layout.observations[0], ObservationLayout(
        "probe.p", "probe 0.1.0", run.layout.observations[0].outputs)), run.layout.flags)
    run.layout = colliding
    with pytest.raises(ValueError, match="map to the same MATLAB name 'probe_p'"):
        run.to_mat(tmp_path / "collision.mat")
    assert not (tmp_path / "collision.mat").exists()
    assert (tmp_path / "ok.mat").parent.is_dir() and synthetic_run(4, DT).to_mat(tmp_path / "ok.mat").is_file()


def test_a_variable_past_the_formats_limit_is_refused_naming_it_and_npz_before_anything_is_written(
        tmp_path, monkeypatch):
    run = synthetic_run(2000, DT)
    monkeypatch.setattr(mat_module, "VARIABLE_BYTES", 10_000)
    with pytest.raises(ValueError, match=r"variable probe_p would hold \d+ bytes.*NPZ"):
        run.to_mat(tmp_path / "big.mat")
    assert not (tmp_path / "big.mat").exists()
    monkeypatch.undo()
    assert run.to_mat(tmp_path / "big.mat").is_file(), "the control: the same run under the real limit"


def test_a_unit_dropped_from_the_file_is_caught_and_a_file_with_its_units_is_not(tmp_path, monkeypatch):
    run = synthetic_run(6, DT, every=2)
    assert unit_problems(run, tmp_path) == []
    written = mat_module._output  # pyright: ignore[reportPrivateUsage]

    def without_its_unit(result, output, names):
        return {**written(result, output, names), "unit": ""}

    monkeypatch.setattr(mat_module, "_output", without_its_unit)
    assert set(unit_problems(run, tmp_path)) == {"probe:p.speed", "probe:p.where", "probe:p.mean"}
    assert SPEED.path == "probe:p.speed"
