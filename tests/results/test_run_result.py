"""``RunResult`` (design #184 items 5 to 10): one object from a run and from a run's folder, indexed by an observation's
handle or id; per-step outputs as series only when recorded, a series not recorded refused naming the ``record=`` that
keeps it; windows by time under one inclusion rule; units read by ``.to()``; flags on every window, in every export and
printout; and the exports (a table pandas and polars take as it is, CSV with its units in its header, NPZ, and
``outputs.json`` beside ``outputs.npz``, read without MIME)."""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rigs import FLAG, SPEED, WHERE, falling_bead, synthetic_run

import mime
from mime.experiment import RUN_EXPERIMENT, RUN_OUTPUTS, RUN_RECORD
from mime.results import RUN_MANIFEST, NotRecorded, Reading, RunResult, Series
from mime.testing.toys import full_toy
from mime.units import UnitError, mm, mm_per_s, ms, s

PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
DT = 1e-3


@pytest.fixture
def bead():
    experiment, swim = falling_bead(DT, 0.0105, 2500.0, 1e-9, 0.0)
    return experiment.build(), swim


# -- recording ------------------------------------------------------------------------------------------------------


def names_its_record(message: str, path: str) -> bool:
    """Whether a refusal of a series not recorded names the ``record=`` that keeps it."""
    return f"record=[{path!r}]" in message


def test_a_series_not_recorded_is_refused_naming_the_record_to_add_and_a_recorded_one_is_read(bead):
    built, swim = bead
    unrecorded = built.run(s(0.02))
    with pytest.raises(NotRecorded) as refused:
        _ = unrecorded[swim].speed
    assert names_its_record(str(refused.value), "swim_speed:bead.speed")
    assert isinstance(unrecorded[swim].mean_speed, Reading), "an output at the end is always there"
    recorded = built.run(s(0.02), record=[swim.speed])
    assert isinstance(recorded[swim].speed, Series) and len(recorded[swim].speed) == 20
    assert not names_its_record("swim_speed:bead.speed is a per-step output, and this run did not record it",
                                "swim_speed:bead.speed"), "the check passes a refusal that names no record="


def test_record_takes_handles_paths_and_observations_and_refuses_what_is_not_a_per_step_output(bead):
    built, swim = bead
    for record in ([swim.speed], ["swim_speed:bead.speed"], [swim], ["swim_speed:bead"]):
        assert built.run(s(0.005), record=record).recorded == ("swim_speed:bead.speed",)
    with pytest.raises(ValueError, match="an output at the end, which the graph reduces"):
        built.run(s(0.005), record=["swim_speed:bead.mean_speed"])
    with pytest.raises(ValueError, match="Did you mean 'swim_speed:bead.speed'"):
        built.run(s(0.005), record=["swim_speed:bead.sped"])
    with pytest.raises(AttributeError, match="its outputs are speed, mean_speed, distance, observed"):
        _ = swim.sped
    with pytest.raises(ValueError, match="nothing is recorded"):
        built.run(s(0.005), every=ms(2))
    with pytest.raises(ValueError, match="not a whole number of base steps"):
        built.run(s(0.005), record=[swim.speed], every=ms(1.5))


def test_every_spaces_the_samples_by_time_from_the_graphs_start_across_runs(bead):
    built, swim = bead
    first = built.run(s(0.007), record=[swim.speed], every=ms(2))
    assert first[swim].speed.time.to(ms) == pytest.approx([2.0, 4.0, 6.0])
    second = built.run(s(0.005), record=[swim.speed], every=ms(2))
    assert second.start.value == pytest.approx(0.007)
    assert second[swim].speed.time.to(ms) == pytest.approx([8.0, 10.0, 12.0])
    with pytest.raises(NotRecorded, match="reaches outside it"):
        second[swim].speed.between(s(0.0))
    assert "not recorded before 0.007 s" not in second[swim].mean_speed.text(), "no flag is declared here"


def test_history_records_every_per_step_output_at_every_step():
    built = full_toy().build()
    result = built.run(s(0.2), history=True)
    assert result.history is not None and result.recorded == ("toy.recorder:probe.last",)
    last = result["toy.recorder:probe"].last
    assert np.array_equal(last.raw, result.history["toy.recorder:probe"]["last"])
    assert np.allclose(result.time.to(s), last.time.to(s))


# -- windows by time ------------------------------------------------------------------------------------------------


@PROPERTY
@given(steps=st.integers(2, 400), dt=st.floats(1e-5, 1e-1), data=st.data())
def test_a_window_holds_the_steps_whose_end_lies_after_its_start_and_not_after_its_end(steps, dt, data):
    """The inclusion rule against the step counts themselves: a start or an end drawn on a step's end, or anywhere
    between two (at least a hundredth of a step from either), holds exactly the steps counted after the start and up
    to the end. The run is float32, and a start on the grid is decided as a node decides its ``after``, by the
    rounding of ``k dt`` in float32, so the step ending there is held when float32 puts it above the start (the test
    restates that arithmetic; a start off the grid is unambiguous). A window is judged by the times handed over, in
    seconds: offsets that differ in steps can multiply out to one time (``dt`` of 1e-5 with offsets 0.99 and 0.01 on
    adjacent steps did), and a window whose end is not after its start is refused by name, which the test expects
    there."""
    run = synthetic_run(steps, dt)
    first = data.draw(st.integers(0, steps - 1), label="the start's step")
    last = data.draw(st.integers(first + 1, steps), label="the end's step")
    into = data.draw(st.one_of(st.just(0.0), st.floats(0.01, 0.99)), label="the start's offset into the next step")
    back = data.draw(st.one_of(st.just(0.0), st.floats(0.01, 0.99)), label="the end's offset back into its step")
    start, end = (first + into) * dt, (last - back) * dt
    if not end > start:
        with pytest.raises(ValueError, match="ends after it starts"):
            run["probe:p"].speed.between(s(start), s(end))
        return
    window = run["probe:p"].speed.between(s(start), s(end))
    ties = into == 0.0 and first > 0 and np.float32(first) * np.float32(dt) > np.float32(start)
    end = last - 1 if back else last
    expected = [k for k in range(1, steps + 1) if (k > first or (ties and k == first)) and k <= end]
    assert window.steps.tolist() == expected
    assert window.raw.tolist() == [float(k) for k in expected]


def test_a_window_reaching_past_what_was_recorded_is_refused_and_one_inside_it_is_read():
    run = synthetic_run(10, DT)
    with pytest.raises(NotRecorded, match="reaches outside it"):
        run["probe:p"].speed.between(s(0.0), s(0.011))
    assert len(run["probe:p"].speed.between(s(0.0), s(0.010))) == 10
    with pytest.raises(ValueError, match="ends after it starts"):
        run["probe:p"].speed.between(s(0.005), s(0.005))
    with pytest.raises(ValueError, match="no sample"):
        run["probe:p"].speed.between(s(0.0051), s(0.0059)).mean()


# -- units ----------------------------------------------------------------------------------------------------------


def test_to_reads_a_series_and_a_reading_in_a_unit_and_refuses_one_of_another_dimension(bead):
    built, swim = bead
    result = built.run(s(0.01), record=[swim.speed])
    speed = result[swim].speed
    assert np.allclose(speed.to(mm_per_s), speed.raw.astype(np.float64) * 1e3)
    assert result[swim].mean_speed.to(mm_per_s) == pytest.approx(result[swim].mean_speed.value * 1e3)
    with pytest.raises(UnitError, match=r"cannot read a quantity of velocity \(m/s\).* in mm, a unit of length"):
        speed.to(mm)
    with pytest.raises(UnitError, match="in mm, a unit of length"):
        result[swim].mean_speed.to(mm)
    assert result[swim].distance.to(mm) == pytest.approx(result[swim].distance.value * 1e3)


# -- flags and exports ----------------------------------------------------------------------------------------------


def mat_flag_problems(result: RunResult, folder: Path, flag: str) -> list[str]:
    """The .mat file, read back by scipy's ``loadmat``: the flag among the file's flags with the steps it was raised
    at, and in the flags of every output at the end and every series recorded."""
    held = loadmat(result.to_mat(folder / "result.mat"), simplify_cells=True)
    found: list[str] = []
    entry = next((v for v in held.get("flags", {}).values() if v["path"] == flag), None)
    if entry is None or not np.size(entry.get("raised_steps", entry["was_raised"] == "true")):
        found.append("the .mat's flags")
    windows = [output for name in result.observations for output in held[name_in_mat(name)].values()
               if "flags_text" in output]
    if not windows or not all(flag in output.get("flags", {}).get(name_in_mat(flag), {}).get("path", "")
                              for output in windows):
        found.append("the .mat's outputs")
    return found


def name_in_mat(name: str) -> str:
    from mime.results._mat import matlab_name

    return matlab_name(name)


def flag_problems(result: RunResult, folder: Path, flag: str) -> list[str]:
    """Every export or printout of ``result`` that does not carry the flag ``flag``, raised in its run: the table's
    flags column, the CSV, the NPZ's flags and manifest, the folder's ``outputs.json``, and the printouts of the run,
    of a reading at the end and of a series."""
    found: list[str] = []
    table = result.table()
    if not any(flag in row for row in table["flags"]):
        found.append("the table's flags column")
    with result.to_csv(folder / "result.csv").open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if rows[0][-1] != "flags" or not any(flag in row[-1] for row in rows[1:]):
        found.append("the CSV")
    with np.load(result.to_npz(folder / "result.npz"), allow_pickle=False) as held:
        manifest = json.loads(str(held["manifest"]))
        if f"flags/{flag}" not in held.files or not held[f"flags/{flag}"].any():
            found.append("the NPZ's flags")
    written = folder / "run"
    written.mkdir(exist_ok=True)
    np.savez(written / RUN_OUTPUTS, allow_pickle=False, **result.arrays())
    (written / RUN_MANIFEST).write_text(json.dumps(result.manifest()), encoding="utf-8")
    on_disk = json.loads((written / RUN_MANIFEST).read_text(encoding="utf-8"))
    RunResult.read(written)  # the folder written is one that RunResult.read reads back
    for where, source in (("the NPZ's manifest", manifest), ("outputs.json", on_disk)):
        if flag not in json.dumps(source["run_flags"]):
            found.append(where)
        at_end = [o for held in source["observations"] for o in held["outputs"] if o["kind"] == "at_end"]
        if not at_end or not all(flag in json.dumps(o["flags"]) for o in at_end):
            found.append(f"{where}' outputs")
    found += mat_flag_problems(result, folder, flag)
    readings = [result[o].layout.outputs for o in result.observations]
    at_end = [result[o.observation][o.name] for outputs in readings for o in outputs if o.kind.value == "at_end"]
    for what, text in (("the run's printout", str(result)), ("a reading's printout", str(at_end[0])),
                       ("a series' printout", str(result.series()[0]))):
        if flag not in text:
            found.append(what)
    return found


def manifest_without_output_flags(manifest):
    """``RunResult.manifest`` with the flags of each output at the end emptied: the drop that only a look at the
    outputs' own entries sees."""

    def dropped(self):
        found = manifest(self)
        for held in found["observations"]:
            for output in held["outputs"]:
                if "flags" in output:
                    output["flags"] = []
        return found

    return dropped


def test_a_flag_raised_in_a_run_is_in_every_export_and_printout_and_a_dropped_one_is_caught(tmp_path, monkeypatch):
    raised = np.zeros(30, dtype=bool)
    raised[[3, 4, 17]] = True
    run = synthetic_run(30, DT, raised=raised)
    assert flag_problems(run, tmp_path, FLAG.path) == []
    read = run.flags[0]
    assert (read.raised, read.steps, read.first, read.last) == (3, 30, pytest.approx(0.004), pytest.approx(0.018))
    assert run["probe:p"].speed.between(s(0.0055)).flags[0].raised == 1
    assert run["probe:p"].speed.between(s(0.0055), s(0.017)).flags == (), "a flag not raised in a window is not shown"
    import mime.results._result as run_module
    import mime.results._series as series

    monkeypatch.setattr(series.FlagSource, "raised_in_spans", lambda self, steps, span: [""] * len(steps))
    assert flag_problems(run, tmp_path, FLAG.path) == ["the table's flags column", "the CSV"]
    monkeypatch.undo()
    monkeypatch.setattr(series.FlagSource, "over", lambda self, start, end: ())
    assert set(flag_problems(run, tmp_path, FLAG.path)) == {
        "the NPZ's manifest", "outputs.json", "the NPZ's manifest' outputs", "outputs.json' outputs",
        "the .mat's outputs", "the run's printout", "a reading's printout", "a series' printout"}
    monkeypatch.undo()
    monkeypatch.setattr(run_module.RunResult, "manifest", manifest_without_output_flags(run_module.RunResult.manifest))
    assert set(flag_problems(run, tmp_path, FLAG.path)) == {"the NPZ's manifest' outputs", "outputs.json' outputs"}
    monkeypatch.undo()
    import mime.results._mat as mat_module

    monkeypatch.setattr(mat_module, "_flag_variable", lambda result, names: {})
    assert flag_problems(run, tmp_path, FLAG.path) == ["the .mat's flags"]
    monkeypatch.undo()
    monkeypatch.setattr(mat_module, "_flags_of", lambda reads, names: {})
    assert flag_problems(run, tmp_path, FLAG.path) == ["the .mat's outputs"]
    monkeypatch.undo()
    assert flag_problems(run, tmp_path, FLAG.path) == []


def test_a_tables_flags_hold_the_flags_raised_between_its_samples_when_every_spaces_them(tmp_path, monkeypatch):
    """Step 5 lies between the samples at steps 3 and 6: the row at step 6 stands for steps 4 to 6, and carries it. A
    span of one step (the planted fault) leaves it out of the table and of the CSV."""
    raised = np.zeros(12, dtype=bool)
    raised[4] = True
    run = synthetic_run(12, DT, raised=raised, every=3)
    assert run.table()["flags"].tolist() == ["", FLAG.path, "", ""]
    import mime.results._series as series

    plain = series.FlagSource.raised_in_spans
    monkeypatch.setattr(series.FlagSource, "raised_in_spans", lambda self, steps, span: plain(self, steps, 1))
    assert run.table()["flags"].tolist() == ["", "", "", ""]
    assert flag_problems(run, tmp_path, FLAG.path) == ["the table's flags column", "the CSV"]


def test_a_flag_in_a_run_that_recorded_nothing_says_so_and_never_blocks_a_read():
    raised = np.zeros(8, dtype=bool)
    raised[-1] = True
    run = synthetic_run(8, DT, raised=raised, recorded=False)
    mean = run["probe:p"].mean
    assert mean.value == pytest.approx(4.5)
    assert ("law:p.fired: unknown, not recorded per step; at the run's last step it was raised, and before it is "
            "unknown") in mean.text()
    assert run.flags[0].was_raised is True


def test_a_flag_clear_at_the_last_step_of_a_run_that_recorded_nothing_is_unknown_and_never_not_raised():
    """The run's last step is all it knows, and a clamp that fired earlier is clear there: the text says unknown, and
    the manifest's ``was_raised`` is ``null`` with ``raised_at_last_step`` beside it. A run that recorded every step
    and saw nothing says not raised: the control, which tells the two apart."""
    unrecorded = synthetic_run(8, DT, raised=np.zeros(8, dtype=bool), recorded=False)
    (read,) = unrecorded.flags
    assert read.was_raised is None and read.at_last is False
    assert "unknown" in read.text() and "not raised" in read.text() and "before it is unknown" in read.text()
    manifest = unrecorded.manifest()
    assert manifest["run_flags"][0]["was_raised"] is None and manifest["run_flags"][0]["raised_at_last_step"] is False
    assert manifest["flags"][0]["was_raised"] is None and manifest["flags"][0]["raised_at_last_step"] is False
    assert all(output["flags"] == [] or output["flags"][0]["was_raised"] is None
               for held in manifest["observations"] for output in held["outputs"] if output["kind"] == "at_end")
    recorded = synthetic_run(8, DT, raised=np.zeros(8, dtype=bool))
    assert recorded.flags == (), "a flag never raised, over every step, is not shown"
    assert recorded.manifest()["flags"][0]["was_raised"] is False
    caught = synthetic_run(8, DT, raised=np.eye(8, dtype=bool)[2], recorded=False)
    assert caught.flags[0].was_raised is None, "raised at step 3 and clear at step 8: unknown, not raised"


def test_the_table_is_a_mapping_of_named_columns_with_units_that_needs_no_dataframe_library(tmp_path):
    run = synthetic_run(12, DT, every=3)
    table = run.table()
    assert isinstance(table, dict) and list(table) == [
        "time [s]", "probe:p.speed [m/s]", "probe:p.where[0] [m]", "probe:p.where[1] [m]", "probe:p.where[2] [m]",
        "flags"]
    assert table.units["probe:p.speed [m/s]"] == "m/s" and table["time [s]"] == pytest.approx([0.003, 0.006, 0.009,
                                                                                              0.012])
    assert all(len(column) == 4 for column in table.values())
    with run.to_csv(tmp_path / "run.csv").open(encoding="utf-8") as file:
        header, *rows = list(csv.reader(file))
    assert header == list(table) and [float(row[1]) for row in rows] == [3.0, 6.0, 9.0, 12.0]
    sources = Path(mime.__file__).parent / "results"
    imported = {alias.name.split(".")[0] for path in sources.glob("*.py")
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for path in sources.glob("*.py")
                 for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.ImportFrom)}
    assert "numpy" in imported and not imported & {"pandas", "polars"}
    importing_scipy = {path.name for path in sources.glob("*.py")
                       for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                       if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scipy"))
                       or (isinstance(node, ast.Import) and any(a.name.startswith("scipy") for a in node.names))}
    assert importing_scipy == {"_mat.py"}, "scipy is imported for the .mat file alone"


def test_a_table_of_series_sampled_at_other_times_is_refused_naming_them(bead):
    built, swim = bead
    result = built.run(s(0.004), record=[swim.speed])
    other = synthetic_run(4, DT, every=2)
    from mime.results._series import series_table

    with pytest.raises(ValueError, match="sampled at other times"):
        series_table([result[swim].speed, other["probe:p"].speed])
    with pytest.raises(NotRecorded, match="recorded no series"):
        built.run(s(0.004)).table()


# -- reading a run back ---------------------------------------------------------------------------------------------


def same_result(a: RunResult, b: RunResult) -> list[str]:
    """Every accessor on which two results differ."""
    found = []
    if (a.steps, a.start_step, a.every, a.recorded, a.observations) != (
            b.steps, b.start_step, b.every, b.recorded, b.observations):
        found.append("their steps, start, spacing, series or observations")
    for id in a.observations:
        for output in a[id].layout.outputs:
            if output.kind.value == "per_step" and output.path not in a.recorded:
                continue
            x, y = a[id][output.name], b[id][output.name]
            if isinstance(x, Series) and (not isinstance(y, Series) or not np.array_equal(x.raw, y.raw)
                                          or not np.array_equal(x.steps, y.steps)):
                found.append(output.path)
            if isinstance(x, Reading) and (not isinstance(y, Reading) or x.text() != y.text()):
                found.append(output.path)
    if [r.text() for r in a.flags] != [r.text() for r in b.flags]:
        found.append("their flags")
    if a.text() != b.text():
        found.append("their printouts")
    return found


def test_run_to_writes_outputs_json_beside_outputs_npz_and_read_gives_the_same_result(bead, tmp_path):
    built, swim = bead
    written = built.run_to(tmp_path / "run", s(0.012), record=[swim.speed], every=ms(3))
    folder = tmp_path / "run"
    assert {path.name for path in folder.iterdir()} >= {RUN_OUTPUTS, RUN_MANIFEST, RUN_RECORD, RUN_EXPERIMENT}
    manifest = json.loads((folder / RUN_MANIFEST).read_text())
    (observation,) = manifest["observations"]
    outputs = {o["name"]: o for o in observation["outputs"]}
    assert outputs["mean_speed"]["unit"] == "m/s" and outputs["mean_speed"]["window"] == [0.0105, 0.012]
    assert outputs["mean_speed"]["value"] == pytest.approx(written.result[swim].mean_speed.value)
    with np.load(folder / RUN_OUTPUTS, allow_pickle=False) as arrays:
        assert arrays[outputs["speed"]["values"]].shape == (4,)
        assert arrays[outputs["mean_speed"]["array"]] == pytest.approx(outputs["mean_speed"]["value"])
    again = RunResult.read(folder)
    assert same_result(written.result, again) == []
    assert again[swim].speed.between(s(0.006)).mean().value == pytest.approx(
        written.result[swim].speed.between(s(0.006)).mean().value)
    assert json.loads((folder / RUN_RECORD).read_text())["recorded"] == ["swim_speed:bead.speed"]


def test_a_run_on_a_modal_volume_reads_from_its_mount_as_from_a_local_folder(bead, tmp_path):
    """A stand-in for the ``mime-runs`` Volume, mounted as the runner mounts it: ``<mount>/<experiment>/<run id>``,
    written by ``mime run --out`` there, beside the runner's ``env-check.json``; read where it is mounted, or after
    ``modal volume get`` copies it out."""
    built, swim = bead
    mount = tmp_path / "mime-runs"
    run = mount / "falling_bead" / "20261031T020000Z-0123abcd"
    run.mkdir(parents=True)
    (run / "env-check.json").write_text("{}\n")
    written = built.run_to(run, s(0.01), record=[swim.speed])
    fetched = tmp_path / "fetched"
    import shutil

    shutil.copytree(run, fetched)
    assert same_result(written.result, RunResult.read(run)) == []
    assert same_result(written.result, RunResult.read(fetched)) == []


def test_an_npz_export_reads_back_and_numpy_reads_it_alone(tmp_path):
    raised = np.zeros(9, dtype=bool)
    raised[4] = True
    run = synthetic_run(9, DT, raised=raised, every=3)
    path = run.to_npz(tmp_path / "run.npz")
    assert same_result(run, RunResult.read(path)) == []
    with np.load(path, allow_pickle=False) as held:
        manifest = json.loads(str(held["manifest"]))
        (outputs,) = [o["outputs"] for o in manifest["observations"]]
        speed = next(o for o in outputs if o["name"] == "speed")
        assert held[speed["values"]].tolist() == [3.0, 6.0, 9.0] and speed["unit"] == "m/s"
        assert held[f"flags/{FLAG.path}"].tolist() == raised.astype(int).tolist()


def test_a_folder_without_a_manifest_or_of_another_version_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="is not a run folder"):
        RunResult.read(tmp_path)
    run = synthetic_run(3, DT)
    path = run.to_npz(tmp_path / "run.npz")
    from mime.results._result import _Npz

    manifest = {**run.manifest(), "version": 99}
    with pytest.raises(ValueError, match="version 99"):
        RunResult.from_manifest(manifest, _Npz(path))


def test_a_vector_output_reads_as_a_series_of_vectors_with_its_components_in_the_table():
    run = synthetic_run(5, DT)
    where = run["probe:p"].where
    assert where.values.value.shape == (5, 3) and where.output == WHERE
    assert where.mean().value == pytest.approx((3.0, 6.0, 9.0))
    assert run["probe:p"]["speed"].output == SPEED
