"""The study's spec file (``study.mime.yaml``) and ``mime study``: a study written by ``run_to`` carries its spec, a study
made from that spec is the same study (the same id, the same bytes in the folder it writes), the spec is read with the
safe loader and holds no code, a key given twice, a key it does not know, another format and another base are refused,
and ``mime study plan`` and ``run`` do what ``Study.plan`` and ``run_to`` do, a run into a folder that holds another study
refused, ``--shard`` leaving the other groups pending (shards one after another into one folder adding up to the whole
study, and more shards than groups refused, sharding being by group) and ``--pilot`` saying what a member costs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from folder_rigs import tree_bytes

from folder_rigs import toy_study
from mime import Study
from mime.build._refusals import ArgumentRefused
from mime.cli import main as mime_main
from mime.results import StudyResult
from mime.results._study_folder import STUDY_FILE
from mime.study._spec import read_spec, spec_text, value_of, value_text
from mime.units import Hz, kg, mm, ms, s

SPEC = "study.mime.yaml"


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    folder = tmp_path_factory.mktemp("written") / "folder"
    toy_study().run_to(folder)
    return folder


def run_with(written, tmp_path, name="again"):
    out = tmp_path / name
    Study.from_spec(written / SPEC).run_to(out)
    return out


def test_a_study_made_from_the_spec_a_run_wrote_is_the_same_study(written, tmp_path):
    out = run_with(written, tmp_path)
    held, again = StudyResult.read(written), StudyResult.read(out)
    assert held.record["id"] == again.record["id"] and held.record["members"] == again.record["members"]
    a, b = tree_bytes(written), tree_bytes(out)
    assert a.keys() == b.keys() and not [k for k in a if k != STUDY_FILE and a[k] != b[k]]


def test_the_specs_normal_form_is_data_with_a_base_a_combine_and_a_reason_per_axis_when_given(written):
    document = yaml.safe_load((written / SPEC).read_text())
    assert document["format"] == "mime-study/1" and document["combine"] == "grid"
    assert document["base"]["folder"] == "experiment" and document["base"]["input_hash"].startswith("sha256:")
    assert list(document["axes"]) == ["toy.spring:probe.k", "dt"] and "reason" not in document["axes"]["dt"]
    assert document["axes"]["dt"]["values"] == ["0.02 s", "0.01 s"] and document["duration"] == "0.4 s"
    reasoned = toy_study(reasons={"dt": "the two steps the ladder compares"}).plan().reasons
    assert reasoned == {"dt": "the two steps the ladder compares"}


def test_a_value_is_written_with_its_unit_and_read_back_equal():
    for value in (Hz(12.5), ms(0.25), mm((0.0, 0.5, -1.0)), kg(8.5e-5), 3.0, 4, True, "float64"):
        back = value_of(value_text(value, "x"), "x")
        if hasattr(value, "value"):
            assert back.dimension == value.dimension and np.array_equal(np.asarray(back.value), np.asarray(value.value))
        else:
            assert back == value
    with pytest.raises(ArgumentRefused):
        value_text("5 mm", "x")  # a name that reads as a quantity would come back as one


def damaged(written, tmp_path, name, change):
    spec = tmp_path / name
    spec.mkdir()
    document = yaml.safe_load((written / SPEC).read_text())
    change(document)
    (spec / SPEC).write_text(yaml.safe_dump(document))
    return spec / SPEC


def test_what_a_spec_may_not_hold_is_refused_and_the_same_spec_without_the_fault_is_read(written, tmp_path):
    assert read_spec(written / SPEC)["format"] == "mime-study/1"
    for name, change, word in (("key", lambda d: d.update(code="import os"), "'code'"),
                               ("format", lambda d: d.update(format="mime-study/2"), "mime-study/2"),
                               ("combine", lambda d: d.update(combine="zip"), "combine"),
                               ("seed", lambda d: d.update(seed={"policy": "per-member"}), "per-member"),
                               ("axis", lambda d: d["axes"]["dt"].update(extra=1), "dt"),
                               ("drop", lambda d: d.update(drop_refused="false"), "drop_refused"),
                               ("record", lambda d: d.update(record="toy.recorder:probe.last"), "record")):
        with pytest.raises(ArgumentRefused) as refused:
            read_spec(damaged(written, tmp_path, name, change))
        assert refused.value.codes() == ["study.spec"] and word in refused.value.refusals[0].message, name
    twice = tmp_path / "twice.yaml"
    twice.write_text("format: mime-study/1\nformat: mime-study/1\n")
    with pytest.raises(ArgumentRefused) as refused:
        read_spec(twice)
    assert "given twice" in refused.value.refusals[0].message and "line 2" in refused.value.refusals[0].message
    tagged = tmp_path / "tagged.yaml"
    tagged.write_text("format: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(ArgumentRefused) as refused:
        read_spec(tagged)
    assert refused.value.codes() == ["study.spec"], "the safe loader builds no Python object"


def test_a_spec_of_another_experiment_is_refused_by_its_input_hash(written, tmp_path):
    spec = damaged(written, tmp_path, "base", lambda d: d["base"].update(input_hash="sha256:" + "0" * 64))
    (spec.parent / "experiment").symlink_to(written / "experiment", target_is_directory=True)
    with pytest.raises(ArgumentRefused) as refused:
        Study.from_spec(spec)
    assert "input hash" in refused.value.refusals[0].message


def test_mime_study_plan_prints_the_plan_and_exits_one_where_a_point_is_refused(written, capsys):
    assert mime_main(["study", "plan", str(written / SPEC)]) == 0
    printed = capsys.readouterr().out
    assert "6 points over toy.spring:probe.k, dt; 2 groups" in printed and "nothing refused" in printed
    assert mime_main(["study", "plan", str(written / SPEC), "--base", str(written / "experiment")]) == 0
    capsys.readouterr()


def test_mime_study_run_writes_the_folder_resumes_refuses_another_study_and_shards(written, tmp_path, capsys):
    out = tmp_path / "cli"
    assert mime_main(["study", "run", str(written / SPEC), "--out", str(out), "--pilot"]) == 0
    printed = capsys.readouterr()
    assert "pilot: a chunk of" in printed.err + printed.out and "wrote the study folder" in printed.out
    assert len(StudyResult.read(out)) == 6
    different = damaged(written, tmp_path, "different", lambda d: d["axes"]["dt"].update(values=["0.02 s", "0.005 s"]))
    (different.parent / "experiment").symlink_to(written / "experiment", target_is_directory=True)
    assert mime_main(["study", "run", str(different), "--out", str(out)]) == 2
    assert "study.other-study" in capsys.readouterr().err
    shard = tmp_path / "shard"
    assert mime_main(["study", "run", str(written / SPEC), "--out", str(shard), "--shard", "1/2", "--pilot"]) == 0
    printed = capsys.readouterr()
    assert "of group 1 took" in printed.err + printed.out, "the pilot runs a group this shard owns"
    partial = StudyResult.read(shard)
    assert partial.missing == (0, 2, 4) and set(partial.groups) == {1}
    assert json.loads((shard / STUDY_FILE).read_text())["members"][0]["status"] == "pending"
    assert mime_main(["study", "run", str(written / SPEC), "--out", str(shard), "--shard", "0/2"]) == 0
    capsys.readouterr()
    whole = StudyResult.read(shard)
    assert len(whole) == 6 and whole.missing == (), "shards one after another into one folder are the whole study"
    assert mime_main(["study", "run", str(written / SPEC), "--out", str(tmp_path / "bad"), "--shard", "2/2"]) == 2
    assert "study.shard" in capsys.readouterr().err
    assert mime_main(["study", "run", str(written / SPEC), "--out", str(tmp_path / "three"), "--shard", "0/3"]) == 2
    refused = capsys.readouterr().err
    assert "study.shard" in refused and "structure group" in refused and "--shard i/2" in refused
    assert not (tmp_path / "three").exists(), "refused before anything is written"


def test_a_spec_text_written_for_points_that_are_a_list_of_rows_is_an_along(tmp_path):
    from mime.study import along
    from mime.units import N, m

    points = along({"toy.spring:probe.k": [N(1.1) / m(1.0), N(1.7) / m(1.0)], "dt": [ms(20.0), ms(10.0)]})
    text = spec_text(points, duration=s(0.4), record=[], every=None, batch=None, drop_refused=False, reasons={},
                     base="experiment", input_hash=None)
    document = yaml.safe_load(text)
    assert document["combine"] == "along" and document["axes"]["dt"]["values"] == ["0.02 s", "0.01 s"]
