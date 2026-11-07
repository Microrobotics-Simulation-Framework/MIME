"""A study written to its folder and read back: gate 1 of S2c-1. Script A over the field's frequency and the run's step
(so the members' series are of two lengths, ragged), written by ``run_to`` and read by ``StudyResult.read``: every
coordinate, reading, series, flag and final-state field equal to the in-memory result's, ``np.array_equal``, and the
folder's layout and manifest closed. Controls: two members' coordinates exchanged in ``study.json`` must be seen (every
pair of members with different coordinates is tried, and the detection rate asserted), a stray file (one named like a
temporary, a group folder past the last, an ``experiment/`` in group 0), a key added to the manifest, a newer format (and a
version that is ``true``), a point in two members and a member in no group are refused."""

from __future__ import annotations

import json
import shutil
from itertools import combinations

import numpy as np
import pytest
from folder_rigs import SPEED, same_study, script_a_study, toy_study, tree_bytes

from mime.build._refusals import ArgumentRefused
from mime.results import StudyResult
from mime.results._study_folder import STUDY_FILE, STUDY_FORMAT, STUDY_VERSION
from mime.saved import read_folder
from mime.saved._versions import CURRENT


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    study = script_a_study()
    memory = study.run()
    folder = tmp_path_factory.mktemp("study") / "folder"
    returned = study.run_to(folder)
    return study, memory, returned, folder


def test_the_folder_read_back_equals_the_result_in_memory_exactly(written):
    _, memory, returned, folder = written
    assert len(memory) == 4 and len(set(memory.groups)) == 2
    assert same_study(memory, returned) == []
    assert same_study(memory, StudyResult.read(folder)) == []
    # Ragged: the other step has twice the samples, so the series could not have been stacked as one array.
    assert {len(memory.member(i)["swim_speed:umr"].speed.raw) for i in range(4)} == {40, 80}
    assert returned.missing == () and "PARTIAL" not in returned.text()


def test_the_manifest_is_the_results_own_and_readable_without_mime(written):
    _, memory, _, folder = written
    document = json.loads((folder / STUDY_FILE).read_text())
    assert (document["format"], document["version"]) == (STUDY_FORMAT, STUDY_VERSION)
    assert [m["status"] for m in document["members"]] == ["done"] * 4
    assert [m["point"] for m in document["members"]] == list(range(4))
    assert [axis["path"] for axis in document["axes"]] == ["field.commands.frequency", "dt"]
    assert document["axes"][0]["unit"] == "Hz" and document["points"][1]["values"][0] == {"value": 10.0, "unit": "Hz"}
    with np.load(folder / "groups" / "0" / "outputs.npz") as held:
        assert "m0/manifest" in held.files and any(key.startswith("m0/series/") for key in held.files)
    # The result in memory knows no folder, so no base hash (and so no id or content key) and no group experiments; all
    # the rest of its manifest is the folder's.
    held = json.loads(json.dumps(memory.manifest()))
    assert held["experiment"]["input_hash"] is None and document["experiment"]["input_hash"].startswith("sha256:")
    for key in ("axes", "points", "run", "dropped", "seed", "drop_refused", "lineage"):
        assert document[key] == held[key], key
    assert [(m["point"], m["group"], m["overrides"]) for m in document["members"]] == \
        [(m["point"], m["group"], m["overrides"]) for m in held["members"]]
    assert [(g["key"], g["members"], g["batches"], g["compiles"]) for g in document["groups"]] == \
        [(g["key"], g["members"], g["batches"], g["compiles"]) for g in held["groups"]]


def test_the_base_is_the_experiment_saved_byte_for_byte_and_the_saved_form_is_the_tips(written, tmp_path):
    study, _, _, folder = written
    from script_a import reproduction_arm

    reproduction_arm().save(tmp_path / "plain")
    assert tree_bytes(folder / "experiment") == tree_bytes(tmp_path / "plain")
    assert read_folder(folder / "experiment").version == CURRENT
    assert (folder / "groups" / "1" / "experiment").is_dir() and not (folder / "groups" / "0" / "experiment").exists()


def _exchange(folder, a: int, b: int) -> None:
    path = folder / STUDY_FILE
    document = json.loads(path.read_text())
    points = {row["index"]: row for row in document["points"]}
    points[a]["values"], points[b]["values"] = points[b]["values"], points[a]["values"]
    path.write_text(json.dumps(document))


def test_two_members_coordinates_exchanged_in_the_manifest_are_always_seen(written, tmp_path):
    """The control, over every pair of members whose coordinates differ: the read-back must differ from the result in
    memory in a coordinate. Its detection rate is 1.0, asserted."""
    _, memory, _, folder = written
    pairs = [(a, b) for a, b in combinations(range(4), 2)
             if not np.array_equal(memory.coordinates["field.commands.frequency"].value[[a, b]],  # type: ignore[union-attr]
                                   memory.coordinates["field.commands.frequency"].value[[b, a]])  # type: ignore[union-attr]
             or not np.array_equal(memory.coordinates["dt"].value[[a, b]], memory.coordinates["dt"].value[[b, a]])]  # type: ignore[union-attr]
    assert len(pairs) == 6
    seen = 0
    for a, b in pairs:
        copy = tmp_path / f"copy_{a}_{b}"
        shutil.copytree(folder, copy)
        _exchange(copy, a, b)
        seen += any(each.startswith("coordinate") for each in same_study(memory, StudyResult.read(copy)))
    print("exchanged coordinates seen:", seen, "of", len(pairs))
    assert seen == len(pairs)


def test_a_stray_file_a_key_added_and_a_newer_format_are_refused(written, tmp_path):
    _, _, _, folder = written
    for name, damage, code in (
            ("stray", lambda c: (c / "notes.txt").write_text("x"), "study.stray-file"),
            ("stray_in_group", lambda c: (c / "groups" / "0" / "extra.npz").write_bytes(b""), "study.stray-file"),
            ("key", lambda c: _edit(c, lambda d: d.update(extra=1)), "study.manifest"),
            ("member_key", lambda c: _edit(c, lambda d: d["members"][0].update(extra=1)), "study.manifest"),
            ("newer", lambda c: _edit(c, lambda d: d.update(version=STUDY_VERSION + 1)), "study.format"),
            ("other_format", lambda c: _edit(c, lambda d: d.update(format="mime-run-outputs")), "study.format"),
            ("version_true", lambda c: _edit(c, lambda d: d.update(version=True)), "study.format"),
            ("named_like_a_temporary", lambda c: (c / "notes.tmp").write_text("x"), "study.stray-file"),
            ("group_past_the_last", lambda c: (c / "groups" / "7").mkdir(), "study.stray-file"),
            ("experiment_in_group_0", lambda c: (c / "groups" / "0" / "experiment").mkdir(), "study.stray-file"),
            ("point_twice", lambda c: _edit(c, lambda d: d["members"][1].update(point=d["members"][0]["point"])),
             "study.manifest"),
            ("group_out_of_range", lambda c: _edit(c, lambda d: d["members"][0].update(group=9)), "study.manifest")):
        copy = tmp_path / name
        shutil.copytree(folder, copy)
        damage(copy)
        with pytest.raises(ArgumentRefused) as refused:
            StudyResult.read(copy)
        assert refused.value.codes() == [code], (name, refused.value.codes())
    with pytest.raises(ArgumentRefused) as refused:
        StudyResult.read(tmp_path)
    assert "study.not-a-study-folder" in refused.value.codes()


def _edit(folder, change) -> None:
    path = folder / STUDY_FILE
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_a_folder_that_holds_another_study_or_a_file_is_refused_by_run_to(tmp_path):
    from mime.build import BuildRefused

    study = toy_study(steps=(), stiffnesses=(1.1, 1.7))
    out = tmp_path / "study"
    study.run_to(out)
    with pytest.raises(BuildRefused) as other:
        toy_study(steps=(), stiffnesses=(1.1, 1.9)).run_to(out)
    assert other.value.codes() == ["study.other-study"] and "never written over another" in other.value.refusals[0].action
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "x.txt").write_text("x")
    with pytest.raises(BuildRefused) as plain:
        study.run_to(tmp_path / "plain")
    assert plain.value.codes() == ["study.folder-not-empty"]


def test_a_value_the_folder_cannot_hold_is_refused_naming_the_axis_and_a_quantity_is_held():
    from mime.results._study_folder import decode_value, encode_value
    from mime.units import Hz, mm

    with pytest.raises(ArgumentRefused) as refused:
        encode_value(object(), "field.commands.frequency")
    assert refused.value.codes() == ["study.unsavable-value"] and "field.commands.frequency" in refused.value.refusals[0].message
    for value in (Hz(12.5), mm((0.0, 0.5, -1.0)), 3, "float64", True):
        back = decode_value(json.loads(json.dumps(encode_value(value, "x"))))
        assert (np.array_equal(np.asarray(back.value), np.asarray(value.value)) and back.dimension == value.dimension) \
            if hasattr(value, "value") else back == value
