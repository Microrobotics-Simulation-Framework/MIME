"""Saved form 13: the FVM's setting ``solve_report_rows``, which form 12 and earlier named ``report_rows`` (design row
#303), in a medium's flow settings, in the flow's resolved entry and its default, and among the flow node's
parameters. Nothing else moves. A folder of an earlier form is read with the old name as the new, and the rebuilt
document, resolved record and derived graph are written back with the old name for its lock's hashes.

The folders that name the setting by its old name, each the round-trip case ``periodic_tube_flow`` with
``FVM(report_rows=64)``, so its document names it, float64, four usable CPUs (``taskset -c 0-3``), one BLAS thread,
saved by ``exp.build().save(folder)``, pinned by its sha256:

- ``form12/flow_report_rows``: written by MIME at the release branch's tip when this step was renumbered from 12 to 13
  (970b3f6, form 12);
- ``form11/flow_report_rows``: written by MIME at the last commit of form 11 (46fcc79), read through form 12's step;
- ``form10/periodic_flow`` (``test_form11.py``) names it too, in its lock alone.

``form12/chain_screw`` is the current folder as form 12 wrote it (no flow: the step leaves it as it was).

**The controls.** With the resolved record's write-back removed, the form-12 flow folder is refused as
``load.input-hash`` on every run (a hash: the same input, the same refusal); with the document's read of the old name
removed, its flow is refused as a setting the FVM does not take (``load.setting-unknown``); with the derived graph's
write-back removed, the folders' own graphs, named as form 13 names them, are not written back as their locks hold them
(a check of keys, where a load's comparison of a rebuilt graph off the lock's environment reads numbers). Version 12's
reader refuses a folder of version 13 as ``load.newer-form``, the document and the lock each once, and the form left
unstepped is read by it with none."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from test_form11 import FLOW10, codes_of, reader_of, restated

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused
from mime.saved import _versions
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, NEW_ROWS, OLD_ROWS, ROWS_RENAMED

SAVED = Path(__file__).parent
FLOW12 = SAVED / "form12" / "flow_report_rows"
FLOW11 = SAVED / "form11" / "flow_report_rows"
CHAIN12 = SAVED / "form12" / "chain_screw"
PINNED = {
    FLOW12: {"experiment.mime.yaml": "14edf31fd2423c0a8f802d9e858255ab2e18f88afaba8010c7afe609d6a2b68f",
             "mime.lock": "0bbbfb1ebe05a3953318368ea9c4b5b3f402ab6efe78425c3a6e92f15858ee08",
             "reference.npz": "92e9160facde462c378e392e004ccbbb61c3afff768cb0c5e7184ca318422c75",
             "scene/scene.usda": "ada6dcbfd8ce8e660a83aae1ed016895ac3002524a281a3515ec0d646a665cf0"},
    FLOW11: {"experiment.mime.yaml": "118e6215d0b2a919cff50135845a77f49f1b77c68dd179dd019bf161ae5e752f",
             "mime.lock": "10f7cf8524ed0d8bb48a026c1784bc34403d80a07e0ce827fb54722c13a97a9e",
             "reference.npz": "92e9160facde462c378e392e004ccbbb61c3afff768cb0c5e7184ca318422c75",
             "scene/scene.usda": "ada6dcbfd8ce8e660a83aae1ed016895ac3002524a281a3515ec0d646a665cf0"},
}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM12_READS = 12


def _keys(data: object, name: str, path: str = "") -> list[str]:
    """Every key ``name`` in ``data``'s dictionaries, by its path; and every key ending ``.name``."""
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == name or str(key).endswith(f".{name}"):
                found.append(f"{path}{key}")
            found += _keys(value, name, f"{path}{key}.")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found += _keys(value, name, f"{path}{index}.")
    return found


def test_version_thirteen_writes_solve_report_rows_and_reads_it_back(tmp_path: Path):
    from cases import CASES

    assert ROWS_RENAMED == 13 < CURRENT
    assert DOCUMENT_SCHEMAS[ROWS_RENAMED] == "mime.document/13" and LOCK_SCHEMAS[ROWS_RENAMED] == "mime.lock/13"
    experiment = next(case for case in CASES if case.name == "periodic_tube_flow").make()
    lock = experiment.save(tmp_path / "flow")
    assert lock["schema"] == LOCK_SCHEMAS[CURRENT]
    held = json.loads((tmp_path / "flow" / LOCK_FILE).read_text(encoding="utf-8"))
    assert _keys(held, OLD_ROWS) == []
    assert sorted(_keys(held, NEW_ROWS)) == sorted([
        "derived.physics.nodes.1.params.solve_report_rows", "resolved.entries.water.flow.settings.solve_report_rows",
        "resolved.profile.defaults.water.flow.solve_report_rows"])
    loaded = Experiment.load(tmp_path / "flow")
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()


FLOWS = pytest.mark.parametrize("folder", [FLOW12, FLOW11], ids=["form12", "form11"])


@FLOWS
def test_each_flow_folder_is_the_folder_its_commit_wrote_byte_for_byte(folder: Path):
    found = {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(folder.rglob("*")) if path.is_file()}
    assert found == PINNED[folder]
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert document["things"]["water"]["fields"]["flow"]["settings"][OLD_ROWS] == 64


@pytest.mark.parametrize("folder", [FLOW12, FLOW11, FLOW10], ids=["form12", "form11", "form10"])
def test_a_folder_of_an_earlier_form_loads_rebuilds_and_verifies(tmp_path: Path, folder: Path):
    copy = Path(shutil.copytree(folder, tmp_path / folder.name))
    loaded = Experiment.load(copy)
    assert loaded.load_report is not None
    version = int(folder.parent.name.removeprefix("form"))
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/{version}, read as version {CURRENT}", "input hash: reproduced exactly"]
    assert loaded.build().verify().reproduced


def test_the_step_leaves_a_folder_without_a_flow_as_it_was():
    """The current folder as form 12 wrote it (a chain, no flow): its document reads as form 12's but for its schema,
    and its document, resolved record and derived graph written back into form 12 are what its lock holds."""
    document = yaml.safe_load((CHAIN12 / DOCUMENT_FILE).read_text(encoding="utf-8"))
    lock = json.loads((CHAIN12 / LOCK_FILE).read_text(encoding="utf-8"))
    assert (document["schema"], lock["schema"]) == ("mime.document/12", "mime.lock/12")
    upgraded, version, _ = _versions.upgraded_document(document)
    assert version == 12 and {**upgraded, "schema": document["schema"]} == document
    assert _versions.document_in_form(upgraded, 12) == document
    assert _versions.resolved_in_form(lock["resolved"], 12) == lock["resolved"]
    assert _versions.physics_in_form(lock["derived"]["physics"], 12) == lock["derived"]["physics"]


@FLOWS
def test_a_flow_folders_derived_graph_is_written_back_as_its_lock_holds_it(folder: Path):
    """The derived graph's write-back alone, on a flow folder's own lock, in any environment (a load compares a rebuilt
    graph with the lock's by its numbers, which an environment unlike the lock's does not reproduce, so it cannot see a
    key left unrenamed): the lock's graph named as this form names it, the flow node's parameter by its new name, is
    written back into the folder's form as the lock holds it."""
    physics = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))["derived"]["physics"]

    def named_now(node: dict[str, object]) -> dict[str, object]:
        params = node.get("params")
        if not isinstance(params, dict) or OLD_ROWS not in params:
            return node
        return {**node, "params": {NEW_ROWS if key == OLD_ROWS else key: value for key, value in params.items()}}

    current = {**physics, "nodes": [named_now(node) for node in physics["nodes"]]}
    assert _keys(current, NEW_ROWS) == ["nodes.1.params.solve_report_rows"] and _keys(current, OLD_ROWS) == []
    assert _versions.physics_in_form(current, int(folder.parent.name.removeprefix("form"))) == physics


def test_the_form_twelve_flow_folder_keeps_its_rows_under_the_new_name(tmp_path: Path):
    loaded = Experiment.load(Path(shutil.copytree(FLOW12, tmp_path / "flow")))
    node = loaded.build().graph.get_node("water.flow")  # type: ignore[attr-defined]
    assert node.constants()[NEW_ROWS] == 64 and OLD_ROWS not in node.constants()


def test_control_the_form_twelve_flow_folder_without_the_write_back_is_refused_by_its_input_hash(tmp_path: Path):
    folder = Path(shutil.copytree(FLOW12, tmp_path / "flow"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "_rows_in_form", lambda resolved: resolved)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_control_the_form_twelve_flow_folder_without_the_read_of_the_old_name_is_refused(tmp_path: Path):
    folder = Path(shutil.copytree(FLOW12, tmp_path / "flow"))
    original = _versions._with_flow_setting  # noqa: SLF001

    def read_unrenamed(medium, was, now):
        return medium if (was, now) == (OLD_ROWS, NEW_ROWS) else original(medium, was, now)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "_with_flow_setting", read_unrenamed)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(folder)
    assert refused.value.codes() == ["load.setting-unknown"] and f"no setting '{OLD_ROWS}'" in str(refused.value)


@pytest.fixture(scope="module")
def current_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from mime.testing.toys import full_toy

    root = tmp_path_factory.mktemp("form13")
    full_toy().save(root / "full_toy")
    return root / "full_toy"


def test_version_twelves_reader_refuses_a_folder_of_version_thirteen_as_newer_each_file_once(current_folder: Path):
    assert codes_of(current_folder) == [], "this MIME reads it"
    with reader_of(FORM12_READS):
        assert codes_of(current_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_twelves_reader_with_no_newer_form(current_folder: Path,
                                                                                              tmp_path: Path):
    unstepped = restated(current_folder, tmp_path / "unstepped", FORM12_READS)
    with reader_of(FORM12_READS):
        assert NEWER not in codes_of(unstepped)
