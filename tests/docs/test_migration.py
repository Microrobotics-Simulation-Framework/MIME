"""The 0.2 to 0.3 migration table (``mime.docs._migration``): every name on the list frozen from the tag ``pre-0.3`` has
one row, a row names nothing else, and each row's 0.3 entry and rung are registered.

The real table passes. Each control doctors a copy of the record or of the frozen list and runs the real check on it;
the inputs are deterministic, so each fails on every run.
"""

import shutil
from pathlib import Path

import pytest

from mime.docs._export import load_namespaces
from mime.docs._migration import NAMES, RECORD, frozen_names, migration_report
from mime.registry import registered

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def records():
    load_namespaces()
    return registered()


def doctored(tmp_path: Path, old: str = "", new: str = "", file: Path = RECORD) -> Path:
    for path in (RECORD, NAMES):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    if old:
        text = (tmp_path / file).read_text(encoding="utf-8")
        assert text.count(old) == 1, old
        (tmp_path / file).write_text(text.replace(old, new), encoding="utf-8")
    return tmp_path


def test_the_table_names_every_frozen_name_once_and_each_target_is_registered(records):
    rows, problems = migration_report(ROOT, records)
    assert problems == []
    names = frozen_names((ROOT / NAMES).read_text(encoding="utf-8"))
    assert sorted(row.name for row in rows) == sorted(names)


def test_control_a_typo_in_a_name_fails(tmp_path, records):
    _, problems = migration_report(doctored(tmp_path, "- name: RigidBodyNode\n", "- name: RigidBodyNod\n"), records)
    assert any("'RigidBodyNod' is not a name of the 0.2 line" in p for p in problems), problems
    assert any("RigidBodyNode (mime.nodes.robot.rigid_body) has no row" in p for p in problems), problems


def test_control_a_target_no_entry_answers_to_and_a_rung_it_lacks_fail(tmp_path, records):
    _, problems = migration_report(doctored(tmp_path, "    to: dynamics rigid_inertial\n",
                                            "    to: dynamics rigid_body\n"), records)
    assert any("'to: dynamics rigid_body' names no registered entry" in p for p in problems), problems
    _, problems = migration_report(doctored(tmp_path, "    rung: mrs\n", "    rung: lbm\n"), records)
    assert any("interaction drag has no rung 'lbm'" in p for p in problems), problems


def test_control_a_row_with_both_a_target_and_a_removal_or_neither_fails(tmp_path, records):
    _, problems = migration_report(doctored(tmp_path, "  - name: CSFFlowNode\n    removed: no 0.3 counterpart\n",
                                            "  - name: CSFFlowNode\n"), records)
    assert any("(CSFFlowNode): give either 'to'" in p for p in problems), problems


def test_control_a_name_given_two_rows_fails(tmp_path, records):
    _, problems = migration_report(doctored(tmp_path, "  - name: RobotArmNode\n", "  - name: RobotArmEffect\n"),
                                   records)
    assert any("RobotArmEffect has 2 rows" in p for p in problems), problems


def test_control_the_frozen_list_edited_fails_by_its_digest(tmp_path, records):
    _, problems = migration_report(doctored(tmp_path, "RigidBodyNode  mime.nodes.robot.rigid_body\n",
                                            "RigidBodyNode  mime.nodes.robot.rigid_body\nTypoNode  mime.nodes.x\n",
                                            NAMES), records)
    assert any("it is not the list frozen from the tag pre-0.3" in p for p in problems), problems
