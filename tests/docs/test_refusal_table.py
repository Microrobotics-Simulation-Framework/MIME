"""The refusal table: one row per code MIME constructs, with the classes it is refused in, generated from the scan of
the source and checked against it; the constructor's check that a MIME code is in it; and the golden numbers of the
scan, measured on the tree this table landed on.

The controls: a planted lab ``Problem`` code under a registered namespace is refused, not raised, and so is one
outside it (R1); a code planted in a ``(code, message, action)`` tuple, and one passed by a variable, are each found by
the scan and fail the table (R2); an uncatalogued MIME code, and a catalogued one in the wrong class, raise where they
are constructed (R3)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
from checkout import ROOT, copy_checkout, edit

from mime.build import REFUSAL_TABLE, BuildRefused, Refusal, RefusalClass, UncataloguedRefusal
from mime.docs import census
from mime.docs._check import refusal_check
from mime.registry import Plan, PlanContext, Problem, register_interaction
from mime.testing import installed_entries
from mime.testing.toys import ToySpring, coupled_toy
from mime.vocabulary import namespace_of, register_namespace

GOLDEN = json.loads((Path(__file__).parent / "golden_numbers.json").read_text(encoding="utf-8"))


def test_the_refusal_table_is_what_the_source_constructs():
    report = refusal_check(ROOT)
    assert report.problems == [], "\n".join(report.problems)


def test_the_scan_reproduces_its_golden_numbers():
    assert census(ROOT / "src/mime").__dict__ == GOLDEN["refusal_scan"]
    assert len(REFUSAL_TABLE) == GOLDEN["refusal_table"]


def test_the_registry_reproduces_its_golden_numbers():
    found: dict[str, int] = {}
    for _, name, _ in installed_entries():
        namespace = name.split(".", 1)[0] if name.split(".", 1)[0] in ("toy", "labx") else "mime"
        found[namespace] = found.get(namespace, 0) + 1
    assert found == GOLDEN["entries"]


# -- R1: a lab's Problem codes are refused, never raised -----------------------------------------------------------

register_namespace("docslab", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class FarSpring(ToySpring):
    """A lab's spring whose plan finds its body too far, under ``code``."""

    name = "docslab.far_spring"
    code: str = "docslab.too-far"

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers),
                    problems=(Problem(self.code, "the body is too far", "move it nearer"),))


def refused_with(entry) -> BuildRefused:
    experiment = coupled_toy()
    experiment.attach(entry, body="probe")
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    return refused.value


def test_r1_a_lab_problem_code_under_its_namespace_is_refused_uncatalogued():
    refused = refused_with(FarSpring())
    assert refused.codes() == ["docslab.too-far"]
    (refusal,) = refused.refusals
    assert refusal.uncatalogued and refusal.category is RefusalClass.USER


@pytest.mark.parametrize("code", ["too-far", "drag.too-far"], ids=["no namespace", "a word of MIME's"])
def test_r1_a_lab_problem_code_outside_its_namespace_is_refused_as_its_entrys_defect(code):
    refused = refused_with(FarSpring(code=code))
    assert refused.codes() == ["entry.defect"]
    assert f"refusal code {code!r} is not in MIME's refusal table" in str(refused)


# -- R2: the scan finds a code wherever it is written ---------------------------------------------------------------

def test_r2_a_code_planted_in_a_tuple_is_found_and_fails_the_table(tmp_path):
    root = copy_checkout(tmp_path)
    edit(root / "src/mime/scene/_frame.py", "def meters_per_unit(",
         'def planted_problem() -> Problem:\n    return ("frame.planted-code", "a planted message", "a planted action")'
         '\n\n\ndef meters_per_unit(')
    problems = refusal_check(root).problems
    assert [p for p in problems if p.startswith("frame.planted-code:")] == [
        f"frame.planted-code: constructed at scene/_frame.py:{_line(root, 'frame.planted-code')} and not in "
        f"{root / 'src/mime/build/refusal_table.txt'}; add the row 'frame.planted-code  user'"]


def test_r2_a_code_passed_by_a_variable_is_found_by_the_cross_check(tmp_path):
    root = copy_checkout(tmp_path)
    edit(root / "src/mime/build/_construct.py", "_REFUSED_ISSUES = {",
         '_REFUSED_ISSUES = {"PLANTED": "maddening.planted", ')
    assert any(p.startswith("maddening.planted: constructed at build/_construct.py:") and
               p.endswith("add the row 'maddening.planted  internal'") for p in refusal_check(root).problems)


def test_r2_a_code_shaped_string_that_is_neither_a_code_nor_a_name_fails(tmp_path):
    root = copy_checkout(tmp_path)
    edit(root / "src/mime/interactions/gravity.py", '"""The gravity family: weight and buoyancy."""',
         '"""The gravity family: weight and buoyancy."""\n\nPLANTED = "gravity.planted-name"')
    assert any("'gravity.planted-name' is shaped like a refusal code and the scan finds it neither constructed nor a "
               "name" in p for p in refusal_check(root).problems)


def test_r2_a_table_row_nothing_constructs_fails(tmp_path):
    root = copy_checkout(tmp_path)
    table = root / "src/mime/build/refusal_table.txt"
    table.write_text(table.read_text(encoding="utf-8") + "load.planted-row  user\n", encoding="utf-8")
    assert f"{table}: load.planted-row is constructed nowhere in the source; delete its row" in \
        refusal_check(root).problems


def _line(root: Path, code: str) -> int:
    lines = (root / "src/mime/scene/_frame.py").read_text(encoding="utf-8").splitlines()
    return next(number for number, line in enumerate(lines, start=1) if code in line)


# -- R3: the constructor ------------------------------------------------------------------------------------------

def test_r3_an_uncatalogued_mime_code_raises_where_it_is_constructed():
    assert namespace_of("drag.planted") == "mime"
    with pytest.raises(UncataloguedRefusal, match=r"refusal code 'drag.planted' is not in MIME's refusal table"):
        Refusal("drag.planted", RefusalClass.USER, "a message", "an action")


def test_r3_a_catalogued_code_in_a_class_the_table_does_not_give_it_raises():
    assert REFUSAL_TABLE["load.input-hash"] == {RefusalClass.USER, RefusalClass.ENVIRONMENT}
    Refusal("load.input-hash", RefusalClass.ENVIRONMENT, "a message", "an action")
    with pytest.raises(UncataloguedRefusal, match=r"is refused as environment and user in MIME's refusal table, not "
                                                  r"as internal"):
        Refusal("load.input-hash", RefusalClass.INTERNAL, "a message", "an action")


def test_r3_a_registered_namespaces_code_is_accepted_uncatalogued():
    refusal = Refusal("docslab.anything", RefusalClass.INTERNAL, "a message", "an action")
    assert refusal.uncatalogued and not Refusal("load.version", RefusalClass.ENVIRONMENT, "m", "a").uncatalogued
