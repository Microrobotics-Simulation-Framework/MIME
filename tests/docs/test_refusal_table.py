"""The refusal table: one row per code MIME constructs, with the classes it is refused in, generated from the scan of
the source and checked against it; the constructor's check that a MIME code is in it; and the golden numbers of the
scan, measured on the tree this table landed on.

The controls: a planted lab ``Problem`` code under a registered namespace is refused, not raised, and so is one
outside it (R1); a code planted in a ``(code, message, action)`` tuple, one passed by a variable, one built where it is
used (an f-string), and a renamed code that a mapping of literal codes still holds, are each found by the scan and fail
the table, as does a code on two rows of the table (R2); an uncatalogued MIME code, and a catalogued one in the wrong
class, raise where they are constructed, and a namespace named like the first word of a MIME code is refused, so that
it cannot take those codes out of the check (R3)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
from checkout import ROOT, copy_checkout, edit

from mime.build import REFUSAL_TABLE, BuildRefused, Refusal, RefusalClass, UncataloguedRefusal
from mime.docs import census
from mime.docs._check import refusal_check
from mime.materials._records import _MEMBER_CODES
from mime.registry import Plan, PlanContext, Problem, register_interaction
from mime.testing import installed_entries
from mime.testing.toys import ToySpring, coupled_toy
from mime.vocabulary import _vocabulary, namespace_of, register_namespace

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


def test_r2_a_code_built_where_it_is_used_fails(tmp_path):
    root = copy_checkout(tmp_path)
    records = root / "src/mime/materials/_records.py"
    built = 'f"material.{member}"'
    edit(records, "found.append((_MEMBER_CODES[member], ", f"found.append(({built}, ")
    assert [p for p in refusal_check(root).problems if "builds its code" in p] == [
        f"materials/_records.py:{_line(root, built, records)}: a (code, message, action) tuple builds its code where "
        f"it is used, so the scan cannot read it; write each code it can be as a literal (a lookup in a mapping of "
        f"literal codes)"]


def test_r2_a_code_renamed_where_it_is_written_and_kept_in_a_mapping_of_codes_fails(tmp_path):
    """The review's plant: one of the member codes renamed at its literal construction, and in the table and the list,
    while the mapping still holds the old code, which a build would then construct uncatalogued."""
    code = _MEMBER_CODES["nominal"]
    renamed = f"{code}-value"
    root = copy_checkout(tmp_path)
    records = root / "src/mime/materials/_records.py"
    edit(records, f'found.append(("{code}", ', f'found.append(("{renamed}", ')
    for path in (root / "src/mime/build/refusal_table.txt", root / "docs/ratchets/refusal_codes.txt"):
        edit(path, f"\n{code} ", f"\n{renamed} ")
    held = f'"nominal": "{code}"'
    assert refusal_check(root).problems == [
        f"materials/_records.py:{_line(root, held, records)}: {code!r} is shaped like a refusal code and the scan finds "
        f"it neither constructed nor a name; if it is a name, add '{code}  # <what it names>' to "
        f"{root / 'docs/ratchets/refusal_names.txt'}"]


def test_r2_a_code_on_two_rows_of_the_table_fails(tmp_path):
    root = copy_checkout(tmp_path)
    table = root / "src/mime/build/refusal_table.txt"
    lines = table.read_text(encoding="utf-8").splitlines()
    number, first = next((n, line) for n, line in enumerate(lines, start=1) if line and not line.startswith("#"))
    code = first.split()[0]
    table.write_text("\n".join([*lines, f"{code}  internal"]) + "\n", encoding="utf-8")
    assert refusal_check(root).problems == [
        f"{table}: line {len(lines) + 1}: {code} has a row already, at line {number}; keep one row"]


def _line(root: Path, text: str, path: Path | None = None) -> int:
    lines = (path or root / "src/mime/scene/_frame.py").read_text(encoding="utf-8").splitlines()
    return next(number for number, line in enumerate(lines, start=1) if text in line)


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


@pytest.fixture
def namespaces(monkeypatch):
    """The registered namespaces, restored after the test: a namespace the check fails to refuse stays in no test."""
    monkeypatch.setattr(_vocabulary, "_namespaces", dict(_vocabulary._namespaces))


def test_r3_a_namespace_named_like_the_first_word_of_a_mime_code_is_refused(namespaces):
    with pytest.raises(ValueError, match=r"namespace 'load' is the first word of MIME's refusal codes \(load\."):
        register_namespace("load", "0.0.1")
    assert namespace_of("load.planted") == "mime"
    with pytest.raises(UncataloguedRefusal, match=r"'load.planted' is not in MIME's refusal table"):
        Refusal("load.planted", RefusalClass.USER, "a message", "an action")
    with pytest.raises(UncataloguedRefusal, match=r"not as internal"):
        Refusal("load.input-hash", RefusalClass.INTERNAL, "a message", "an action")
    taken: list[str] = []
    for word in sorted({code.split(".", 1)[0] for code in REFUSAL_TABLE} - {"post-build"}):
        try:
            register_namespace(word, "0.0.1")
        except ValueError:
            continue
        taken.append(word)
    assert taken == [], f"namespaces taking MIME's codes were registered: {taken}"
    register_namespace("docslab_other", "0.0.1")


def test_r3_a_registered_namespaces_code_is_accepted_uncatalogued():
    refusal = Refusal("docslab.anything", RefusalClass.INTERNAL, "a message", "an action")
    assert refusal.uncatalogued and not Refusal("load.version", RefusalClass.ENVIRONMENT, "m", "a").uncatalogued
