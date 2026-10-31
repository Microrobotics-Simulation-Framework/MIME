"""The refusal table: one row per code MIME constructs, with the classes it is refused in, generated from the scan of
the source and checked against it; the constructor's check that a MIME code is in it; and the golden numbers of the
scan, measured on the tree this table landed on.

The controls: a planted lab ``Problem`` code under a registered namespace is refused, not raised, and so is one
outside it (R1); a code planted in a ``(code, message, action)`` tuple, one passed by a variable, one built where it is
used (an f-string), and a renamed code that a mapping of literal codes still holds, are each found by the scan and fail
the table, as does a code on two rows of the table (R2); an uncatalogued MIME code, and a catalogued one in the wrong
class, raise where they are constructed, and a namespace named like the first word of a MIME code is refused, so that
it cannot take those codes out of the check (R3). The codes no test names are read in one pass over the tests' text,
which a property holds to the search it replaced, over drawn texts and codes; its control is a reading that drops the
rule about a dot. A property over random texts reaches a single branch of the reading too rarely to be a control (a
reading without one rule failed it in 4, 1, 0 and 12 of 12 fresh runs), so a table of cases, one or more for each
branch, holds the reading to the search on every run, and each fault below fails it on every run."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import re

import pytest
from checkout import ROOT, copy_checkout, edit
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.build import REFUSAL_TABLE, BuildRefused, Refusal, RefusalClass, UncataloguedRefusal
from mime.docs import census
from mime.docs._check import _NAMED, named_strings, refusal_check, unnamed_codes
from mime.materials._records import _MEMBER_CODES
from mime.registry import Plan, PlanContext, Problem, register_interaction, registered
from mime.testing import installed_entries
from mime.testing.toys import ToySpring, coupled_toy
from mime.vocabulary import MIME_NAMESPACE, _vocabulary, namespace_of, register_namespace

GOLDEN = json.loads((Path(__file__).parent / "golden_numbers.json").read_text(encoding="utf-8"))


#: Code characters, the characters either side of a name, and others: a text drawn from these meets every case of the
#: search's boundaries (a dot before a letter, a dot before a hyphen, a hyphen, a non-ASCII letter, a line's end).
TEXT_CHARACTERS = "ab1_.-. :'\"\n(é"


def searched(code: str, text: str) -> bool:
    return re.search(_NAMED.format(re.escape(code)), text) is not None


@settings(deadline=None)
@given(text=st.text(TEXT_CHARACTERS, max_size=40), drawn=st.lists(st.text("ab1_.-", min_size=1, max_size=6)),
       cuts=st.lists(st.tuples(st.integers(0, 40), st.integers(0, 40))))
def test_the_one_pass_reading_names_a_code_exactly_when_the_search_finds_it(text, drawn, cuts):
    codes = [*drawn, *(text[start:end] for start, end in cuts if text[start:end] and re.fullmatch(r"[A-Za-z0-9_.-]+",
                                                                                              text[start:end]))]
    names = named_strings(text)
    assert [code for code in codes if (code in names) != searched(code, text)] == []


#: ``(text, code, whether the search finds the code named)``, one or more for each branch of the reading: a run of code
#: characters whole, and not its start before a dot and a letter, digit or underscore; its start before a dot that
#: none of those follows (at the run's end, before a hyphen, before another dot), one or several such dots in a run; a
#: hyphen as a code character; a character that is no code character ending a run; and a code character before a name,
#: which extends the run.
BRANCHES = [
    ("x a.b-c y", "a.b-c", True), ("a.b", "a.b", True), ("a.b", "a", False), ("a._b", "a", False),
    ("a.1", "a", False), ("see a. x", "a", True), ("a.", "a", True), ("c.-d", "c", True), ("a..b", "a", True),
    ("a.-b. x", "a.-b", True), ("a.-b. x", "a", True), ("a.-b. x", "a.-b.", True), ("a.b.-c", "a.b", True),
    ("a.b.-c", "a", False), ("a-b", "a-b", True), ("a-b", "a", False), ("a-b", "b", False), ("aé", "a", True),
    ("é a", "a", True), ("xa", "a", False), ("x.a", "a", False), ('"a"', "a", True), ("a\nb", "a", True),
    ("see a.b, a. and c.-d", "a.b", True), ("see a.b, a. and c.-d", "a", True), ("see a.b, a. and c.-d", "c", True),
]


def branch_mismatches(reading) -> list[tuple[str, str, bool]]:
    """The cases of ``BRANCHES`` that ``reading`` (a function of the text to the set of its names) gets wrong."""
    return [(text, code, named) for text, code, named in BRANCHES if (code in reading(text)) != named]


def reading_with(code_characters: str, dot_end: str, every_dot: bool = True):
    """A reading like ``named_strings`` with one rule changed: the characters of a run, the pattern of the dots a name
    may end before, and whether every such dot cuts or only the first."""
    run, end = re.compile(code_characters), re.compile(dot_end)

    def reading(text: str) -> set[str]:
        found: set[str] = set()
        for each in set(run.findall(text)):
            found.add(each)
            cuts = list(end.finditer(each))
            found.update(each[:cut.start()] for cut in (cuts if every_dot else cuts[:1]))
        return found
    return reading


READINGS = {
    "drops the rule about a dot": reading_with(r"[A-Za-z0-9_.-]+", r"(?!)"),
    "cuts only at a run's first dot": reading_with(r"[A-Za-z0-9_.-]+", r"\.(?![A-Za-z0-9_])", every_dot=False),
    "treats a dot before a hyphen as no boundary": reading_with(r"[A-Za-z0-9_.-]+", r"\.(?![A-Za-z0-9_-])"),
    "does not take a hyphen for a code character": reading_with(r"[A-Za-z0-9_.]+", r"\.(?![A-Za-z0-9_])"),
    "cuts at every dot": reading_with(r"[A-Za-z0-9_.-]+", r"\."),
    "cuts only at the last dot": reading_with(r"[A-Za-z0-9_.-]+", r"\.(?![A-Za-z0-9_])(?!.*\.(?![A-Za-z0-9_]))"),
}


def test_the_table_of_branches_is_what_the_search_finds():
    assert [(text, code, named) for text, code, named in BRANCHES if searched(code, text) != named] == []


def test_the_one_pass_reading_gets_every_branch_right():
    assert branch_mismatches(named_strings) == []
    assert branch_mismatches(reading_with(r"[A-Za-z0-9_.-]+", r"\.(?![A-Za-z0-9_])")) == []


@pytest.mark.parametrize("fault", READINGS)
def test_control_a_reading_with_one_rule_changed_fails_the_table_of_branches(fault):
    assert branch_mismatches(READINGS[fault]) != []


def test_a_code_with_a_character_that_is_no_code_character_is_searched_for(tmp_path):
    (tmp_path / "test_x.py").write_text('x a:b y\nsee "a b" and a.b\n', encoding="utf-8")
    assert unnamed_codes(["a:b", "a b", "a.b", "a:c", "zz"], tmp_path) == {"a:c", "zz"}
    # A lookup among the names only, for every code, would call the first two unnamed.
    names = named_strings((tmp_path / "test_x.py").read_text(encoding="utf-8"))
    assert {code for code in ["a:b", "a b", "a.b", "a:c", "zz"] if code not in names} != {"a:c", "zz"}


def test_control_a_reading_without_the_rule_about_a_dot_differs_from_the_search():
    runs_only = set(re.findall(r"[A-Za-z0-9_.-]+", "see a.b, a. and c.-d"))
    assert [code for code in ("a", "c") if (code in runs_only) != searched(code, "see a.b, a. and c.-d")] == ["a", "c"]
    assert [code for code in ("a", "c", "a.b") if (code in named_strings("see a.b, a. and c.-d"))
            != searched(code, "see a.b, a. and c.-d")] == []


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
    assert REFUSAL_TABLE["load.assets"] == {RefusalClass.USER, RefusalClass.ENVIRONMENT}
    Refusal("load.assets", RefusalClass.ENVIRONMENT, "a message", "an action")
    with pytest.raises(UncataloguedRefusal, match=r"is refused as environment and user in MIME's refusal table, not "
                                                  r"as internal"):
        Refusal("load.assets", RefusalClass.INTERNAL, "a message", "an action")


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
        Refusal("load.assets", RefusalClass.INTERNAL, "a message", "an action")
    taken: list[str] = []
    for word in sorted({code.split(".", 1)[0] for code in REFUSAL_TABLE} - {"post-build"}):
        try:
            register_namespace(word, "0.0.1")
        except ValueError:
            continue
        taken.append(word)
    assert taken == [], f"namespaces taking MIME's codes were registered: {taken}"
    register_namespace("docslab_other", "0.0.1")


def test_a_namespace_named_like_the_first_word_of_a_mime_entry_is_refused(namespaces):
    """The review's case, ``gravity``: registered, it would make ``namespace_of("gravity")`` answer ``gravity``, taking
    MIME's entry for its own. Every MIME entry's first word is refused; a fresh name registers."""
    with pytest.raises(ValueError, match=r"namespace 'gravity' is the first word of MIME's entries \(gravity\)"):
        register_namespace("gravity", "0.0.1")
    assert namespace_of("gravity") == "mime"
    words = {record.name.split(".", 1)[0] for record in registered() if record.namespace == MIME_NAMESPACE}
    assert {"gravity", "drag", "magnetic", "tube", "rigid_inertial", "swim_speed"} <= words
    taken: list[str] = []
    for word in sorted(words):
        try:
            register_namespace(word, "0.0.1")
        except ValueError:
            continue
        taken.append(word)
    assert taken == [], f"namespaces taking MIME's entries were registered: {taken}"
    assert all(namespace_of(word) == "mime" for word in words)
    register_namespace("docslab_fresh", "0.0.1")
    assert namespace_of("docslab_fresh.law") == "docslab_fresh"


def test_r3_a_registered_namespaces_code_is_accepted_uncatalogued():
    refusal = Refusal("docslab.anything", RefusalClass.INTERNAL, "a message", "an action")
    assert refusal.uncatalogued and not Refusal("load.version", RefusalClass.ENVIRONMENT, "m", "a").uncatalogued
