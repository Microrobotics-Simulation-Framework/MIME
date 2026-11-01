"""The generality pilot: the page generators serve a body that does not swim, with no new page type.

Two sets are built and read. The ``Magnetic.Force`` set is a gradient-pulled magnet's: ``Magnetic.Force``,
``PointDipole``, ``Magnetic.Torque`` and ``Gravity``, their nodes, the magnetic and gravity families and the load
mechanism. ``Drag`` is the second, and the one that does swim: its entry, family and node pages, read the same way. What
the pages are held to:

- an entry without rungs has no rung section, and one with rungs has one per rung;
- a family page compares only members that can replace one another (the family's suite holds a swap case, read from the
  family itself), by the rows under its comparison heading, whatever the table is headed; a family with one member says
  so, and the glance of an entry says "compared on its page" only where the family page does, with the family page's
  count of members;
- every setting and value a family names as a material's property is on the page of each of its members, and an entry
  that writes its own ``plan`` says that what ``plan`` adds is not in its tables;
- no entry, node or family page of the ``Magnetic.Force`` set holds a word only a swimmer fills in the text the
  generators write (hand-written text is not scanned), none of the generators' strings names one, a data-only entry's
  page has exactly the sections its data calls for, and no section of any page of either set is a heading with nothing
  under it;
- a validity line is printed whole, as a bullet of the Validity section, under a sentence that says a condition is a
  point or a region, and an entry whose support note states where it holds declares a validity line;
- interface names on an entry page link to the vocabulary, a target to its mechanism page, and the vocabulary page
  names the entries that publish a field;
- every node an entry of a set builds has a page that is its equations and symbols or says it is undocumented, and
  names the entry; every mechanism a set delivers into lists the entry.

Each plant breaks one generator (or the data it reads), in the process that builds, at the branch that serves the rule,
and the check over the built pages must name it: a plant that the check passes would be a control that cannot fail. A
control that must pass (hand-written text that names a swimmer's word) is built the same way and must be named by
nothing. The pages are built from the checkout in a fresh process, as the build reads the registry: a test of another
file may have registered entries of its own, and the pages of this process would hold them.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from checkout import ROOT

from mime.docs import _build, _entry_page, _family_page, _markdown, _node_page, _reference_pages
from mime.docs._build import Built, build_pages
from mime.docs._context import (
    VOCABULARY_PATH,
    Context,
    Record,
    built_rungs,
    builds_of,
    class_var,
    entry_path,
    family_path,
    kind_dir,
    mechanism_path,
    node_path,
)
from mime.docs._export import export, load_namespaces, registry_export
from mime.docs._markdown import items, mapping, strings
from mime.testing import CaseKind, registered_families
from mime.vocabulary import lookup

#: Words only a swimmer fills: a template that prints or requires one is keyed on a swimmer.
SWIMMER = re.compile(r"helix|helical|swim|tube|vessel|rotation|rotating", re.IGNORECASE)
#: The words of a support note that state where a law holds, which a validity line on the same entry must then carry.
WHERE_IT_HOLDS = re.compile(r"valid while|outside", re.IGNORECASE)
GENERATORS = (_entry_page, _family_page, _markdown, _node_page, _reference_pages)
#: A data-only entry that is not a swimmer's: one read, one delivery, no rung, no family, no value and no validity line.
FIXTURE = ("interaction", "toy.hold")
#: The sections its page has: those a swimmer fills, and every other, are not in it.
FIXTURE_HEADINGS = ["# interaction toy.hold", "## At a glance", "## What it models, and when not to use it",
                    "## Settings", "## Roles, reads and deliveries", "## Nodes it builds", "## Gradient status",
                    "## Validity", "## References", "## Other declarations"]


@dataclass(frozen=True)
class PilotSet:
    name: str
    entries: tuple[tuple[str, str], ...]
    #: The target kinds whose mechanism pages the set reads.
    mechanisms: tuple[str, ...]
    #: Whether the set is a non-swimmer's, whose entry, node and family pages hold no word only a swimmer fills.
    swimmer_free: bool


MAGNETIC_FORCE = PilotSet(
    "Magnetic.Force",
    (("interaction", "magnetic.force"), ("field_provider", "point_dipole"), ("interaction", "magnetic.torque"),
     ("interaction", "gravity")), ("load",), True)
DRAG = PilotSet("Drag", (("interaction", "drag"),), ("load", "operator"), False)
SETS = (MAGNETIC_FORCE, DRAG)


def page_path(path: str) -> str:
    return f"pages/{path}"


def sections(text: str, level: int = 2) -> dict[str, str]:
    """The body of each heading of ``level`` of a page, up to the next heading of that level or a higher one."""
    found: dict[str, str] = {}
    marker = "#" * level + " "
    current: str | None = None
    for line in text.splitlines():
        if line.startswith(marker):
            current = line.removeprefix(marker)
            found[current] = ""
        elif current is not None and not re.match(r"#{1," + str(level) + "} ", line):
            found[current] += line + "\n"
        elif current is not None:
            current = None
    return found


def headings(text: str) -> list[str]:
    """Every heading of a page, outside a code fence."""
    found: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", line):
            found.append(line)
    return found


def empty_sections(text: str) -> list[str]:
    """Each heading (outside a code fence) followed by another heading of its level or a higher one, or by nothing."""
    lines = [line for line in text.splitlines() if line.strip()]
    empty: list[str] = []
    fenced = False
    for number, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
        heading = None if fenced else re.match(r"(#{1,6}) ", line)
        if heading is None:
            continue
        following = re.match(r"(#{1,6}) ", lines[number + 1]) if number + 1 < len(lines) else None
        if number + 1 == len(lines) or (following is not None and len(following.group(1)) <= len(heading.group(1))):
            empty.append(line)
    return empty


def table_rows(section: str) -> int:
    """The rows under a table's header in a section, whatever the header says."""
    return max(sum(line.startswith("|") for line in section.splitlines()) - 2, 0)


def swappable(family: Any) -> bool:
    """Whether the members of ``family`` can replace one another, read from the cases the family declares: the rule's
    authority, not the generators' helper for it."""
    return any(case.value == "swap" for case in family.cases)


def source_problems(sources: Mapping[str, str]) -> list[str]:
    """Each string a generator's source holds that names a swimmer's field (its docstrings included)."""
    problems: list[str] = []
    for name, text in sorted(sources.items()):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and SWIMMER.search(node.value):
                problems.append(f"{name} holds the string {node.value[:50]!r}, which names a swimmer's field")
    return problems


def generator_sources() -> dict[str, str]:
    return {Path(module.__file__ or "").name: Path(module.__file__ or "").read_text(encoding="utf-8")
            for module in GENERATORS}


def region_note_problems(record: Record) -> list[str]:
    """A support note that states where the law holds, and no validity line on the entry that says so."""
    notes = [str(mapping(row).get("note", "")) for row in items(class_var(record, "support", []))]
    stated = [note for note in notes if WHERE_IT_HOLDS.search(note)]
    if stated and not strings(class_var(record, "validity", [])):
        return [f"{record['name']}: a support note states where the law holds ({stated[0][:50]!r}) and the entry "
                f"declares no validity line"]
    return []


def fixture_problems(built: Built, by_key: Mapping[tuple[str, str], Record]) -> list[str]:
    """The page of a data-only entry has the sections its data calls for, and no other."""
    record = by_key[FIXTURE]
    page = built.pages.get(page_path(entry_path(record)))
    found = [] if page is None else headings(page.text)
    return [] if found == FIXTURE_HEADINGS else [
        f"the page of the data-only entry {record['name']} has the sections {found}, and its data calls for "
        f"{FIXTURE_HEADINGS}"]


def set_problems(built: Built, generated: Built, records: Sequence[Record], held: PilotSet) -> list[str]:
    """What the pages of ``held`` get wrong against the generality rule. ``generated`` is the same build with no
    hand-written text: the swimmer-word scan reads it, since authored text may use any word its subject needs."""
    problems: list[str] = []
    families = {family.name: family for family in registered_families()}
    by_key = {(kind_dir(str(record["kind"])), str(record["name"])): record for record in records}

    def text_of(path: str) -> str | None:
        page = built.pages.get(page_path(path))
        if page is None:
            problems.append(f"{held.name}: no page {path}")
        return None if page is None else page.text

    pages: dict[str, str] = {}
    for key in held.entries:
        record = by_key[key]
        who, path = f"{held.name}: {record['name']}", entry_path(record)
        if (text := text_of(path)) is None:
            continue
        pages[path] = text
        problems += [f"{held.name}: {problem}" for problem in region_note_problems(record)]
        rungs = strings(class_var(record, "rungs", []))
        if not rungs:
            for stray in ("\n## Rungs", "### Rung ", "A rung is a section of this page", "Rungs built:"):
                if stray in text:
                    problems.append(f"{who} has no rungs, and its page holds a rung section ({stray.strip()!r})")
            if "## Nodes it builds" not in text:
                problems.append(f"{who} has no rungs, and its page has no 'Nodes it builds' section")
        else:
            for rung in ("", *rungs):
                if rung and f"### Rung {rung}" not in text:
                    problems.append(f"{who} has the rung {rung}, and its page has no section for it")
            if "\n## Rungs\n" not in text:
                problems.append(f"{who} has rungs, and its page has no 'Rungs' section")
        problems += validity_problems(who, record, text)
        problems += link_problems(who, record, text)
        if bool(record.get("overrides_plan")) != ("`plan` adds" in sections(text).get("Roles, reads and deliveries", "")):
            problems.append(f"{who} {'writes' if record.get('overrides_plan') else 'does not write'} its own plan, and "
                            f"its page {'does not say' if record.get('overrides_plan') else 'says'} that what the "
                            f"plan adds is not in its tables")
        family = class_var(record, "family")
        if family is not None:
            problems += glance_problems(who, text, text_of(family_path(str(family))), records, str(family),
                                        families.get(str(family)), rungs)
        for dotted in sorted({node for nodes in builds_of(record).values() for node in nodes}):
            node = dotted.rsplit(".", 1)[1]
            if (node_text := text_of(node_path(node))) is None:
                continue
            pages[node_path(node)] = node_text
            documented = "## Governing equations" in node_text and "## Symbols" in node_text
            if not documented and "undocumented: this node class declares no `NodeMeta`" not in node_text:
                problems.append(f"{held.name}: the node page of {node} is neither documented nor says it is not")
            if f"]({'../' + entry_path(record)})" not in sections(node_text).get("Built by", ""):
                problems.append(f"{held.name}: the node page of {node} does not link its entry {record['name']}")
        for label, delivery in sorted(mapping(class_var(record, "delivers", {})).items()):
            kind = lookup(str(mapping(delivery)["target"]))
            if kind is None or kind.kind.value not in held.mechanisms:
                problems.append(f"{who} delivers {label} into a kind with no mechanism page in the set")
            elif (mech := text_of(mechanism_path(kind.kind.value))) is not None \
                    and f"]({'../' + entry_path(record)})" not in sections(mech).get("Entries that deliver into it", ""):
                problems.append(f"{held.name}: the {kind.kind.value} mechanism page does not list {record['name']}")
        problems += publishing_problems(who, record, built)
    for name in sorted({str(class_var(by_key[key], "family")) for key in held.entries
                        if class_var(by_key[key], "family") is not None}):
        if (text := text_of(family_path(name))) is None:
            continue
        pages[family_path(name)] = text
        members = [record for record in records if class_var(record, "family") == name]
        problems += family_problems(held.name, name, families[name], members, text)
        for member in members:
            if (member_text := text_of(entry_path(member))) is not None:
                problems += material_problems(held.name, member, families[name], member_text)
    for path, text in sorted(pages.items()):
        problems += [f"{held.name}: {path} has an empty section {line!r}" for line in empty_sections(text)]
        if held.swimmer_free and (found := SWIMMER.search(generated.pages[page_path(path)].text)):
            problems.append(f"{held.name}: {path} holds {found.group()!r}, a word only a swimmer fills")
    if held.swimmer_free:
        problems += fixture_problems(built, by_key)
    return problems


def validity_problems(who: str, record: Record, text: str) -> list[str]:
    """The Validity section of a page, and only it: its sentence, and its bullets, which are the declared lines whole."""
    validity = strings(class_var(record, "validity", []))
    section = sections(text).get("Validity", "")
    glance = sections(text).get("At a glance", "")
    count = f"{len(validity)} declared line{'' if len(validity) == 1 else 's'}" if validity else "none declared"
    problems = [] if f"- Validity: {count}\n" in glance else [
        f"{who}: the glance does not say 'Validity: {count}'"]
    if not validity:
        return problems + ([] if "none declared" in section else [
            f"{who} declares no validity line, and its page does not say so"])
    intro = section.strip().splitlines()[0] if section.strip() else ""
    if not ("point" in intro and "region" in intro):
        problems.append(f"{who}: the validity section's sentence does not say a condition is a point or a region: "
                        f"{intro!r}")
    printed = [line.removeprefix("- ") for line in section.splitlines() if line.startswith("- ")]
    problems += [f"{who}: a validity line is not printed whole in the Validity section: {line[:60]!r}"
                 for line in validity if line not in printed]
    problems += [f"{who}: the Validity section prints a line the entry does not declare: {line[:60]!r}"
                 for line in printed if line not in validity]
    return problems


def link_problems(who: str, record: Record, text: str) -> list[str]:
    """Each interface a law reads is linked to the vocabulary, and each target it delivers into to its mechanism page."""
    problems: list[str] = []
    for label, read in sorted(mapping(class_var(record, "reads", {})).items()):
        name = str(mapping(read)["interface"])
        if f"[`{name}`](../../{VOCABULARY_PATH}#" not in text:
            problems.append(f"{who}: the read {label} of {name} is not linked to the vocabulary page")
    for label, delivery in sorted(mapping(class_var(record, "delivers", {})).items()):
        target = str(mapping(delivery)["target"])
        kind = lookup(target)
        if kind is not None and f"[`{target}`](../../{mechanism_path(kind.kind.value)})" not in text:
            problems.append(f"{who}: the delivery {label} into {target} is not linked to the {kind.kind.value} "
                            f"mechanism page")
    return problems


def publishing_problems(who: str, record: Record, built: Built) -> list[str]:
    """Each interface an entry publishes is linked to the vocabulary on its page, and the vocabulary page names the
    entry against it."""
    problems: list[str] = []
    page = built.pages.get(page_path(entry_path(record)))
    vocabulary = built.pages.get(page_path(VOCABULARY_PATH))
    for name in sorted(mapping(class_var(record, "publishes", {}))):
        if page is not None and f"[`{name}`](../../{VOCABULARY_PATH}#" not in page.text:
            problems.append(f"{who}: the interface {name} it publishes is not linked to the vocabulary page")
        row = next((line for line in (vocabulary.text if vocabulary else "").splitlines()
                    if line.startswith(f"| `{name}` |")), "")
        if f"]({entry_path(record)})" not in row:
            problems.append(f"{who}: the vocabulary page does not say that {record['name']} publishes {name}")
    return problems


def material_problems(name: str, member: Record, family: Any, text: str) -> list[str]:
    """Each setting or value the family names as a material's property is on the page of the member."""
    asked = sections(text).get("Settings", "") + sections(text).get("Roles, reads and deliveries", "")
    return [f"{name}: the page of {member['name']} does not carry {material}, which its family {family.name} names as "
            f"a material's property" for material in family.material if f"`{material}`" not in asked]


def glance_problems(who: str, text: str, family_text: str | None, records: Sequence[Record], name: str,
                    family: Any, rungs: list[str]) -> list[str]:
    """What the glance says of the family: its count of members is the family page's, and it says the family is compared
    where the family's declared cases say its members can replace one another, and the page compares."""
    glance = sections(text).get("At a glance", "")
    problems: list[str] = []
    if family_text is not None and family is not None:
        count = sum(len(built_rungs(record)) for record in records if class_var(record, "family") == name)
        listed = table_rows(sections(family_text).get("Members", ""))
        says = re.search(r", with (\d+) (?:built rung|member)", glance)
        if not says or int(says.group(1)) != listed:
            problems.append(f"{who}: the glance counts {says.group(1) if says else 'no'} members of the family, and the "
                            f"family page lists {listed}")
        compared = table_rows(sections(family_text).get("Comparison", "")) > 0
        claims = "; compared on its page" in glance
        if claims != (swappable(family) and count >= 2) or claims != compared:
            problems.append(f"{who}: the glance says the family is {'' if claims else 'not '}compared on its page, the "
                            f"family's cases say {'it is' if swappable(family) and count >= 2 else 'it is not'}, and "
                            f"the family page {'does' if compared else 'does not'}")
    if not rungs and "built rung" in glance:
        problems.append(f"{who} has no rungs, and its glance counts built rungs")
    return problems


def family_problems(held: str, name: str, family: Any, members: Sequence[Record], text: str) -> list[str]:
    """The comparison of a family page: rows only for members that can replace one another, read from the family's
    declared cases, and what it says where it has none."""
    problems: list[str] = []
    built = sum(len(built_rungs(record)) for record in members)
    comparison = sections(text).get("Comparison", "")
    rows = table_rows(comparison)
    if not swappable(family) or built < 2:
        if rows:
            problems.append(f"{held}: the {name} family page compares its members, which "
                            + ("cannot replace one another" if not swappable(family) else "are fewer than two"))
    elif rows != built:
        problems.append(f"{held}: the {name} family page does not compare members that can replace one another "
                        f"({rows} rows for {built} members)")
    if not swappable(family) and built == 1 and not ("one " in comparison and "nothing to compare" in comparison):
        problems.append(f"{held}: the {name} family page does not say it has one member and nothing to compare")
    if not swappable(family) and built > 1 and "not alternatives" not in comparison:
        problems.append(f"{held}: the {name} family page does not say its members are not alternatives")
    if built == 1 and "not alternatives" in comparison:
        problems.append(f"{held}: the {name} family page calls the one member of the family not alternatives")
    if swappable(family) and built < 2:
        rungs = any(rung for record in members for rung in built_rungs(record))
        if f"two built {'rungs' if rungs else 'members'}" not in comparison:
            problems.append(f"{held}: the {name} family page does not name its members as {'rungs' if rungs else 'members'} "
                            f"in saying a comparison starts at two")
    listed = sections(text).get("Members", "")
    return problems + [f"{held}: the {name} family page does not list {record['name']}" for record in members
                       if f"]({'../' + entry_path(record)})" not in listed]


def pilot_problems(records: Sequence[Record], root: Path = ROOT) -> dict[str, list[str]]:
    """Build the pages (and the same pages with no hand-written text) and read each set; a build that raises is a
    problem of every set."""
    try:
        built = build_pages(root)
        with tempfile.TemporaryDirectory() as empty:
            generated = build_pages(Path(empty))
    except Exception as error:  # a template that needs a field the entry lacks is the failure under test
        return {held.name: [f"{held.name}: the build raised {type(error).__name__}: {error}"] for held in SETS}
    return {held.name: [*(f"{held.name}: the build failed: {problem}" for problem in built.problems),
                        *set_problems(built, generated, records, held)] for held in SETS}


def main(plant: str | None) -> None:
    """The entry of the fresh process: build the pages, with the plant named (or none), and print what each set got
    wrong as JSON. The records the check holds the pages to are the unplanted ones."""
    records = list(export(load_namespaces())["entries"])
    with pytest.MonkeyPatch.context() as patch:
        if plant is not None:
            {**PLANTS_BY_NAME, **CONTROLS}[plant](patch)
        found = pilot_problems(records)
    print(json.dumps(found))


def pilot_in_a_fresh_process(plant: str | None = None) -> dict[str, list[str]]:
    script = (f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
              f"import test_generality_pilot\ntest_generality_pilot.main({plant!r})")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return list(registry_export()["entries"])


# -- the passing arm: the pages of both sets satisfy every rule ---------------------------------------------------------

def test_the_pages_of_the_magnetic_force_set_and_of_drag_meet_the_generality_rule():
    assert pilot_in_a_fresh_process() == {"Magnetic.Force": [], "Drag": []}
    assert source_problems(generator_sources()) == []


def test_the_set_is_the_set_the_registry_holds(records):
    """The oracle behind the checks is not empty: the entries are there, rungless and rungful as the rules assume."""
    by_key = {(kind_dir(str(record["kind"])), str(record["name"])): record for record in records}
    for held in SETS:
        assert all(key in by_key for key in held.entries), held.name
    assert [class_var(by_key[key], "rungs", []) for key in MAGNETIC_FORCE.entries] == [[]] * 4
    assert class_var(by_key[DRAG.entries[0]], "rungs") == ["surrogate", "mrs"]
    assert [class_var(by_key[key], "family") for key in MAGNETIC_FORCE.entries] == ["magnetic", None, "magnetic",
                                                                                      "gravity"]
    assert any("outside" in line for line in strings(class_var(by_key[MAGNETIC_FORCE.entries[1]], "validity")))


def test_the_fixture_is_a_data_only_entry_and_the_gravity_page_carries_what_the_family_names(records):
    by_key = {(kind_dir(str(record["kind"])), str(record["name"])): record for record in records}
    fixture = by_key[FIXTURE]
    assert [len(mapping(class_var(fixture, name, {}))) for name in ("reads", "delivers", "values")] == [1, 1, 0]
    assert not strings(class_var(fixture, "rungs", [])) and not strings(class_var(fixture, "validity", []))
    assert class_var(fixture, "family") is None
    families = {family.name: family for family in registered_families()}
    gravity = by_key[MAGNETIC_FORCE.entries[3]]
    assert set(families["gravity"].material) <= set(mapping(class_var(gravity, "values")))


def test_a_support_note_that_states_where_a_law_holds_needs_a_validity_line_on_its_entry(records):
    by_key = {(kind_dir(str(record["kind"])), str(record["name"])): record for record in records}
    torque = by_key[("interaction", "magnetic.torque")]
    assert region_note_problems(torque) == []
    assert all(region_note_problems(by_key[key]) == [] for held in SETS for key in held.entries)
    bare = {**torque, "class_vars": {**torque["class_vars"], "validity": {**torque["class_vars"]["validity"],
                                                                           "value": []}}}
    (problem,) = region_note_problems(bare)
    assert "states where the law holds" in problem and "declares no validity line" in problem


# -- the family page where it has nothing to compare --------------------------------------------------------------------

def _family_context(records: Sequence[Record], family: Any) -> Context:
    return Context(tuple(records), {family.name: family}, {}, {})


def test_a_family_of_one_member_says_it_has_one_member_and_nothing_to_compare(records):
    families = {family.name: family for family in registered_families()}
    page = _family_page.family_page(families["gravity"], _family_context(records, families["gravity"])).text
    comparison = sections(page)["Comparison"]
    assert "one member" in comparison and "nothing to compare" in comparison and "not alternatives" not in comparison
    two = _family_page.family_page(families["magnetic"], _family_context(records, families["magnetic"])).text
    assert "not alternatives" in sections(two)["Comparison"]


def test_a_swappable_family_with_one_built_member_names_its_members_as_rungs_or_as_members(records):
    families = {family.name: family for family in registered_families()}
    swap = replace(families["gravity"], cases=(*families["gravity"].cases, CaseKind.SWAP))
    rungless = _family_page.family_page(swap, _family_context(records, swap)).text
    assert "two built members" in sections(rungless)["Comparison"] and "rung" not in sections(rungless)["Comparison"]
    one_rung = [with_class_vars(record, reserved_rungs={"mrs": "held back here"}) if record["name"] == "drag" else record
                for record in records]
    hydrodynamic = _family_page.family_page(families["hydrodynamic"], _family_context(one_rung, families["hydrodynamic"]))
    assert "two built rungs" in sections(hydrodynamic.text)["Comparison"]


# -- the plants: one generator broken at the branch that serves each rule -----------------------------------------------

Patch = Callable[[pytest.MonkeyPatch], None]


def with_class_vars(record: Record, **changed: object) -> Record:
    held = {name: dict(value) for name, value in mapping(record["class_vars"]).items()}
    for name, value in changed.items():
        held[name] = {"type": "t", "declared_by": "x", "default": None, **held.get(name, {}), "value": value}
    return {**record, "class_vars": held}


def plant_a_rung_section(patch: pytest.MonkeyPatch) -> None:
    """The branch for an entry without rungs removed: every entry is printed as though it had a rung."""
    real = _entry_page._rungs

    def planted(w: Any, record: Record, context: Any, path: str) -> None:
        if not strings(class_var(record, "rungs", [])):
            record = with_class_vars(record, rungs=["default"], builds={"default": builds_of(record).get("", [])})
        real(w, record, context, path)

    patch.setattr(_entry_page, "_rungs", planted)


def plant_a_missing_rung_section(patch: pytest.MonkeyPatch) -> None:
    """The mirror: an entry with rungs printed as though it had none."""
    real = _entry_page._rungs

    def planted(w: Any, record: Record, context: Any, path: str) -> None:
        real(w, with_class_vars(record, rungs=[]), context, path)

    patch.setattr(_entry_page, "_rungs", planted)


def plant_a_comparison_of_complementary_laws(patch: pytest.MonkeyPatch) -> None:
    """The family page's test of whether members can replace one another answered yes for every family."""
    real = _family_page._comparison

    def planted(w: Any, family: Any, members: Any, path: str) -> None:
        real(w, replace(family, cases=(*family.cases, CaseKind.SWAP)), members, path)

    patch.setattr(_family_page, "_comparison", planted)


def plant_a_comparison_in_a_table_of_another_header(patch: pytest.MonkeyPatch) -> None:
    """A second route to a comparison: the page says its members are not alternatives, and tables them under another
    header."""
    real = _family_page._comparison

    def planted(w: Any, family: Any, members: Any, path: str) -> None:
        real(w, family, members, path)
        if not _family_page.alternatives(family):
            w.table(["Entry", "Cost"], [[record["name"], "not measured"] for record, _ in members])

    patch.setattr(_family_page, "_comparison", planted)


def plant_a_helper_that_says_every_family_is_swappable(patch: pytest.MonkeyPatch) -> None:
    """The generators' shared test of whether members can replace one another answered yes for every family."""
    patch.setattr(_entry_page, "alternatives", lambda family: True)
    patch.setattr(_family_page, "alternatives", lambda family: True)


def plant_a_glance_that_says_compared(patch: pytest.MonkeyPatch) -> None:
    """The glance's test of whether the family page compares answered yes for every family."""
    patch.setattr(_entry_page, "alternatives", lambda family: True)


def plant_a_glance_count_one_too_many(patch: pytest.MonkeyPatch) -> None:
    """The glance counts a member the family page does not list."""
    real = _entry_page._family_members

    def planted(context: Any, family: str) -> str:
        return re.sub(r"with (\d+)", lambda found: f"with {int(found.group(1)) + 1}", real(context, family))

    patch.setattr(_entry_page, "_family_members", planted)


def plant_a_one_member_family_called_not_alternatives(patch: pytest.MonkeyPatch) -> None:
    """The family page as it was before the pilot's review: a family with one member says its members are not
    alternatives."""
    real = _family_page._comparison

    def planted(w: Any, family: Any, members: Any, path: str) -> None:
        if len(members) < 2 and not _family_page.alternatives(family):
            w.heading(2, "Comparison")
            w.paragraph("The members of this family are not alternatives to one another, so they are not compared.")
            return
        real(w, family, members, path)

    patch.setattr(_family_page, "_comparison", planted)


def plant_a_template_that_requires_a_helix(patch: pytest.MonkeyPatch) -> None:
    """A line of the settings template reads a swimmer's field from the entry."""
    real = _entry_page._settings

    def planted(w: Any, record: Record) -> None:
        real(w, record)
        w.paragraph(f"Helix pitch: {record['fields']['helix_pitch']}")

    patch.setattr(_entry_page, "_settings", planted)


def plant_a_template_that_prints_a_helix(patch: pytest.MonkeyPatch) -> None:
    """The same line, with the field defaulted: it builds, and prints a meaningless line on every page."""
    real = _entry_page._settings

    def planted(w: Any, record: Record) -> None:
        real(w, record)
        w.paragraph(f"Helix pitch: {record['fields'].get('helix_pitch', 'none')}")

    patch.setattr(_entry_page, "_settings", planted)


def plant_a_section_of_a_driven_swimmer_on_every_page(patch: pytest.MonkeyPatch) -> None:
    """A section a swimmer would fill, printed for every entry: no word of the scan, and no section left empty."""
    real = _entry_page._settings

    def planted(w: Any, record: Record) -> None:
        real(w, record)
        w.heading(2, "Drive frequency")
        w.paragraph("not set")

    patch.setattr(_entry_page, "_settings", planted)


def plant_a_point_only_validity_sentence(patch: pytest.MonkeyPatch) -> None:
    """The validity section's sentence as it was before the pilot: a condition is an operating point."""

    def planted(w: Any, record: Record) -> None:
        lines = strings(class_var(record, "validity", []))
        w.heading(2, "Validity")
        if not lines:
            w.paragraph("none declared")
            return
        w.paragraph("Declared validity lines: free text, not parsed, and not yet checked against the operating point.")
        w.bullets(lines)

    patch.setattr(_entry_page, "_validity", planted)


def plant_a_validity_line_clipped_to_its_point(patch: pytest.MonkeyPatch) -> None:
    """Each line printed up to its first colon or semicolon: the clause that says where it holds is dropped."""
    real = _entry_page._validity

    def planted(w: Any, record: Record) -> None:
        clipped = [re.split(r"[:;]", line)[0] for line in strings(class_var(record, "validity", []))]
        real(w, with_class_vars(record, validity=clipped))

    patch.setattr(_entry_page, "_validity", planted)


def plant_validity_lines_under_another_heading(patch: pytest.MonkeyPatch) -> None:
    """The lines printed whole, but under a heading of their own: the Validity section holds only its sentence."""

    def planted(w: Any, record: Record) -> None:
        lines = strings(class_var(record, "validity", []))
        w.heading(2, "Validity")
        if not lines:
            w.paragraph("none declared")
            return
        w.paragraph("Declared validity lines: free text, not parsed, and not yet checked against the point or the "
                    "region each one describes.")
        w.heading(2, "Notes")
        w.bullets(lines)

    patch.setattr(_entry_page, "_validity", planted)


def plant_validity_lines_given_an_operating_point_prefix(patch: pytest.MonkeyPatch) -> None:
    """Every line printed under the operating point it does not state, ahead of its own words."""
    real = _entry_page._validity

    def planted(w: Any, record: Record) -> None:
        prefixed = ["operating point: " + line for line in strings(class_var(record, "validity", []))]
        real(w, with_class_vars(record, validity=prefixed))

    patch.setattr(_entry_page, "_validity", planted)


def plant_a_glance_that_counts_declared_lines_as_plural_of_one(patch: pytest.MonkeyPatch) -> None:
    """The count of declared lines with the plural of any count: ``1 declared lines``."""
    patch.setattr(_entry_page, "plural", lambda count, noun: f"{count} {noun}s")


def plant_an_entry_page_without_what_the_family_names(patch: pytest.MonkeyPatch) -> None:
    """``Gravity`` as it was declared before the review: its values known only to ``plan``, so its page carries none."""
    real = _build.export

    def planted(not_loaded: Mapping[str, str]) -> dict[str, Any]:
        data = real(not_loaded)
        data["entries"] = [with_class_vars(record, values={}) if record["name"] == "gravity" else record
                           for record in data["entries"]]
        return data

    patch.setattr(_build, "export", planted)


def plant_a_page_that_hides_what_plan_adds(patch: pytest.MonkeyPatch) -> None:
    """The line that says what the entry's own ``plan`` adds is not in its tables, left out."""
    patch.setattr(_entry_page, "PLAN_ADDS", "")


def plant_interface_names_that_are_not_linked(patch: pytest.MonkeyPatch) -> None:
    """The name of an interface printed as plain code, as the entry page did before the review."""
    patch.setattr(_entry_page, "_interface", lambda path, name: _markdown.code(name))


def plant_targets_that_are_not_linked(patch: pytest.MonkeyPatch) -> None:
    """The target of a delivery printed as plain code, as the entry page did before the review."""
    patch.setattr(_entry_page, "_target", lambda path, name: _markdown.code(name))


def plant_a_vocabulary_that_names_no_publisher(patch: pytest.MonkeyPatch) -> None:
    """The vocabulary page with its "Published by" cell saying none registered for every name."""
    patch.setattr(_reference_pages, "_published_by", lambda held, records: "none registered")


def plant_a_title_left_a_first_level_heading(patch: pytest.MonkeyPatch) -> None:
    """A hand-written source inlined with its title as it stands: a second page title, and an empty section above it."""
    patch.setattr(_build, "demote_title", lambda text: text)


def hand_written_text_naming_a_swimmers_word(patch: pytest.MonkeyPatch) -> None:
    """Hand-written text for ``Magnetic.Torque`` that uses the words its subject needs ("in a rotating field"): authored
    prose, which the scan of what the generators write must not read."""
    real = _build._hand_written
    text = ("# The torque of a field on a moment\n\n**What it models.** tau = m x B: it turns the body toward the "
            "field, and in a rotating field it drives the body's rotation.\n")

    def planted(docs: Path, ids: Any, held: Any, problems: list[str]) -> tuple[dict[str, str], int]:
        sources, count = real(docs, ids, held, problems)
        if docs.is_dir():
            sources = {**sources, "entries/interaction/magnetic.torque.md": _build.demote_title(text)}
        return sources, len(sources)

    patch.setattr(_build, "_hand_written", planted)


#: Each plant, the sets whose check must name it, and the text of the problem it must name.
PLANTS = [
    (plant_a_rung_section, ["Magnetic.Force"], "has no rungs, and its page holds a rung section"),
    (plant_a_missing_rung_section, ["Drag"], "has rungs, and its page has no"),
    (plant_a_comparison_of_complementary_laws, ["Magnetic.Force"], "family page compares its members, which cannot"),
    (plant_a_comparison_in_a_table_of_another_header, ["Magnetic.Force"],
     "family page compares its members, which cannot"),
    (plant_a_helper_that_says_every_family_is_swappable, ["Magnetic.Force"], "the glance says the family is compared"),
    (plant_a_glance_that_says_compared, ["Magnetic.Force"], "the glance says the family is compared"),
    (plant_a_glance_count_one_too_many, ["Magnetic.Force", "Drag"], "the glance counts"),
    (plant_a_one_member_family_called_not_alternatives, ["Magnetic.Force"],
     "calls the one member of the family not alternatives"),
    (plant_a_template_that_requires_a_helix, ["Magnetic.Force", "Drag"], "the build raised KeyError"),
    (plant_a_template_that_prints_a_helix, ["Magnetic.Force"], "a word only a swimmer fills"),
    (plant_a_section_of_a_driven_swimmer_on_every_page, ["Magnetic.Force"], "the page of the data-only entry"),
    (plant_a_point_only_validity_sentence, ["Magnetic.Force", "Drag"], "does not say a condition is a point or a"),
    (plant_a_validity_line_clipped_to_its_point, ["Magnetic.Force", "Drag"], "a validity line is not printed whole"),
    (plant_validity_lines_under_another_heading, ["Magnetic.Force", "Drag"], "a validity line is not printed whole"),
    (plant_validity_lines_given_an_operating_point_prefix, ["Magnetic.Force", "Drag"],
     "prints a line the entry does not declare"),
    (plant_a_glance_that_counts_declared_lines_as_plural_of_one, ["Magnetic.Force"],
     "the glance does not say 'Validity: 1 declared line'"),
    (plant_an_entry_page_without_what_the_family_names, ["Magnetic.Force"], "does not carry mass"),
    (plant_a_page_that_hides_what_plan_adds, ["Magnetic.Force", "Drag"], "writes its own plan"),
    (plant_interface_names_that_are_not_linked, ["Magnetic.Force", "Drag"], "is not linked to the vocabulary page"),
    (plant_targets_that_are_not_linked, ["Magnetic.Force", "Drag"], "is not linked to the load mechanism page"),
    (plant_a_vocabulary_that_names_no_publisher, ["Magnetic.Force"], "does not say that point_dipole publishes"),
    (plant_a_title_left_a_first_level_heading, ["Magnetic.Force"], "has an empty section"),
]
PLANTS_BY_NAME = {plant.__name__: plant for plant, _, _ in PLANTS}
#: Controls that must pass: built as a plant is, and named by nothing.
CONTROLS = {hand_written_text_naming_a_swimmers_word.__name__: hand_written_text_naming_a_swimmers_word}


@pytest.mark.parametrize("plant, caught_on, named", PLANTS, ids=[plant.__name__ for plant, _, _ in PLANTS])
def test_each_plant_is_named_by_the_check_over_the_pages_of_the_set_it_serves(plant, caught_on, named):
    """The rate of detection of each of these over 20 runs is recorded in the pull request."""
    found = pilot_in_a_fresh_process(plant.__name__)
    for name in caught_on:
        assert any(named in problem for problem in found[name]), (plant.__name__, name, found[name])


def test_hand_written_text_that_names_a_swimmers_word_is_not_named():
    """The scan reads what the generators wrote: a source for ``Magnetic.Torque`` may say "in a rotating field"."""
    assert pilot_in_a_fresh_process(hand_written_text_naming_a_swimmers_word.__name__) == {
        "Magnetic.Force": [], "Drag": []}


def test_a_generator_string_that_names_a_swimmers_field_is_named_and_the_real_ones_are_not():
    sources = generator_sources()
    assert source_problems(sources) == []
    sources["_entry_page.py"] += '\nTEMPLATE = "Rotation rate: {rotation_rate}"\n'
    (problem,) = source_problems(sources)
    assert problem.startswith("_entry_page.py holds the string 'Rotation rate: {rotation_rate}'")
