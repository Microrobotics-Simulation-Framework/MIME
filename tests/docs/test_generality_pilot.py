"""The generality pilot: the page generators serve a body that does not swim, with no new page type.

Two sets are built and read. The ``Magnetic.Force`` set is a gradient-pulled magnet's: ``Magnetic.Force``,
``PointDipoleField``, ``Magnetic.Torque`` and ``Gravity``, their nodes, the magnetic and gravity families and the load
mechanism. ``Drag`` is the second, and the one that does swim: its entry, family and node pages, read the same way. What
the pages are held to:

- an entry without rungs has no rung section, and one with rungs has one per rung;
- a family page compares only members that can replace one another (its suite holds a swap case), and the glance of an
  entry says "compared on its page" only where the family page does;
- no entry, node or family page of the ``Magnetic.Force`` set holds a word only a swimmer fills, none of the generators'
  strings names one, and no section of any page of either set is a heading with nothing under it;
- a validity line is printed whole, under a sentence that says a condition is a point or a region;
- every node an entry of a set builds has a page that is its equations and symbols or says it is undocumented, and
  names the entry; every mechanism a set delivers into lists the entry.

Each plant breaks one generator, in the process that builds, at the branch that serves the rule, and the check over the built pages
must name it: a plant that the check passes would be a control that cannot fail. The pages are built from the checkout
in a fresh process, as the build reads the registry: a test of another file may have registered entries of its own,
and the pages of this process would hold them."""

import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from checkout import ROOT

from mime.docs import _build, _entry_page, _family_page, _markdown, _node_page, _reference_pages
from mime.docs._build import Built, build_pages
from mime.docs._context import (
    Record,
    alternatives,
    builds_of,
    class_var,
    entry_path,
    family_path,
    mechanism_path,
    node_path,
)
from mime.docs._export import registry_export
from mime.docs._markdown import mapping, strings
from mime.testing import CaseKind, registered_families
from mime.vocabulary import lookup

#: Words only a swimmer fills: a template that prints or requires one is keyed on a swimmer.
SWIMMER = re.compile(r"helix|helical|swim|tube|vessel|rotation|rotating", re.IGNORECASE)
GENERATORS = (_entry_page, _family_page, _markdown, _node_page, _reference_pages)
COMPARISON_TABLE = "| Member |"


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
    (("interaction", "magnetic.force"), ("field_provider", "point_dipole_field"), ("interaction", "magnetic.torque"),
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


def set_problems(built: Built, records: Sequence[Record], held: PilotSet) -> list[str]:
    """What the pages of ``held`` get wrong against the generality rule."""
    problems: list[str] = []
    families = {family.name: family for family in registered_families()}
    by_key = {(str(record["kind"]).replace(" ", "_"), str(record["name"])): record for record in records}

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
        validity = strings(class_var(record, "validity", []))
        section = sections(text).get("Validity", "")
        if not validity:
            if "none declared" not in section:
                problems.append(f"{who} declares no validity line, and its page does not say so")
        else:
            intro = section.strip().splitlines()[0] if section.strip() else ""
            if not ("point" in intro and "region" in intro):
                problems.append(f"{who}: the validity section's sentence does not say a condition is a point or a "
                                f"region: {intro!r}")
            for line in validity:
                if line not in text:
                    problems.append(f"{who}: a validity line is not printed whole: {line[:60]!r}")
        family = class_var(record, "family")
        if family is not None:
            glance = sections(text).get("At a glance", "")
            fam_text = text_of(family_path(str(family)))
            if fam_text is not None:
                compared = COMPARISON_TABLE in sections(fam_text).get("Comparison", "")
                says = "; compared on its page" in glance
                if says != compared:
                    problems.append(f"{who}: the glance says the family is {'' if says else 'not '}compared "
                                    f"on its page, and the family page {'does' if compared else 'does not'}")
            if not rungs and "built rung" in glance:
                problems.append(f"{who} has no rungs, and its glance counts built rungs")
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
    for name in sorted({str(class_var(by_key[key], "family")) for key in held.entries
                        if class_var(by_key[key], "family") is not None}):
        if (text := text_of(family_path(name))) is None:
            continue
        pages[family_path(name)] = text
        members = [record for record in records if class_var(record, "family") == name]
        comparison = sections(text).get("Comparison", "")
        if not alternatives(families[name]) or len(members) < 2:
            if COMPARISON_TABLE in comparison:
                problems.append(f"{held.name}: the {name} family page compares its members, which cannot replace one "
                                f"another" if not alternatives(families[name]) else
                                f"{held.name}: the {name} family page compares fewer than two members")
        elif COMPARISON_TABLE not in comparison:
            problems.append(f"{held.name}: the {name} family page does not compare members that can replace one another")
        if not alternatives(families[name]) and len(members) > 1 and "not alternatives" not in comparison:
            problems.append(f"{held.name}: the {name} family page does not say its members are not alternatives")
        listed = sections(text).get("Members", "")
        problems += [f"{held.name}: the {name} family page does not list {record['name']}" for record in members
                     if f"]({'../' + entry_path(record)})" not in listed]
    for path, text in sorted(pages.items()):
        problems += [f"{held.name}: {path} has an empty section {line!r}" for line in empty_sections(text)]
        if held.swimmer_free and (found := SWIMMER.search(text)):
            problems.append(f"{held.name}: {path} holds {found.group()!r}, a word only a swimmer fills")
    return problems


def pilot_problems(records: Sequence[Record], root: Path = ROOT) -> dict[str, list[str]]:
    """Build the pages and read each set; a build that raises is a problem of every set."""
    try:
        built = build_pages(root)
    except Exception as error:  # a template that needs a field the entry lacks is the failure under test
        return {held.name: [f"{held.name}: the build raised {type(error).__name__}: {error}"] for held in SETS}
    return {held.name: [*(f"{held.name}: the build failed: {problem}" for problem in built.problems),
                        *set_problems(built, records, held)] for held in SETS}


def main(plant: str | None) -> None:
    """The entry of the fresh process: build the pages, with the plant named (or none), and print what each set got
    wrong as JSON."""
    records = list(registry_export()["entries"])
    with pytest.MonkeyPatch.context() as patch:
        if plant is not None:
            PLANTS[plant][0](patch)
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
    by_key = {(str(record["kind"]).replace(" ", "_"), str(record["name"])): record for record in records}
    for held in SETS:
        assert all(key in by_key for key in held.entries), held.name
    assert [class_var(by_key[key], "rungs", []) for key in MAGNETIC_FORCE.entries] == [[]] * 4
    assert class_var(by_key[DRAG.entries[0]], "rungs") == ["surrogate"]
    assert [class_var(by_key[key], "family") for key in MAGNETIC_FORCE.entries] == ["magnetic", None, "magnetic",
                                                                                      "gravity"]
    assert any("outside" in line for line in strings(class_var(by_key[MAGNETIC_FORCE.entries[1]], "validity")))


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


def plant_a_glance_that_says_compared(patch: pytest.MonkeyPatch) -> None:
    """The glance's test of whether the family page compares answered yes for every family."""
    patch.setattr(_entry_page, "alternatives", lambda family: True)


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


def plant_a_title_left_a_first_level_heading(patch: pytest.MonkeyPatch) -> None:
    """A hand-written source inlined with its title as it stands: a second page title, and an empty section above it."""
    patch.setattr(_build, "demote_title", lambda text: text)


#: Each plant, the sets whose check must name it, and the text of the problem it must name.
PLANTS = {plant.__name__: (plant, caught_on, named) for plant, caught_on, named in [
    (plant_a_rung_section, ["Magnetic.Force"], "has no rungs, and its page holds a rung section"),
    (plant_a_missing_rung_section, ["Drag"], "has rungs, and its page has no"),
    (plant_a_comparison_of_complementary_laws, ["Magnetic.Force"], "family page compares its members, which cannot"),
    (plant_a_glance_that_says_compared, ["Magnetic.Force"], "the glance says the family is compared"),
    (plant_a_template_that_requires_a_helix, ["Magnetic.Force", "Drag"], "the build raised KeyError"),
    (plant_a_template_that_prints_a_helix, ["Magnetic.Force"], "a word only a swimmer fills"),
    (plant_a_point_only_validity_sentence, ["Magnetic.Force", "Drag"], "does not say a condition is a point or a"),
    (plant_a_validity_line_clipped_to_its_point, ["Magnetic.Force", "Drag"], "a validity line is not printed whole"),
    (plant_a_title_left_a_first_level_heading, ["Magnetic.Force"], "has an empty section"),
]}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_each_plant_is_named_by_the_check_over_the_pages_of_the_set_it_serves(plant):
    """The rate of detection of each of these over 20 runs is recorded in the pull request."""
    _, caught_on, named = PLANTS[plant]
    found = pilot_in_a_fresh_process(plant)
    for name in caught_on:
        assert any(named in problem for problem in found[name]), (plant, name, found[name])


def test_a_generator_string_that_names_a_swimmers_field_is_named_and_the_real_ones_are_not():
    sources = generator_sources()
    assert source_problems(sources) == []
    sources["_entry_page.py"] += '\nTEMPLATE = "Rotation rate: {rotation_rate}"\n'
    (problem,) = source_problems(sources)
    assert problem.startswith("_entry_page.py holds the string 'Rotation rate: {rotation_rate}'")
