"""The page generators, through the real command. ``mime docs build`` runs in a fresh process and writes the pages; the
oracle is the registry export (``mime registry --json``, also a fresh process), so a page that is missing, or an entry
that has no page, is a difference between two readings of the same registry.

What the pages are held to:

- every entry of the export has a page, the lab fixture's among them, by the namespace rules (its hand-written half
  is counted, not failed; its generated half is the control); every class variable the export carries is on its page;
  every node an entry builds has a page; every family has a page, and every relative link resolves to a page;
- the family page of ``ToyDrag``, which has two built rungs, prints the contract, the members and a comparison no cell
  of which is blank; the hydrodynamic family prints its contract and its rung list and no comparison, as it has one;
- the "at a glance" block links the other homes (the node page's equation, the family page) and copies none;
- no page holds an absolute path of the machine that built it;
- a hand-written source is inlined; one that no longer resolves fails the build, naming the file, and no page is written
  (W8b), the passing arm being the source that resolves;
- W7: an empty equations, assumptions or references field prints ``none declared`` and is counted, and a filled one does
  not;
- a display equation appears on one page."""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from checkout import ROOT
from planted_nodes import META, PlantedNode, planted

from mime.docs._build import (
    Built,
    absolute_path_problems,
    equation_problems,
    repository_relative,
    strip_front_matter,
)
from mime.docs._context import Context
from mime.docs._entry_page import entry_page
from mime.docs._export import registry_export
from mime.docs._markdown import Page
from mime.docs._node_page import node_page


def build(root: Path, out: Path | None = None, *extra: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "mime.cli", "docs", "build", "--root", str(root), *extra]
    return subprocess.run([*command, *(["--out", str(out)] if out is not None else [])], capture_output=True,
                          text=True, check=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pages of the checkout, written by the real command into a directory of its own."""
    out = tmp_path_factory.mktemp("built")
    done = build(ROOT, out)
    assert done.returncode == 0, done.stdout + done.stderr
    return out


@pytest.fixture(scope="module")
def export() -> dict:
    return registry_export()


def text_of(built: Path, page: str) -> str:
    return (built / page).read_text(encoding="utf-8")


def page_of(entry: dict) -> str:
    return f"pages/entries/{entry['kind'].replace(' ', '_')}/{entry['name']}.md"


# -- the oracle: the registry export ----------------------------------------------------------------------------------

def test_every_entry_of_the_registry_export_has_a_page_naming_its_version(built, export):
    entries = export["entries"]
    assert len(entries) >= 20, "the export holds too few entries to be an oracle"
    missing = [page_of(entry) for entry in entries if not (built / page_of(entry)).is_file()]
    assert missing == []
    for entry in entries:
        page = text_of(built, page_of(entry))
        assert f"`{entry['name']}` at version `{entry['version']}`" in page, page_of(entry)


def test_every_page_the_build_wrote_is_an_entry_a_node_a_family_a_mechanism_or_a_reference_page(built, export):
    names = {page_of(entry) for entry in export["entries"]}
    found = {path.relative_to(built).as_posix() for path in built.rglob("*.md")}
    entries = {path for path in found if path.startswith("pages/entries/")}
    assert entries == names
    rest = {"/".join(path.split("/")[:2]) if path.count("/") > 1 else path for path in found - entries}
    assert rest == {"index.md", "pages/nodes", "pages/families", "pages/mechanisms", "pages/vocabulary.md",
                    "pages/concepts.md"}


def test_every_class_variable_the_export_carries_is_on_its_entrys_page(built, export):
    """The export is complete (a test fails when an entry declares a class variable it does not carry), so a page that
    names each one omits nothing the entry declares at class level."""
    missing = [f"{entry['name']}: {name}" for entry in export["entries"] for name in entry["class_vars"]
               if f"`{name}`" not in text_of(built, page_of(entry)) and name not in {
                   "name", "version", "references", "validity", "gradient", "differentiability", "builds", "roles",
                   "reads", "delivers", "values", "rungs", "reserved_rungs", "support", "family"}]
    assert missing == []


def test_every_node_an_entry_builds_has_a_page_and_every_family_has_one(built, export):
    nodes = {dotted.rsplit(".", 1)[1] for entry in export["entries"]
             for built_nodes in entry["class_vars"]["builds"]["value"].values() for dotted in built_nodes}
    assert len(nodes) >= 15
    written = {path.stem for path in (built / "pages" / "nodes").glob("*.md")}
    assert written == nodes
    families = {entry["class_vars"]["family"]["value"] for entry in export["entries"]
                if entry["class_vars"].get("family", {}).get("value")}
    assert families <= {path.stem for path in (built / "pages" / "families").glob("*.md")}


def test_every_relative_link_of_every_page_resolves_to_a_page(built):
    pattern = re.compile(r"\]\((?!https?:|#)([^)#\s]+)")
    broken: list[str] = []
    for page in sorted(built.rglob("*.md")):
        for target in pattern.findall(page.read_text(encoding="utf-8")):
            if not (page.parent / target).resolve().is_file():
                broken.append(f"{page.relative_to(built)} -> {target}")
    assert broken == []


def test_no_page_holds_an_absolute_path_of_the_machine_that_built_it(built):
    text = "\n".join(path.read_text(encoding="utf-8") for path in built.rglob("*") if path.is_file())
    assert str(ROOT) not in text and str(built) not in text
    assert not re.search(r"/(?:home|Users|tmp|runner)/", text)


# -- the lab fixture: the control for the generated half ---------------------------------------------------------------

def test_the_lab_fixture_yields_a_page_by_the_namespace_rules(built, export):
    """The fixture ships no page source and no ``NodeMeta``: its entry still has a page, which says the hand-written
    part is not written and counts it, and its node's page says it is undocumented. The oracle is the export."""
    (record,) = [entry for entry in export["entries"] if entry["namespace"] == "labx"]
    page = text_of(built, page_of(record))
    assert record["name"] == "labx.nudge" and page.startswith("# interaction labx.nudge\n")
    assert "No hand-written part" in page
    assert "- Derivative: `exact`" in page and "| `force` |" in page
    (node,) = [dotted.rsplit(".", 1)[1] for dotted in record["class_vars"]["builds"]["value"][""]]
    assert node == "NudgeNode"
    assert "undocumented" in text_of(built, f"pages/nodes/{node}.md")


def test_a_lab_entrys_references_are_printed_as_written_and_never_cited_against_mimes_bibliography(built):
    """A lab's bibliography is its own: its keys print as code, and a MIME entry's as citations."""
    assert "[@CoxBrenner1967]" in text_of(built, "pages/entries/interaction/drag.md")
    assert not re.search(r"\[@", text_of(built, "pages/entries/interaction/labx.nudge.md"))


# -- the family pages --------------------------------------------------------------------------------------------------

def test_the_toy_family_page_prints_its_contract_its_members_and_a_comparison_with_no_blank_cell(built):
    page = text_of(built, "pages/families/toy.md")
    assert "## Contract" in page and "Sign rule: toy.body.load on the body opposes toy.body.velocity" in page
    members = re.search(r"## Members\n\n(.*?)\n\n", page, re.DOTALL)
    assert members and "`linear`" in members.group(1) and "`quadratic`" in members.group(1)
    comparison = page.split("## Comparison", 1)[1]
    rows = [line for line in comparison.splitlines() if line.startswith("| [")]
    assert len(rows) == 2
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert all(cells), f"a blank cell in {row!r}"
        assert cells[2:4] == ["not measured", "not measured"] and cells[5] == "not measured"


def test_the_hydrodynamic_family_page_prints_its_contract_and_rung_list_and_no_comparison_until_a_second_rung(built):
    page = text_of(built, "pages/families/hydrodynamic.md")
    assert "the resistance dissipates" in page and "## Members" in page
    assert "[`surrogate`](../entries/interaction/drag.md#rung-surrogate)" in page
    assert "A comparison starts when the family has two built rungs" in page
    assert "| Member |" not in page


def test_a_family_whose_members_are_not_alternatives_is_not_compared(built):
    assert "are not alternatives to one another" in text_of(built, "pages/families/magnetic.md") or \
        "A comparison starts" in text_of(built, "pages/families/magnetic.md")


# -- the entry page ----------------------------------------------------------------------------------------------------

def test_the_at_a_glance_block_links_the_other_homes_and_copies_none_of_them(built):
    page = text_of(built, "pages/entries/interaction/drag.md")
    glance = page.split("## At a glance", 1)[1].split("\n## ", 1)[0]
    assert "[`hydrodynamic`](../../families/hydrodynamic.md)" in glance
    assert "(../../nodes/SurrogateDragNode.md#governing-equations)" in glance
    assert "$$" not in page


def test_the_drag_rung_is_a_section_with_its_nodes_and_its_support(built):
    page = text_of(built, "pages/entries/interaction/drag.md")
    section = page.split("### Rung surrogate", 1)[1].split("\n## ", 1)[0]
    assert "[`SurrogateDragNode`](../../nodes/SurrogateDragNode.md)" in section
    assert "a rigid body whose dynamics takes a resistance" in section


def test_a_rung_no_round_trip_case_builds_prints_declared_unchecked(built):
    page = text_of(built, "pages/entries/interaction/toy.twist.md")
    assert "declared, unchecked" in page


def test_the_vocabulary_marks_a_reserved_entry_and_lists_every_name(built):
    from mime.vocabulary import names

    page = text_of(built, "pages/vocabulary.md")
    assert all(f"`{name}`" in page for name in names())
    assert "body.resistance" in page and "Namespace toy" in page


def test_the_index_has_a_front_door_by_goal_and_links_the_concepts_note(built):
    page = text_of(built, "index.md")
    assert "## By goal" in page and "What can act on a body?" in page and "Which model, when there are several?" in page
    assert "(pages/concepts.md)" in page and "(pages/vocabulary.md)" in page
    assert "(pages/entries/interaction/drag.md)" in page


# -- the build ---------------------------------------------------------------------------------------------------------

def test_the_default_output_is_the_untracked_docs_build_directory_of_the_checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "docs").mkdir(parents=True)
    done = build(root)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (root / "docs" / "_build" / "index.md").is_file() and (root / "docs" / "_build" / "pages").is_dir()
    assert "wrote" in done.stdout and str(root) not in done.stdout


def test_generated_pages_are_never_committed_the_output_directory_is_ignored():
    assert "/docs/_build/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    done = subprocess.run(["git", "check-ignore", "-q", "docs/_build/pages/entries/interaction/drag.md"], cwd=ROOT,
                          check=False)
    assert done.returncode == 0


def test_a_build_clears_only_a_pages_directory_an_earlier_build_made(tmp_path):
    out = tmp_path / "out"
    assert build(ROOT, out).returncode == 0
    (out / "pages" / "stale.md").write_text("an earlier build's page\n", encoding="utf-8")
    assert build(ROOT, out).returncode == 0
    assert not (out / "pages" / "stale.md").exists()
    (out / "pages" / ".mime-docs-build").unlink()
    (out / "pages" / "mine.md").write_text("the user's file\n", encoding="utf-8")
    refused = build(ROOT, out)
    assert refused.returncode != 0 and (out / "pages" / "mine.md").read_text(encoding="utf-8") == "the user's file\n"


def source(root: Path, relative: str, text: str) -> Path:
    path = root / "docs" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_hand_written_source_that_resolves_is_inlined_without_its_front_matter(tmp_path):
    root = tmp_path / "checkout"
    source(root, "entries/interaction/toy.drag.md", '---\ndescribes: "0.2.0"\n---\n\nIt resists a toy body.\n')
    source(root, "families/toy.md", "Choose by cost.\n")
    source(root, "mechanisms/load.md", "Loads add.\n")
    done = build(root, tmp_path / "out")
    assert done.returncode == 0, done.stdout + done.stderr
    page = (tmp_path / "out" / "pages" / "entries" / "interaction" / "toy.drag.md").read_text(encoding="utf-8")
    assert "It resists a toy body." in page and "describes" not in page and "No hand-written part" not in page
    assert "Choose by cost." in (tmp_path / "out" / "pages" / "families" / "toy.md").read_text(encoding="utf-8")
    assert "Loads add." in (tmp_path / "out" / "pages" / "mechanisms" / "load.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(("relative", "text", "named"), [
    ("entries/interaction/no_such_entry.md", "x\n", "names the interaction no_such_entry, which no entry registers"),
    ("entries/interaction/toy.drag.md", '---\ndescribes: "0.1.0"\n---\n', "describes toy.drag 0.1.0, which is not "
                                                                         "registered"),
    ("entries/interaction/toy.drag.md", "no front matter\n", "names no version it describes"),
    ("mechanisms/no_such_kind.md", "x\n", "names the target kind no_such_kind"),
    ("nodes/NoSuchNode.md", "x\n", "names the node NoSuchNode, which no entry builds"),
    ("families/no_such_family.md", "x\n", "names the family no_such_family, which is not registered"),
    ("entries/interaction/toy.drag.txt", "x\n", "is not a page source"),
], ids=["an-entry-that-is-gone", "a-version-that-is-gone", "no-version-named", "a-target-kind-that-is-gone",
        "a-node-that-is-gone", "a-family-that-is-gone", "not-a-page-source"])
def test_w8b_a_stale_source_fails_the_build_naming_the_file_and_writes_no_page(tmp_path, relative, text, named):
    """A source that no longer resolves must not be skipped, leaving a page missing and CI green: the build exits 1,
    names the file, and writes nothing, not even the pages that were fine."""
    root = tmp_path / "checkout"
    file = source(root, relative, text)
    out = tmp_path / "out"
    done = build(root, out)
    assert done.returncode == 1, done.stdout + done.stderr
    failure = [line for line in done.stdout.splitlines() if line.startswith("FAIL: ")]
    assert failure and all(file.name in line for line in failure) and any(named in line for line in failure)
    assert "no page written" in done.stdout and not out.exists()


# -- helpers that need no process --------------------------------------------------------------------------------------

def test_a_runtime_path_is_made_repository_relative_when_it_is_written(tmp_path):
    root = tmp_path / "checkout"
    text = f"refused at {root}/docs/entries/x.md and in {root}"
    assert repository_relative(text, root) == "refused at docs/entries/x.md and in ."


def test_a_page_holding_an_absolute_path_fails_the_build_and_a_relative_one_does_not():
    pages = {"pages/a.md": Page("a.md", "refused at " + "/" + "home/runner/work/MIME/docs/x.md\n"),
             "pages/b.md": Page("b.md", "refused at docs/x.md, see /entries and docs/tmp/y\n")}
    (problem,) = absolute_path_problems(pages)
    assert problem.startswith("pages/a.md: carries an absolute path of the machine that built it (/home/r")


def test_strip_front_matter_removes_one_block_at_the_top_only():
    assert strip_front_matter("---\na: 1\n---\n\nBody\n---\nnot front matter\n") == "\nBody\n---\nnot front matter\n"
    assert strip_front_matter("Body\n---\na: 1\n---\n") == "Body\n---\na: 1\n---\n"


def test_the_build_summary_counts_the_hatches_it_printed(tmp_path):
    done = build(ROOT, tmp_path / "out")
    hatches = next(line for line in done.stdout.splitlines() if line.startswith("hatches printed: "))
    assert "none declared" in hatches and "undocumented" in hatches and "no hand-written part" in hatches


# -- W7: none declared -------------------------------------------------------------------------------------------------

EMPTY = Context((), {}, {}, {})


def hatched(page: Page, kind: str = "none declared") -> list[str]:
    return sorted(hatch.item for hatch in page.hatches if hatch.kind == kind)


def test_w7_an_empty_equations_assumptions_or_references_field_prints_none_declared_and_is_counted():
    """PlantedNode has equations and neither assumptions nor references; the plants empty the equations and fill the
    other two, so that each field is the one that fails in turn."""
    page = node_page(PlantedNode, "planted_nodes.PlantedNode", EMPTY)
    assert hatched(page) == ["node PlantedNode assumptions", "node PlantedNode references"]
    assert page.text.count("none declared") >= 2
    blank = planted("Blank", meta=replace(META, governing_equations="", assumptions=("a",), references=_refs()))
    assert hatched(node_page(blank, "planted_nodes.Blank", EMPTY)) == ["node Blank equations"]
    bare = planted("Bare", meta=replace(META, governing_equations="   \n", assumptions=(), references=()))
    assert hatched(node_page(bare, "planted_nodes.Bare", EMPTY)) == [
        "node Bare assumptions", "node Bare equations", "node Bare references"]


def _refs():
    from maddening.core.compliance.metadata import Reference

    return (Reference("Sola2017", "quaternion kinematics"),)


def test_w7_the_passing_arm_a_node_with_all_three_prints_no_none_declared_for_them():
    full = planted("Full", meta=replace(META, assumptions=("small deflection",), references=_refs()))
    page = node_page(full, "planted_nodes.Full", EMPTY)
    assert hatched(page) == []
    assert "small deflection" in page.text and "[@Sola2017]" in page.text and "$$" in page.text
    assert "none declared" not in page.text.split("## Validated regimes")[0].split("## Assumptions")[1].split(
        "## Limitations")[0]


def entry(references: object) -> dict:
    return {"kind": "interaction", "name": "planted.law", "version": "0.1.0", "namespace": "mime", "defined_in": "x.Y",
            "fields": {}, "settings": {},
            "class_vars": {"references": {"type": "t", "declared_by": "x", "default": [], "value": references},
                           "differentiability": {"type": "t", "declared_by": "x", "default": None, "value": "exact"}}}


def test_w7_an_entrys_empty_references_print_none_declared_and_a_filled_or_unestablished_field_does_not():
    assert hatched(entry_page(entry([]), EMPTY)) == ["interaction planted.law 0.1.0 references"]
    assert "none declared" in entry_page(entry([]), EMPTY).text
    filled = entry_page(entry(["Sola2017"]), EMPTY)
    assert hatched(filled) == [] and "[@Sola2017]" in filled.text
    unknown = entry_page(entry({"not_established": True}), EMPTY)
    assert hatched(unknown) == [] and "not established" in unknown.text


def test_w7_the_none_declared_hatches_of_a_build_are_counted_in_its_summary():
    built = Built(pages={"p": Page("p", "x", hatches=tuple(node_page(PlantedNode, "m.P", EMPTY).hatches))})
    assert built.hatches()["none declared"] == 2 and "none declared 2" in built.summary()[1]


# -- one equation, one page --------------------------------------------------------------------------------------------

def test_a_display_equation_on_two_pages_fails_naming_both_and_one_on_each_alone_passes():
    equation = "$$\n\\mathbf{F} = -c \\mathbf{v}\n$$\n"
    pages = {"pages/nodes/A.md": Page("A.md", equation), "pages/nodes/B.md": Page("B.md", equation)}
    (problem,) = equation_problems(pages)
    assert "pages/nodes/A.md" in problem and "pages/nodes/B.md" in problem
    reflowed = {"pages/nodes/A.md": Page("A.md", equation),
                "pages/nodes/B.md": Page("B.md", "$$\\mathbf{F}=-c\\,\\mathbf{v}$$\n".replace("\\,", " "))}
    assert len(equation_problems(reflowed)) == 1, "the same LaTeX with other whitespace is the same equation"
    different = {"pages/nodes/A.md": Page("A.md", equation),
                 "pages/nodes/B.md": Page("B.md", "$$\n\\mathbf{F} = -k \\mathbf{x}\n$$\n")}
    assert equation_problems(different) == []
    assert equation_problems({"p": Page("p", equation + "again\n" + equation)}) != []


def test_the_pages_hold_one_copy_of_each_equation(built):
    texts = {path.relative_to(built).as_posix(): Page(path.name, path.read_text(encoding="utf-8"))
             for path in built.rglob("*.md")}
    assert equation_problems(texts) == []


def test_json_of_the_export_has_the_keys_the_pages_read(export):
    """The pages read the export's shape; this names it, so a change of the export fails here and not in a page."""
    entry_ = next(entry for entry in export["entries"] if entry["name"] == "drag")
    assert {"kind", "name", "version", "namespace", "defined_in", "class_vars", "fields", "settings"} <= set(entry_)
    assert json.dumps(export, sort_keys=True)
    assert shutil.which("git")


# -- the gradient record: every field it has is printed, so a field 7s adds (``switches``) is rendered with no edit ----

def gradient_entry(**fields: object) -> dict:
    record = entry([])
    record["class_vars"]["differentiability"]["value"] = "piecewise"
    record["class_vars"]["gradient"] = {"type": "t", "declared_by": "x", "default": None,
                                        "value": {"where": "w", "reason": "r", "route": "q", **fields}}
    return record


def test_every_field_of_a_gradient_record_is_printed_so_a_field_added_later_is_rendered_with_no_edit():
    """``GradientRecord.switches`` (3.19 item 4) lands with the switch foundation. This page prints whatever the record
    has: a record with a ``switches`` field prints it, and the export's completeness test already makes the field part of
    the export."""
    page = entry_page(gradient_entry(switches={"clamp": {"kind": "clamp"}}), EMPTY).text
    assert "- **Switches**: `{\"clamp\": {\"kind\": \"clamp\"}}`" in page
    assert "- **Where**: w" in page and "- **Route to exact**: q" in page


def test_the_real_gradient_records_are_printed_field_for_field(built, export):
    titles = {"where": "Where", "reason": "Why", "route": "Route to exact"}
    for record in export["entries"]:
        held = record["class_vars"]["gradient"]["value"]
        if held:
            page = text_of(built, page_of(record))
            wanted = [titles.get(key, key.replace("_", " ").capitalize()) for key in held]
            assert all(f"- **{title}**:" in page for title in wanted), record["name"]
