"""The page generators, through the real command. ``mime docs build`` runs in a fresh process and writes the pages; the
oracle is the registry export (``mime registry --json``, also a fresh process), so a page that is missing, or an entry
that has no page, is a difference between two readings of the same registry.

What the pages are held to:

- every entry of the export has a page, the lab fixture's among them, by the namespace rules (its hand-written half
  is counted, not failed; its generated half is the control); every class variable the export carries is on its page;
  every node an entry builds has a page; every family has a page, and every relative link resolves to a page;
- the family page of ``ToyDrag``, which has two built rungs, prints the contract, the members and a comparison no cell
  of which is blank; the hydrodynamic family prints its contract, its two rungs and their comparison;
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

from mime.core import switch
from mime.docs._build import (
    Built,
    absolute_path_problems,
    demote_title,
    equation_problems,
    repository_relative,
    strip_front_matter,
)
from mime.docs._context import Context
from mime.docs._entry_page import entry_page
from mime.docs._export import encode, registry_export
from mime.docs._markdown import Page, anchor, cell
from mime.docs._node_page import node_page
from mime.registry import GradientRecord


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
                    "pages/concepts.md", "pages/results.md", "pages/errors.md", "pages/errors"}


def test_every_class_variable_the_export_carries_is_on_its_entrys_page(built, export):
    """The export is complete (a test fails when an entry declares a class variable it does not carry), so a page that
    names each one omits nothing the entry declares at class level."""
    missing = [f"{entry['name']}: {name}" for entry in export["entries"] for name in entry["class_vars"]
               if f"`{name}`" not in text_of(built, page_of(entry)) and name not in {
                   "name", "version", "references", "validity", "gradient", "differentiability", "builds", "roles",
                   "reads", "delivers", "values", "publishes", "rungs", "reserved_rungs", "support", "family",
                   "rung_gradients", "callback_gradient", "outputs", "raises"}]
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


LINK = re.compile(r"\]\((?!https?:)([^)\s]*)\)")


def anchors_of(text: str) -> set[str]:
    """The ids the headings of a page give (outside a code fence), by the generator's own rule."""
    found: set[str] = set()
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (heading := re.match(r"#{1,6} (.+)", line)):
            found.add(anchor(heading.group(1)))
    return found


def broken_links(root: Path) -> list[str]:
    """Each relative link under ``root`` that names no page, or names an anchor the page it names has no heading for;
    a link to an anchor of its own page (``#rung-surrogate``) is read against the page itself."""
    broken: list[str] = []
    for page in sorted(root.rglob("*.md")):
        for url in LINK.findall(page.read_text(encoding="utf-8")):
            target, _, fragment = url.partition("#")
            home = (page.parent / target).resolve() if target else page
            if not home.is_file():
                broken.append(f"{page.relative_to(root)} -> {url}: no such page")
            elif fragment and fragment not in anchors_of(home.read_text(encoding="utf-8")):
                broken.append(f"{page.relative_to(root)} -> {url}: no heading with that anchor")
    return broken


def test_every_relative_link_of_every_page_resolves_to_a_page_and_to_a_heading(built):
    assert broken_links(built) == []
    assert sum(len(LINK.findall(path.read_text(encoding="utf-8"))) for path in built.rglob("*.md")) > 100


def test_the_link_check_fails_a_dead_anchor_and_a_dead_page_and_passes_a_live_one(tmp_path):
    """The control: the check strips nothing. The glance block's dead ``#governing-equations`` is the plant."""
    (tmp_path / "nodes").mkdir()
    (tmp_path / "nodes" / "N.md").write_text("# Node N\n\n```\n# Not a heading\n```\n\n## Built by\n", encoding="utf-8")
    page = tmp_path / "a.md"
    page.write_text("[live](nodes/N.md#built-by) [top](nodes/N.md) [own](#rung-x)\n\n### Rung x\n", encoding="utf-8")
    assert broken_links(tmp_path) == []
    page.write_text("[dead](nodes/N.md#governing-equations) [gone](nodes/M.md) [fenced](nodes/N.md#not-a-heading) "
                    "[own](#rung-y)\n\n### Rung x\n", encoding="utf-8")
    assert [problem.split(": ")[1] for problem in broken_links(tmp_path)] == [
        "no heading with that anchor", "no such page", "no heading with that anchor", "no heading with that anchor"]


def test_no_page_holds_an_absolute_path_of_the_machine_that_built_it(built):
    text = "\n".join(path.read_text(encoding="utf-8") for path in built.rglob("*") if path.is_file())
    assert str(ROOT) not in text and str(built) not in text
    assert not re.search(r"/(?:home|Users|tmp|runner)/", text)


# -- the lab fixture: the control for the generated half ---------------------------------------------------------------

def test_the_lab_fixture_yields_a_page_by_the_namespace_rules(built, export):
    """The fixture ships no page source and no ``NodeMeta``: its entry still has a page, which says the hand-written
    part is not written and counts it, and its node's page says it is undocumented. The oracle is the export. (Its
    chain segment, ``labx.gantry``, is the same case, on a dynamics.)"""
    (record,) = [entry for entry in export["entries"] if entry["namespace"] == "labx" and entry["name"] == "labx.nudge"]
    page = text_of(built, page_of(record))
    assert record["name"] == "labx.nudge" and page.startswith("# interaction labx.nudge\n")
    assert "No hand-written part" in page
    assert "- Derivative: `exact`" in page and "| `force` |" in page
    (node,) = [dotted.rsplit(".", 1)[1] for dotted in record["class_vars"]["builds"]["value"][""]]
    assert node == "NudgeNode"
    assert "undocumented" in text_of(built, f"pages/nodes/{node}.md")


def test_an_observations_declared_outputs_and_an_entrys_flags_are_on_its_page_a_labs_as_mimes(built, export):
    """Design #184 item 6: an observation's page lists its outputs from their declarations, each with its dimension,
    unit and when it has a value, and what the graph reduces it from; the lab fixture's ``labx.height`` gets the same
    table with no page source and no MIME edit. The drag's page lists the flag its surrogate raises. The oracle is the
    export's declarations; the control, an entry that declares none (``labx.nudge``), has neither section."""
    for name in ("swim_speed", "labx.height"):
        (record,) = [entry for entry in export["entries"] if entry["name"] == name]
        page = text_of(built, page_of(record))
        assert "## Outputs" in page
        for output in record["class_vars"]["outputs"]["value"]:
            when = "per step" if output["kind"] == "per_step" else "at the end"
            row = (f"| `{output['field']}` | {when} | {output['dimension']['dimension']} | "
                   f"`{output['dimension']['unit']}` |")
            assert row in page, row
    swim = text_of(built, "pages/entries/observation/swim_speed.md")
    assert "`speed` by the rule `mean`, over the steps that end after its setting `after`" in swim
    assert "`height` by the rule `integral`, over every step of the run" in text_of(
        built, "pages/entries/observation/labx.height.md")
    drag = text_of(built, "pages/entries/interaction/drag.md")
    assert "## Flags" in drag and "| `clamp_fired` | `surrogate` | its node |" in drag
    nudge = text_of(built, "pages/entries/interaction/labx.nudge.md")
    assert "## Outputs" not in nudge and "## Flags" not in nudge


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


def test_the_hydrodynamic_family_page_prints_its_contract_its_rungs_and_their_comparison(built):
    """Its second built rung, the BEM, starts the comparison: a row for each rung, no cell blank."""
    page = text_of(built, "pages/families/hydrodynamic.md")
    assert "the resistance dissipates" in page and "## Members" in page
    assert "[`surrogate`](../entries/interaction/drag.md#rung-surrogate)" in page
    assert "[`bem`](../entries/interaction/drag.md#rung-bem)" in page
    rows = [line for line in page.split("## Comparison", 1)[1].splitlines() if line.startswith("| [")]
    assert [row.split("`")[1] for row in rows] == ["drag surrogate", "drag bem"]
    assert all(all(cell.strip() for cell in row.strip("|").split("|")) for row in rows)


def test_a_family_whose_members_are_not_alternatives_is_not_compared(built):
    page = text_of(built, "pages/families/magnetic.md")
    assert "are not alternatives to one another" in page and "| Member |" not in page


# -- the entry page ----------------------------------------------------------------------------------------------------

def test_the_at_a_glance_block_links_the_other_homes_and_copies_none_of_them(built):
    page = text_of(built, "pages/entries/interaction/drag.md")
    glance = page.split("## At a glance", 1)[1].split("\n## ", 1)[0]
    assert "[`hydrodynamic`](../../families/hydrodynamic.md), with 2 built rungs among its members" in glance
    assert "(../../nodes/SurrogateDragNode.md)" in glance and "(../../nodes/BEMDragNode.md#governing-equations)" in glance
    assert "Undocumented, so with no equations to link: [`SurrogateDragNode`](../../nodes/SurrogateDragNode.md)" in glance
    assert "$$" not in page


def test_the_glance_block_links_the_equations_of_a_documented_node_and_the_anchor_is_there(tmp_path):
    """A node with ``NodeMeta`` has the heading the link names; a node without one is not linked to it."""
    record = entry([])
    record["class_vars"]["builds"] = {"type": "t", "declared_by": "x", "default": None, "value": {
        "": ["planted_nodes.PlantedNode", "planted_nodes.Doctored"]}}
    bare = planted("Doctored", meta=None)
    context = Context((record,), {}, {}, {"planted_nodes.PlantedNode": PlantedNode, "planted_nodes.Doctored": bare})
    page = entry_page(record, context)
    glance = page.text.split("## At a glance", 1)[1].split("\n## ", 1)[0]
    assert "The equations are on the node page: [equations](../../nodes/PlantedNode.md#governing-equations)" in glance
    assert "Undocumented, so with no equations to link: [`Doctored`](../../nodes/Doctored.md)" in glance
    for generated in (page, node_page(PlantedNode, "planted_nodes.PlantedNode", context),
                      node_page(bare, "planted_nodes.Doctored", context)):
        target = tmp_path / "pages" / generated.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated.text, encoding="utf-8")
    assert broken_links(tmp_path) == []


def test_the_drag_rung_is_a_section_with_its_nodes_and_its_support(built):
    page = text_of(built, "pages/entries/interaction/drag.md")
    section = page.split("### Rung surrogate", 1)[1].split("\n## ", 1)[0]
    assert "[`SurrogateDragNode`](../../nodes/SurrogateDragNode.md)" in section
    assert "a rigid body whose dynamics takes a resistance" in section


def test_a_rung_no_round_trip_case_builds_prints_declared_unchecked(built):
    page = text_of(built, "pages/entries/interaction/toy.twist.md")
    assert "declared, unchecked" in page


def test_the_vocabulary_lists_every_name_of_a_fresh_process(built):
    """Every name the vocabulary holds once MIME and each namespace are loaded (read in a fresh process, as the build
    reads it: a test of this process may have defined names of its own)."""
    done = subprocess.run([sys.executable, "-c", "from mime.docs._export import load_namespaces\n"
                           "load_namespaces()\nfrom mime.vocabulary import names\nprint('\\n'.join(names()))"],
                          capture_output=True, text=True, check=True)
    page = text_of(built, "pages/vocabulary.md")
    listed = done.stdout.split()
    assert len(listed) >= 8 and all(f"`{name}`" in page for name in listed)
    assert "body.resistance" in page and "Namespace toy" in page


def test_the_index_has_a_front_door_by_goal_and_links_the_concepts_note(built):
    page = text_of(built, "index.md")
    assert "## By goal" in page and "What can act on a body?" in page and "Which model, when there are several?" in page
    assert "(pages/concepts.md)" in page and "(pages/vocabulary.md)" in page and "(pages/results.md)" in page
    assert "(pages/entries/interaction/drag.md)" in page


def test_the_index_counts_built_members_and_says_a_comparison_starts_once_two_members_that_swap_are_built(built):
    page = text_of(built, "index.md")
    assert "| Family | Members | Built members |" in page and "Rungs built" not in page
    assert "once two of its members that can replace one another are built, a comparison" in page


def test_a_family_page_with_no_hand_written_part_says_how_to_choose_among_its_laws(built):
    """The placeholder speaks of the family's laws: a family whose members have no rungs reads it too."""
    for family in ("contact", "gravity", "hydrodynamic", "magnetic", "toy"):
        assert ("No hand-written part: how to choose among this family's laws, or to attach them together, is not "
                "written yet.") in text_of(built, f"pages/families/{family}.md")


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
    after_a_colon = {"pages/c.md": Page("c.md", "cwd:" + "/" + "home/zed/checkout and (" + "/" + "tmp/x)\n")}
    (problem,) = absolute_path_problems(after_a_colon)
    assert problem.startswith("pages/c.md: carries an absolute path") and "(/home/z" in problem
    assert absolute_path_problems({"pages/d.md": Page("d.md", "https://example.org/home/page and a/home/b\n")}) == []


def test_a_source_opening_with_its_title_is_inlined_with_the_title_one_level_down():
    assert demote_title("# A title\n\n**What.** Body.\n") == "### A title\n\n**What.** Body.\n"
    assert demote_title("\n# A title\nBody\n# Not the first\n") == "\n### A title\nBody\n# Not the first\n"
    assert demote_title("**What.** Body.\n# Later\n") == "**What.** Body.\n# Later\n"


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


def section(page: Page, title: str) -> str:
    """The text under the heading ``## title`` of a page, up to the next heading."""
    return page.text.split(f"## {title}\n\n", 1)[1].split("\n## ", 1)[0].strip()


def test_w7_an_empty_equations_assumptions_or_references_field_prints_none_declared_and_is_counted():
    """PlantedNode has equations and neither assumptions nor references; the plants empty the equations and fill the
    other two, so that each field is the one that fails in turn."""
    page = node_page(PlantedNode, "planted_nodes.PlantedNode", EMPTY)
    assert hatched(page) == ["node PlantedNode assumptions", "node PlantedNode references"]
    assert page.text.count("none declared") >= 2
    assert section(page, "Assumptions") == "none declared" and section(page, "References") == "none declared"
    assert section(page, "Governing equations").startswith("$$")
    blank = planted("Blank", meta=replace(META, governing_equations="", assumptions=("a",), references=_refs()))
    blank_page = node_page(blank, "planted_nodes.Blank", EMPTY)
    assert hatched(blank_page) == ["node Blank equations"]
    assert section(blank_page, "Governing equations") == "none declared"
    assert section(blank_page, "Assumptions") == "- a" and section(blank_page, "References") == "[@Sola2017]"
    bare = planted("Bare", meta=replace(META, governing_equations="   \n", assumptions=(), references=()))
    bare_page = node_page(bare, "planted_nodes.Bare", EMPTY)
    assert hatched(bare_page) == ["node Bare assumptions", "node Bare equations", "node Bare references"]
    assert [section(bare_page, title) for title in ("Governing equations", "Assumptions", "References")] == [
        "none declared"] * 3


def _refs():
    from maddening.core.compliance.metadata import Reference

    return (Reference("Sola2017", "quaternion kinematics"),)


def test_w7_the_passing_arm_a_node_with_all_three_prints_no_none_declared_for_them():
    full = planted("Full", meta=replace(META, assumptions=("small deflection",), references=_refs()))
    page = node_page(full, "planted_nodes.Full", EMPTY)
    assert hatched(page) == []
    assert "small deflection" in page.text and "[@Sola2017]" in page.text and "$$" in page.text
    assert "none declared" not in page.text.split("## Declared regimes")[0].split("## Assumptions")[1].split(
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


# -- the gradient record: every field it has is printed, switches as a table ---------------------------------------------

def gradient_entry(**fields: object) -> dict:
    record = entry([])
    record["class_vars"]["differentiability"]["value"] = "cut"
    record["class_vars"]["gradient"] = {"type": "t", "declared_by": "x", "default": None,
                                        "value": {"where": "w", "reason": "r", "route": "q", **fields}}
    return record


DECLARED = (
    switch.Declaration(owner="planted.clamp", indicator="the offset less the clamp, m", route="a smooth saturation",
                       orders={"pose": 0, "twist": -1}),
    switch.Declaration(owner="planted.time", indicator="the step count against the start", route="none: planted",
                       orders={"load": 1}, settings=("start_step",)),
)


def switch_entry() -> dict:
    """An entry whose gradient record is 7s's: switches only, ``where``, ``reason`` and ``route`` unset, exported as the
    registry export exports a ``GradientRecord`` (so the shape is the real one, not a stand-in)."""
    record = entry([])
    record["class_vars"]["differentiability"]["value"] = "piecewise_jump"
    record["class_vars"]["gradient"] = {"type": "t", "declared_by": "x", "default": None,
                                        "value": encode(GradientRecord(switches=DECLARED))}
    return record


def test_a_gradient_record_prints_each_field_it_has_by_name_and_skips_one_it_leaves_unset():
    page = entry_page(gradient_entry(extra="e"), EMPTY).text
    assert "- **Where**: w" in page and "- **Route to exact**: q" in page and "- **Extra**: e" in page
    unset = entry_page(gradient_entry(where=None), EMPTY).text
    assert "Where" not in unset and "null" not in unset and "- **Why**: r" in unset


def test_a_record_that_carries_switches_prints_each_with_its_owner_indicator_orders_and_route_readably():
    page = entry_page(switch_entry(), EMPTY).text
    section = page.split("## Gradient status", 1)[1]
    assert "`piecewise_jump`" in section
    assert "| `planted.clamp` | the offset less the clamp, m | `pose` 0 (a kink); `twist` -1 (a jump) | " \
        "a smooth saturation |" in section
    assert "| `planted.time` | the step count against the start. Reads the settings `start_step` | " \
        "`load` 1 (continuously differentiable) | none: planted |" in section
    for unreadable in ("null", "too long to show", "Where", "Why", "Route to exact**", "Switches**"):
        assert unreadable not in section, unreadable


def test_the_texts_of_a_switch_are_recorded_for_the_number_rule():
    recorded = dict(entry_page(switch_entry(), EMPTY).rendered)
    assert recorded["switch planted.clamp indicator"] == "the offset less the clamp, m"
    assert recorded["switch planted.time route"] == "none: planted"


def gradient_sections(page: str, record: dict) -> list[tuple[dict, str]]:
    """Each gradient record of an entry with the part of its page that prints it: the entry's under "Gradient
    status", or, where its rungs declare their own, each rung's in the rung's own section."""
    by_rung = record["class_vars"]["rung_gradients"]["value"]
    if not by_rung:
        return [(record["class_vars"]["gradient"]["value"],
                 page.split("## Gradient status", 1)[1].split("\n## ", 1)[0])]
    rungs = page.split("\n## Rungs\n", 1)[1].split("\n## ", 1)[0]
    return [(held["gradient"], rungs.split(f"### Rung {rung}\n", 1)[1].split("\n### ", 1)[0])
            for rung, held in by_rung.items()]


def test_the_real_gradient_records_are_printed_field_for_field(built, export):
    titles = {"where": "Where", "reason": "Why", "route": "Route to exact"}
    switched = 0
    for record in export["entries"]:
        for held, section in gradient_sections(text_of(built, page_of(record)), record):
            if not held:
                continue
            wanted = [titles.get(key, key.replace("_", " ").capitalize()) for key, value in held.items()
                      if value is not None and key != "switches"]
            assert all(f"- **{title}**:" in section for title in wanted), record["name"]
            assert "null" not in section and "too long to show" not in section, record["name"]
            for declared in held.get("switches", []):
                switched += 1
                assert f"| `{declared['owner']}` |" in section, (record["name"], declared["owner"])
                assert cell(declared["indicator"]) in section and cell(declared["route"]) in section
                assert all(f"`{group}` {order}" in section for group, order in declared["orders"].items())
    assert switched >= 10, "too few switches on the real pages to be evidence"


# -- a rung's own status: drag's rungs differ ------------------------------------------------------------------------

def rung_problems(entry_text: str, family_text: str, record: dict) -> list[str]:
    """What the drag's page and the hydrodynamic family's page get wrong of its rungs' own statuses: each rung's
    section prints its status and its own switches and no other rung's; the glance, the gradient section and the
    family's members and comparison print each rung's status."""
    found: list[str] = []
    rungs = entry_text.split("\n## Rungs\n", 1)[1].split("\n## ", 1)[0]
    glance = entry_text.split("## At a glance", 1)[1].split("\n## ", 1)[0]
    gradient = entry_text.split("## Gradient status", 1)[1].split("\n## ", 1)[0]
    by_rung = record["class_vars"]["rung_gradients"]["value"]
    owners = {rung: [d["owner"] for d in (held["gradient"] or {}).get("switches", [])]
              for rung, held in by_rung.items()}
    for rung, held in by_rung.items():
        status = held["differentiability"]
        own = rungs.split(f"### Rung {rung}\n", 1)[1].split("\n### ", 1)[0]
        if f"The derivative through this rung is `{status}`" not in own:
            found.append(f"rung {rung}'s section does not print its status {status}")
        found += [f"rung {rung}'s section lists {owner}, {other}'s" for other, held_owners in owners.items()
                  if other != rung for owner in held_owners if f"| `{owner}` |" in own]
        found += [f"rung {rung}'s section leaves out {owner}" for owner in owners[rung] if f"| `{owner}` |" not in own]
        anchor_ = f"[`{rung}`](#rung-{rung})"
        if f"{anchor_} `{status}`" not in glance or f"{anchor_}: `{status}`" not in gradient:
            found.append(f"the glance or the gradient section does not print rung {rung} as {status}")
        rows = [line for line in family_text.splitlines()
                if f"drag.md#rung-{rung})" in line or f"`drag {rung}`" in line]
        if len(rows) != 2 or not all(line.rstrip().endswith((f"`{status}` |", f"`{status}` | not measured |"))
                                     for line in rows):
            found.append(f"the family page does not print rung {rung} as {status} in its members and comparison")
    return found


def test_each_rung_prints_its_own_status_and_only_its_own_switches(built, export):
    """Drag's surrogate rung is ``piecewise_jump`` with its nine switches, its bem rung ``exact`` with none: each rung's
    section, the glance, the gradient section and the family page say so of each."""
    record = next(entry for entry in export["entries"] if entry["name"] == "drag")
    held = record["class_vars"]["rung_gradients"]["value"]
    assert {rung: value["differentiability"] for rung, value in held.items()} == {"surrogate": "piecewise_jump",
                                                                                  "bem": "exact"}
    assert len(held["surrogate"]["gradient"]["switches"]) == 9 and held["bem"]["gradient"] is None
    problems = rung_problems(text_of(built, page_of(record)), text_of(built, "pages/families/hydrodynamic.md"),
                             record)
    assert problems == []


def test_control_pages_that_print_the_entrys_status_for_every_rung_fail(export, monkeypatch):
    """The same check on the pages generated here with each rung's status and record read as the entry's: it fails."""
    from mime.docs import _entry_page, _family_page
    from mime.docs._family_page import family_page
    from mime.testing import family_named

    record = next(entry for entry in export["entries"] if entry["name"] == "drag")
    whole = record["class_vars"]["differentiability"]["value"]
    monkeypatch.setattr(_entry_page, "rung_status", lambda record_, rung: whole)
    monkeypatch.setattr(_family_page, "rung_status", lambda record_, rung: whole)
    monkeypatch.setattr(_entry_page, "rung_record", lambda record_, rung: record["class_vars"]["gradient"]["value"])
    hydrodynamic = family_named("hydrodynamic")
    context = Context(tuple(export["entries"]), {"hydrodynamic": hydrodynamic}, {}, {})
    problems = rung_problems(entry_page(record, context).text, family_page(hydrodynamic, context).text, record)
    print("\n" + "\n".join(problems))
    assert any("bem" in problem for problem in problems)
