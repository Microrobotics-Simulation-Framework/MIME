"""``mime docs check`` over the checkout passes, printing every stage's summary, each list's length and its wall time;
over a doctored copy it fails, exit 1, naming what is wrong; over a folder that is no checkout it fails, rather than
passing on nothing.

The controls of its wiring, each through the real command in a fresh process: one per stage, a defect only that stage
finds, planted in a copy of the checkout or as an entry the process registers (``planted_entries.py``), fails naming
it. A stage the command stopped running would let its control pass.

The controls of the follow-ups to the first documentation phase, each through the real command too: a node taking a
retired 0.2 id fails, naming the id as retired, and one taking an id the frozen list keeps for another class fails,
while the keeping class passes (on a copy of the list doctored to keep one); a MIME entry citing a key the bibliography
does not hold fails, naming the entry and the key, while one citing a key it holds passes; a lab's value not
established, and a lab's key, are counted and named in the summary, never failed.

And the checks of the four entry fields, each run on a planted entry class: ``builds`` keyed by the rungs an entry
builds, a gradient record for every status but ``exact`` (a ``cut`` or ``unverified`` one stating its reason and
route), ``references`` as keys, a declared family registered, and a value declared not established listed."""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, edit
from planted_entries import Unequated
from planted_nodes import ROWS, KeeperNode, NamedLikeRigidBody, PlantedNode, UsurperNode

from mime.docs import read_list, row_key
from mime.docs._check import LISTS, Report, entry_problems, list_path, read_lists
from mime.gradients import GradientStatus
from mime.registry import NOT_ESTABLISHED, EntryKind, GradientRecord, Interaction, Registered, RoleKind
from mime.testing.toys._nodes import ToySpringNode


def docs_check(root: Path = ROOT, plant: str | None = None) -> subprocess.CompletedProcess[str]:
    """``mime docs check --root root`` in a fresh process, which first registers the planted entry ``plant``."""
    command = ["docs", "check", "--root", str(root)]
    prelude = "" if plant is None else (f"sys.path.insert(0, {str(Path(__file__).parent)!r})\nimport planted_entries\n"
                                        f"planted_entries.plant({plant!r})\n")
    return subprocess.run([sys.executable, "-c", f"import sys\n{prelude}from mime.cli import main\n"
                                                 f"sys.exit(main({command!r}))"],
                          capture_output=True, text=True, check=False, cwd=ROOT)


def failures(done: subprocess.CompletedProcess[str]) -> list[str]:
    assert done.returncode == 1, done.stdout + done.stderr
    return [line.removeprefix("FAIL: ") for line in done.stdout.splitlines() if line.startswith("FAIL: ")]


def test_mime_docs_check_passes_on_the_checkout():
    done = docs_check()
    print(f"\n{done.stdout}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every check passed" in done.stdout
    lengths = ", ".join(f"{name} {len(held)}" for name, held in read_lists(ROOT).items())
    stages = [r"registry: \d+ entries \(", r"not established outside MIME: 0$",
              r"references: \d+ keys of MIME's entries, checked against docs/bibliography\.bib; 0 of other namespaces'",
              r"gradients: \d+ entries not exact", r"gradients cut or unverified: \d+",
              r"nodes the entries build: \d+; with NodeMeta and a symbol table: \d+; undocumented: \d+",
              r"refusal table: \d+ codes; \d+ that no test names", re.escape(f"lists: {lengths}"),
              r"scripts/check_citations\.py: OK", r"scripts/check_anomalies\.py: OK",
              r"mime docs check: passed in [\d.]+ s"]
    assert [stage for stage in stages if not re.search(f"^{stage}", done.stdout, re.MULTILINE)] == []


def test_mime_docs_check_fails_a_doctored_checkout_naming_what_is_wrong(tmp_path):
    root = copy_checkout(tmp_path)
    path = list_path(root, "builds_unchecked")
    path.write_text(path.read_text(encoding="utf-8") + "interaction toy.drag\n", encoding="utf-8")
    done = docs_check(root)
    assert done.returncode == 1
    assert f"FAIL: {path}:" in done.stdout and "'interaction toy.drag' has no reason" in done.stdout


def test_mime_docs_check_fails_on_a_folder_that_is_no_checkout(tmp_path):
    done = docs_check(tmp_path)
    assert done.returncode == 1 and "is not a checkout of MIME" in done.stdout


# -- the wiring: each stage, through the real command -------------------------------------------------------------

def test_the_command_runs_the_completeness_test_on_every_registered_entry():
    assert failures(docs_check(plant="undeclared")) == [
        "interaction docsplant.undeclared 0.1.0 (planted_entries.Undeclared): Undeclared.table is held at class level "
        "and is not a declared ClassVar, so the registry export does not carry it: declare it table: ClassVar[...] "
        "where it is first set"]


def test_the_command_checks_the_entry_fields_against_the_not_established_list(tmp_path):
    root = copy_checkout(tmp_path)
    path = list_path(root, "not_established")
    path.write_text(path.read_text(encoding="utf-8") + "interaction toy.spring 0.1.0 references  # planted\n",
                    encoding="utf-8")
    (problem,) = failures(docs_check(root))
    assert problem.endswith("'interaction toy.spring 0.1.0 references' now passes; delete this line")


def pinned(root: Path, node: type) -> Path:
    """``root``'s floor of symbol rows, with every row of ``node``'s table pinned."""
    path = list_path(root, "symbol_rows")
    path.write_text(path.read_text(encoding="utf-8") + "".join(f"{row_key(node, row)}  # pinned at landing\n"
                                                               for row in ROWS), encoding="utf-8")
    return root


def test_the_command_checks_the_symbol_table_of_every_node_an_entry_builds(tmp_path):
    (node,) = Unequated.builds[""]
    assert failures(docs_check(pinned(copy_checkout(tmp_path), node), plant="unequated")) == [
        "UnequatedNode: the row 'a': a does not occur in governing_equations"]


def test_the_command_passes_an_entry_whose_nodes_table_is_clean_and_pinned(tmp_path):
    done = docs_check(pinned(copy_checkout(tmp_path), PlantedNode), plant="documented")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "with NodeMeta and a symbol table: 1;" in done.stdout


def test_the_command_checks_that_a_units_line_names_a_node_with_a_table(tmp_path):
    root = copy_checkout(tmp_path)
    path = list_path(root, "symbol_units")
    path.write_text(path.read_text(encoding="utf-8") + "ToySpringNode x  # planted\n", encoding="utf-8")
    (problem,) = failures(docs_check(root))
    assert problem.endswith("'ToySpringNode x' names no node with a symbol table; delete this line")


def test_the_command_checks_the_refusal_table_against_the_source(tmp_path):
    root = copy_checkout(tmp_path)
    table = root / "src/mime/build/refusal_table.txt"
    first = next(line for line in table.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))
    edit(table, f"{first}\n", "")
    code, *classes = first.split()
    (problem,) = [p for p in failures(docs_check(root)) if p.startswith(f"{code}: constructed at ")]
    assert problem.endswith(f" and not in {table}; add the row '{code}  {' '.join(classes)}'")


@pytest.mark.parametrize(("script", "plant", "named"), [
    ("scripts/check_citations.py", ("docs/planted.md", None, "As shown [@NoSuchWork1900].\n"),
     "docs/planted.md:1: [@NoSuchWork1900] not found"),
    ("scripts/check_anomalies.py", ("docs/validation/known_anomalies.yaml", "resolution_status: resolved\n",
                                    "resolution_status: planted\n"), "resolution_status 'planted' is not one of"),
], ids=["citations", "anomalies"])
def test_the_command_runs_each_compliance_script(tmp_path, script, plant, named):
    root = copy_checkout(tmp_path)
    path, old, new = plant
    if old is None:
        (root / path).write_text(new, encoding="utf-8")
    else:
        text = (root / path).read_text(encoding="utf-8")
        (root / path).write_text(text.replace(old, new, 1), encoding="utf-8")
    done = docs_check(root)
    assert failures(done) == [f"{script} failed (exit 1); its output is above"]
    assert named in done.stdout


# -- the 0.2 ids, the bibliography and a lab's values, through the real command ----------------------------------------

def kept(root: Path, by: str) -> Path:
    """``root``'s frozen list with MIME-NODE-012 kept by the class at the dotted path ``by``, for the control."""
    path = root / "docs/data/node_ids_0.2.yaml"
    text = path.read_text(encoding="utf-8")
    start, end = text.index("  - id: MIME-NODE-012\n"), text.index("  - id: MIME-NODE-013\n")
    row = text[start:end]
    held = row[:row.index("    retired:")] + f"    kept_by: {by}\n    evidence: planted for the control\n"
    path.write_text(text[:start] + held + text[end:], encoding="utf-8")
    return root


def test_a_node_taking_a_retired_0_2_id_fails_naming_the_id_as_retired(tmp_path):
    (problem,) = failures(docs_check(pinned(copy_checkout(tmp_path), NamedLikeRigidBody), plant="retired_id"))
    assert problem.startswith("RigidBodyNode: algorithm_id MIME-NODE-003 is retired (the 0.2 line's RigidBodyNode: "
                              "changed in RigidInertialNode")
    assert problem.endswith("an id names one node for good, so give it the next id, MIME-NODE-104")


def test_a_node_taking_a_kept_0_2_id_it_is_not_kept_by_fails(tmp_path):
    root = kept(pinned(copy_checkout(tmp_path), UsurperNode), "planted_nodes.KeeperNode")
    assert failures(docs_check(root, plant="taken_id")) == [
        "UsurperNode: algorithm_id MIME-NODE-012 is kept by planted_nodes.KeeperNode (the 0.2 line's GravityNode, "
        "found the same), and planted_nodes.UsurperNode is not that class; give it the next id, MIME-NODE-104"]


def test_the_class_keeping_a_0_2_id_passes(tmp_path):
    done = docs_check(kept(pinned(copy_checkout(tmp_path), KeeperNode), "planted_nodes.KeeperNode"), plant="kept_id")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "with NodeMeta and a symbol table: 1;" in done.stdout


def test_a_frozen_id_both_retired_and_kept_fails(tmp_path):
    root = copy_checkout(tmp_path)
    edit(root / "docs/data/node_ids_0.2.yaml", "    retired: no 0.3 counterpart\n  - id: MIME-NODE-101\n",
         "    retired: no 0.3 counterpart\n    kept_by: planted_nodes.KeeperNode\n    evidence: planted\n"
         "  - id: MIME-NODE-101\n")
    (problem,) = failures(docs_check(root))
    assert problem.endswith("MIME-NODE-100 is either retired, saying what changed, or kept_by a class, with its "
                            "evidence")


def references_line(done: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    found = re.search(r"^references: (\d+) keys of MIME's entries, checked against docs/bibliography\.bib; (\d+) of "
                      r"other namespaces' entries, counted, not checked", done.stdout, re.MULTILINE)
    assert found is not None, done.stdout
    return int(found.group(1)), int(found.group(2))


def test_a_mime_entrys_key_the_bibliography_does_not_hold_fails_naming_the_entry_and_the_key():
    assert failures(docs_check(plant="mime_cited")) == [
        "interaction docsplant_mime_cited 0.1.0: references 'NoSuchWork1900', which docs/bibliography.bib has no entry "
        "for"]


def test_a_mime_entrys_key_the_bibliography_holds_passes_and_a_labs_key_is_counted_not_checked():
    mime, other = references_line(checkout := docs_check())
    assert checkout.returncode == 0 and other == 0
    known, cited = docs_check(plant="mime_known"), docs_check(plant="lab_cited")
    assert known.returncode == 0 and references_line(known) == (mime + 1, 0), known.stdout
    assert cited.returncode == 0 and references_line(cited) == (mime, 1), cited.stdout


def test_a_labs_value_not_established_is_counted_and_named_not_failed():
    done = docs_check(plant="lab_unestablished")
    assert done.returncode == 0, done.stdout + done.stderr
    listed = "\nnot established outside MIME: 1\n  - interaction docsplant.unestablished 0.1.0 references\n"
    assert listed in done.stdout


# -- the entry fields -----------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Planted(Interaction):
    name = "planted.law"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    builds = {"": (ToySpringNode,)}


Planted.__module__ = "mime.planted"


def problems_of(tmp_path, cls: type[Interaction], listed: str = "") -> list[str]:
    path = tmp_path / "not_established.txt"
    path.write_text(listed, encoding="utf-8")
    report = Report()
    entry_problems([Registered(EntryKind.INTERACTION, cls.name, cls.version, cls)],
                   read_list(path, LISTS["not_established"]), report, bibliography={"Purcell1977"})
    return report.problems


def variant(**attributes) -> type[Interaction]:
    return type("Planted", (Planted,), {"__module__": "mime.planted", **attributes})


def test_a_clean_entry_passes(tmp_path):
    assert problems_of(tmp_path, Planted) == []


def test_builds_missing_or_keyed_by_the_wrong_rungs_fails(tmp_path):
    assert problems_of(tmp_path, variant(builds={})) == [
        "interaction planted.law 0.1.0: it declares no builds; declare each rung's node classes, builds = {'': (...)}"]
    assert problems_of(tmp_path, variant(rungs=("one", "two"), reserved_rungs={"two": "later"})) == [
        "interaction planted.law 0.1.0: builds names the rungs [''], and it builds ['one'] ('' for an entry without "
        "rungs)"]


def test_a_status_but_exact_needs_its_record_and_cut_needs_its_reason_and_route(tmp_path):
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.PIECEWISE)) == [
        "interaction planted.law 0.1.0: it is piecewise and declares no gradient record (where, reason, route to "
        "exact)"]
    unrouted = GradientRecord("everywhere", "no derivative is taken", NOT_ESTABLISHED)
    listed = "interaction planted.law 0.1.0 gradient.route  # planted\n"
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.CUT, gradient=unrouted), listed) == [
        "interaction planted.law 0.1.0: it is cut, and a cut entry states its reason and its route to exact"]
    assert problems_of(tmp_path, variant(gradient=unrouted), listed)[0] == \
        "interaction planted.law 0.1.0: it is exact and declares a gradient record; an exact entry has none"


def test_a_value_not_established_fails_until_listed_and_a_listed_one_now_established_fails(tmp_path):
    unknown = variant(references=NOT_ESTABLISHED)
    (unlisted,) = problems_of(tmp_path, unknown)
    assert unlisted.startswith("interaction planted.law 0.1.0 references: a value declared not established")
    assert problems_of(tmp_path, unknown, "interaction planted.law 0.1.0 references  # planted\n") == []
    (stale,) = problems_of(tmp_path, Planted, "interaction planted.law 0.1.0 references  # planted\n")
    assert stale.endswith("'interaction planted.law 0.1.0 references' now passes; delete this line")


def test_references_that_are_not_keys_and_an_unregistered_family_fail(tmp_path):
    assert problems_of(tmp_path, variant(references=("Cox and Brenner",))) == [
        "interaction planted.law 0.1.0: references ('Cox and Brenner',) are not bibliography keys"]
    (problem,) = problems_of(tmp_path, variant(family="no_such_family"))
    assert problem.startswith("interaction planted.law 0.1.0: family: no family 'no_such_family' is registered")
