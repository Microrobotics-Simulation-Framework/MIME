"""``mime docs check`` over the checkout passes, printing every stage's summary, each list's length and its wall time;
over a doctored copy it fails, exit 1, naming what is wrong; over a folder that is no checkout it fails, rather than
passing on nothing.

The controls of its wiring, each through the real command in a fresh process: one per stage, a defect only that stage
finds, planted in a copy of the checkout or as an entry the process registers (``planted_entries.py``), fails naming
it. A stage the command stopped running would let its control pass.

The controls of the follow-ups to the first documentation phase, each through the real command too: a node taking a
retired 0.2 id fails, naming the id as retired, and one taking an id the frozen list keeps for another class fails,
while the keeping class passes (on a copy of the list doctored to keep one); a MIME entry citing a key the bibliography
does not hold fails, naming the entry and the key, while one citing a key it holds passes; a lab's values not
established (its references, and its gradient record's three fields), and a lab's key, are counted and named in the
summary, never failed. A key's count is judged against the count the planted run's own process made before the plant,
so no entry another test registers in this process can move it.

And the checks of the four entry fields, each run on a planted entry class: ``builds`` keyed by the rungs an entry
builds, a gradient record for every status but ``exact`` (a ``cut`` or ``unverified`` one stating its reason and
route), ``references`` as keys, a declared family registered, and a value declared not established listed."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, docs_check, edit, failures
from planted_entries import Unequated
from planted_nodes import ROWS, KeeperNode, NamedLikeRigidBody, PlantedNode, TakesNextNode, UsurperNode

from mime.core import switch
from mime.docs import read_list, row_key
from mime.docs._check import LISTS, Report, entry_problems, list_path, read_lists
from mime.gradients import GradientStatus
from mime.registry import NOT_ESTABLISHED, EntryKind, GradientRecord, Interaction, Registered, RoleKind
from mime.testing.toys._nodes import ToySpringNode


def test_mime_docs_check_passes_on_the_checkout():
    done = docs_check(hash_seeds=True)
    print(f"\n{done.stdout}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every check passed" in done.stdout
    lengths = ", ".join(f"{name} {len(held)}" for name, held in read_lists(ROOT).items())
    stages = [r"registry: \d+ entries \(", r"not established outside MIME: 0$",
              r"references: \d+ keys of MIME's entries, checked against docs/bibliography\.bib; 0 of other namespaces'",
              r"gradients: \d+ entries not exact", r"gradients cut or unverified: \d+",
              r"nodes the entries build: \d+; with NodeMeta and a symbol table: \d+; undocumented: \d+",
              r"refusal table: \d+ codes; \d+ that no test names",
              r"docs walk \(.*/docs\): \d+ hand-written sources required of mime \(\d+ entries, \d+ mechanisms, \d+ "
              r"assets, \d+ nodes\); \d+ present; page sources: \d+$",
              r"ledger \(unpaged\.txt\): \d+ lines", r"exempt from the hand-written half by namespace \(toy\): \d+",
              r"outside MIME, without a hand-written source here, counted, not failed: labx 1 ",
              r"number tokens: a number token is a run of digits", r"number rule: \d+ files \(\d+ package READMEs",
              r"same-token groups \(lines of one file carrying the same tokens\): \d+", r"idioms: 3 ",
              r"word-numeral warnings: \d+, never a failure", re.escape(f"lists: {lengths}"),
              r"pages: \d+ \(.*\) from \d+ hand-written sources in [\d.]+ s", r"hatches printed: ",
              r"citations: \d+ in \d+ pages and sources, \d+ checked against \d+ bibliography entries, 0 not found",
              r"none declared in MIME's pages: \d+ fields, \d+ of the list's \d+ lines at the start reason; ",
              r"pages built in fresh processes under 8 hash seeds: byte-equal",
              r"scripts/check_anomalies\.py: OK",
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
    ("scripts/check_anomalies.py", ("docs/validation/known_anomalies.yaml", "resolution_status: resolved\n",
                                    "resolution_status: planted\n"), "resolution_status 'planted' is not one of"),
], ids=["anomalies"])
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


# -- the switch checks, through the real command -------------------------------------------------------------------

def test_the_command_holds_an_entrys_status_to_its_switches():
    assert failures(docs_check(plant="exact_jump")) == [
        "interaction docsplant.exact_jump 0.1.0: it is declared exact, and its switches' lowest order is -1 "
        "(planted_entries.jump): it is piecewise_jump"]


def test_the_command_checks_the_raw_switch_lists_item_form(tmp_path):
    root = copy_checkout(tmp_path)
    path = list_path(root, "raw_switches")
    path.write_text(path.read_text(encoding="utf-8") + "core/_geometry.py exponential where  # planted: no count\n",
                    encoding="utf-8")
    (problem,) = failures(docs_check(root))
    assert problem.startswith(f"{path}:") and "an item is a site, '<file> <function> <family> <occurrence>'" in problem


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


def test_the_next_id_named_is_past_every_id_in_use(tmp_path):
    """A retired id beside a node taking MIME-NODE-104: the id the failure names is the one after both lists."""
    root = pinned(pinned(copy_checkout(tmp_path), NamedLikeRigidBody), TakesNextNode)
    (problem,) = failures(docs_check(root, plant="next_id"))
    assert problem.startswith("RigidBodyNode: algorithm_id MIME-NODE-003 is retired")
    assert problem.endswith("an id names one node for good, so give it the next id, MIME-NODE-105")


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


def references_line(done: subprocess.CompletedProcess[str], prefix: str = "") -> tuple[int, int]:
    found = re.search(rf"^{prefix}references: (\d+) keys of MIME's entries, checked against docs/bibliography\.bib; "
                      r"(\d+) of other namespaces' entries, counted, not checked", done.stdout, re.MULTILINE)
    assert found is not None, done.stdout
    return int(found.group(1)), int(found.group(2))


def references_before(done: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    """The references line the planted run's own process counted before the plant: its baseline, which no entry
    another test registers in this process can reach."""
    return references_line(done, prefix="before the plant: ")


def paged(root: Path, name: str) -> Path:
    """``root`` with the hand-written file of the planted interaction of MIME's ``name``, which the docs walk needs."""
    (root / "docs/entries/interaction").mkdir(parents=True, exist_ok=True)
    (root / f"docs/entries/interaction/{name}.md").write_text("---\ndescribes: 0.1.0\n---\n\nA plant.\n",
                                                              encoding="utf-8")
    return root


def test_a_mime_entrys_key_the_bibliography_does_not_hold_fails_naming_the_entry_and_the_key(tmp_path):
    entry, page = failures(docs_check(paged(copy_checkout(tmp_path), "docsplant_mime_cited"), plant="mime_cited"))
    assert entry == ("interaction docsplant_mime_cited 0.1.0: references 'NoSuchWork1900', which docs/bibliography.bib "
                     "has no entry for")
    # the same key, read back from the page that cites it (the citation check reads the entries' references there)
    assert re.fullmatch(r"pages/entries/interaction/docsplant_mime_cited\.md:\d+: \[@NoSuchWork1900\] is not an "
                        r"entry of docs/bibliography\.bib; add the work to the bibliography, or correct the key", page)


def test_a_mime_entrys_key_the_bibliography_holds_passes(tmp_path):
    done = docs_check(paged(copy_checkout(tmp_path), "docsplant_mime_known"), plant="mime_known", baseline=True)
    assert done.returncode == 0, done.stdout + done.stderr
    mime, other = references_before(done)
    assert references_line(done) == (mime + 1, other)


def test_a_labs_key_is_counted_not_checked():
    done = docs_check(plant="lab_cited", baseline=True)
    assert done.returncode == 0, done.stdout + done.stderr
    mime, other = references_before(done)
    assert references_line(done) == (mime, other + 1)


def test_a_labs_value_not_established_is_counted_and_named_not_failed():
    done = docs_check(plant="lab_unestablished")
    assert done.returncode == 0, done.stdout + done.stderr
    listed = "\nnot established outside MIME: 1\n  - interaction docsplant.unestablished 0.1.0 references\n"
    assert listed in done.stdout


def test_a_labs_gradient_record_not_established_is_counted_and_named_not_failed():
    done = docs_check(plant="lab_unestablished_gradient")
    assert done.returncode == 0, done.stdout + done.stderr
    listed = "\nnot established outside MIME: 3\n" + "".join(
        f"  - interaction docsplant.unestablished_gradient 0.1.0 gradient.{field}\n"
        for field in ("reason", "route", "where"))
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


PLANTED_SWITCH = switch.Declaration(owner="planted.kink", indicator="a planted gap, m", orders={"twist": 0},
                                    route="none: planted")


def test_a_status_but_exact_needs_its_record_and_cut_needs_its_reason_and_route(tmp_path):
    for status in (GradientStatus.PIECEWISE_KINK, GradientStatus.PIECEWISE_JUMP, GradientStatus.UNVERIFIED):
        assert problems_of(tmp_path, variant(differentiability=status)) == [
            f"interaction planted.law 0.1.0: it is {status.value} and declares no gradient record (its switches, or "
            f"where, reason, route to exact)"]
    unrouted = GradientRecord(where="everywhere", reason="no derivative is taken", route=NOT_ESTABLISHED)
    listed = "interaction planted.law 0.1.0 gradient.route  # planted\n"
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.CUT, gradient=unrouted), listed) == [
        "interaction planted.law 0.1.0: it is cut, and a cut entry states its reason and its route to exact"]
    assert problems_of(tmp_path, variant(gradient=unrouted), listed)[0] == \
        "interaction planted.law 0.1.0: it is exact and declares where, why and a route to exact; an exact entry " \
        "declares none"


def test_a_piecewise_record_is_its_switches_and_a_status_no_switch_gives_states_where_why_and_route(tmp_path):
    switched = GradientRecord(switches=(PLANTED_SWITCH,))
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.PIECEWISE_KINK, gradient=switched)) == []
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.PIECEWISE_JUMP,
                                         gradient=GradientRecord(where="at a planted switch", reason="planted",
                                                                 route="none"))) == [
        "interaction planted.law 0.1.0: it is piecewise_jump and declares no switch; declare the switches its law "
        "evaluates in its gradient record"]
    assert problems_of(tmp_path, variant(differentiability=GradientStatus.FIXED_KEY, gradient=switched)) == [
        "interaction planted.law 0.1.0: it is fixed_key, and a fixed_key entry states where, its reason and its "
        "route to exact"]


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
