"""``mime docs check`` over the checkout passes, printing each list's length and its wall time; over a doctored copy it
fails, exit 1, naming what is wrong; over a folder that is no checkout it fails, rather than passing on nothing.

And the checks of the four entry fields, each run on a planted entry class: ``builds`` keyed by the rungs an entry
builds, a gradient record for every status but ``exact`` (a ``cut`` or ``unverified`` one stating its reason and
route), ``references`` as keys, a declared family registered, and a value declared not established listed."""

import re
import subprocess
import sys
from dataclasses import dataclass

from checkout import ROOT, copy_checkout

from mime.docs import read_list
from mime.docs._check import LISTS, Report, entry_problems, list_path
from mime.gradients import GradientStatus
from mime.registry import NOT_ESTABLISHED, EntryKind, GradientRecord, Interaction, Registered, RoleKind
from mime.testing.toys._nodes import ToySpringNode


def docs_check(root=ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "mime.cli", "docs", "check", "--root", str(root)],
                          capture_output=True, text=True, check=False, cwd=ROOT)


def test_mime_docs_check_passes_on_the_checkout():
    done = docs_check()
    print(f"\n{done.stdout}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every check passed" in done.stdout
    assert re.search(r"lists: refusal_codes \d+, refusal_names \d+, symbol_rows 0", done.stdout)
    assert re.search(r"mime docs check: passed in [\d.]+ s", done.stdout)


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
                   read_list(path, LISTS["not_established"]), report)
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
