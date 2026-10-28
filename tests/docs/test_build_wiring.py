"""The build's guards, through the real commands. ``equation_problems``, ``absolute_path_problems`` and
``repository_relative`` are tested as functions in ``test_pages.py``; what this holds is the calls that wire them into
``build_pages`` and the problems into ``mime docs build`` and ``mime docs check``, so that a guard dropped from its
caller, or the build's problems dropped from the check, fails here. Each test runs a command in a fresh process after a
prelude plants an entry (``planted_entries.py``), so no other test's process holds it:

- a display equation on two pages: ``equation_problems`` is called and its problems stop the build and fail the check
  (a lab's entry building two nodes with one equation; no node of MIME carries ``NodeMeta`` before the pilot, so the
  real build has none to trip it);
- a path of the machine that is not the checkout's: ``absolute_path_problems`` is called, and stops both commands;
- the checkout's own path in a page's text: ``repository_relative`` is applied, so the page reads relative to the
  checkout (no check, which would read the digits of the path as numbers: the build is the wiring under test);
- and each has its passing arm, run the same way."""

import re

from checkout import ROOT, docs_run, failures

TWIN = re.compile(r"the display equation '.*' is on pages/nodes/PlantedNode\.md and on pages/nodes/SameEquationNode\.md; "
                  r"an equation has one home, a node's governing_equations, and the other page links to it")
ELSEWHERE = "/home/zed/elsewhere/data.txt"
ABSOLUTE = re.compile(r"pages/entries/interaction/docsplant\.lined\.md: carries an absolute path of the machine that "
                      r"built it \(/home/z")


def lined(text: str) -> str:
    return f"import planted_entries\nplanted_entries.plant_validity({text!r})\n"


def build(out, **kwargs):
    return docs_run("build", "--root", str(ROOT), "--out", str(out), **kwargs)


def check(**kwargs):
    return docs_run("check", "--root", str(ROOT), "--skip-hash-seed-builds", **kwargs)


def test_a_display_equation_on_two_pages_stops_the_build_and_fails_the_check(tmp_path):
    out = tmp_path / "out"
    done = build(out, plant="lab_twinned")
    assert done.returncode == 1 and not out.exists() and "no page written" in done.stdout
    assert any(TWIN.fullmatch(line.removeprefix("FAIL: ")) for line in done.stdout.splitlines()), done.stdout
    assert any(TWIN.fullmatch(problem) for problem in failures(check(plant="lab_twinned")))


def test_the_passing_arm_one_node_with_the_equation_builds_and_the_checkout_checks(tmp_path):
    """A node with ``NodeMeta`` on one page builds; the checkout as it stands (no plant) checks."""
    done = build(tmp_path / "out", plant="documented")
    assert done.returncode == 0, done.stdout
    assert (tmp_path / "out" / "pages" / "nodes" / "PlantedNode.md").is_file()
    passing = check()
    assert passing.returncode == 0, passing.stdout


def test_a_path_of_another_machine_in_a_page_stops_the_build_and_fails_the_check(tmp_path):
    out = tmp_path / "out"
    done = build(out, prelude=lined(f"read {ELSEWHERE}"))
    assert done.returncode == 1 and not out.exists()
    assert any(ABSOLUTE.match(line.removeprefix("FAIL: ")) for line in done.stdout.splitlines()), done.stdout
    assert any(ABSOLUTE.match(problem) for problem in failures(check(prelude=lined(f"read {ELSEWHERE}"))))


def test_the_passing_arm_a_relative_path_in_a_page_builds_and_checks(tmp_path):
    assert build(tmp_path / "out", prelude=lined("read docs/data.txt")).returncode == 0
    done = check(prelude=lined("read docs/data.txt"))
    assert done.returncode == 0, done.stdout


def test_the_checkouts_own_path_in_a_page_is_written_relative_to_it(tmp_path):
    out = tmp_path / "out"
    done = build(out, prelude=lined(f"see {ROOT}/docs/data.txt and {ROOT} itself"))
    assert done.returncode == 0, done.stdout
    page = (out / "pages" / "entries" / "interaction" / "docsplant.lined.md").read_text(encoding="utf-8")
    assert "see docs/data.txt and . itself" in page and str(ROOT) not in page
