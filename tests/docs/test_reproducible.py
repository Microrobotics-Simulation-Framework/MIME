"""W8: the pages are the same text under two hash seeds, and ``mime docs check`` says so.

``mime docs check`` builds every page twice, each build in a fresh process under its own ``PYTHONHASHSEED``, and fails on
any byte that differs, naming the page. The control plants a generator that writes a set's iteration order: twenty or more
strings, and the two fixed seeds, which this test checks once to give the plant two different orders. So the plant's two
builds differ on every run, and no detection rate over repeated runs is measured: fixed seeds make every run the same.

W8 bounds only this: byte-equal pages across two builds under two hash seeds, in one environment. It does not bound
timestamps, address-dependent ordering (``id()``, sets of objects), directory-listing order, or locale and machine
differences; none of those is planted or tested here."""

import subprocess
import sys
from pathlib import Path

from checkout import ROOT, docs_check, failures

from mime.docs._reproducible import SEEDS, build_command, differences, reproducibility_problems

#: The plant's strings: twenty-four, so that two seeds order them differently with a certainty the test checks.
WORDS = [f"word{n:02d}" for n in range(24)]
ORDER = f"print(','.join(set({WORDS!r})))"

#: A generator that is the real one plus a page that writes a set's iteration order.
PLANTED = f"""
import sys
from pathlib import Path
from mime.docs._build import build_pages, write_pages
from mime.docs._markdown import Page

root, out = Path(sys.argv[1]), Path(sys.argv[2])
built = build_pages(root)
built.pages["pages/planted.md"] = Page("planted.md", ",".join(set({WORDS!r})) + "\\n")
write_pages(built, out)
"""


def planted_command(root: Path, out: Path) -> list[str]:
    return [sys.executable, "-c", PLANTED, str(root), str(out)]


def test_the_two_seeds_give_the_plant_two_different_orders():
    """Checked once, as the control requires: were the orders the same, the plant would pass and prove nothing."""
    orders = [subprocess.run([sys.executable, "-c", ORDER], capture_output=True, text=True, check=True,
                             env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}).stdout for seed in SEEDS]
    assert len(set(orders)) == 2 and all(sorted(order.strip().split(",")) == WORDS for order in orders)


def test_w8_the_planted_set_order_makes_the_two_builds_differ_naming_the_page():
    problems = reproducibility_problems(ROOT, planted_command)
    (problem,) = problems
    assert problem.startswith("pages/planted.md: differs between PYTHONHASHSEED=1 and PYTHONHASHSEED=2, from byte ")
    assert "sort it" in problem


def test_w8_the_real_generator_passes_the_same_comparison():
    """The passing arm, which must pass on every run: the real command under the same two seeds."""
    assert reproducibility_problems(ROOT, build_command) == []


def test_w8_mime_docs_check_fails_on_a_planted_hash_order_dependence_and_passes_without_it():
    """Through the real command: a prelude replaces the command the check builds with, in the fresh process that runs
    it, and the check fails, naming the page; the same command without the plant passes."""
    prelude = ("import sys\nimport mime.docs._reproducible as reproducible\n"
               "sys.path.insert(0, " + repr(str(Path(__file__).parent)) + ")\n"
               "import test_reproducible\nreproducible.build_command = test_reproducible.planted_command\n")
    done = docs_check(prelude=prelude, hash_seeds=True)
    (problem,) = failures(done)
    assert problem.startswith("pages/planted.md: differs between PYTHONHASHSEED=1 and PYTHONHASHSEED=2")
    assert "pages built twice in fresh processes under two hash seeds: not equal" in done.stdout
    passing = docs_check(hash_seeds=True)
    assert passing.returncode == 0 and "under two hash seeds: byte-equal" in passing.stdout


def test_a_build_that_fails_is_a_problem_and_a_page_in_one_build_only_is_named():
    def failing(root: Path, out: Path) -> list[str]:
        return [sys.executable, "-c", "import sys; print('it broke'); sys.exit(3)"]

    (problem,) = reproducibility_problems(ROOT, failing)[:1]
    assert "PYTHONHASHSEED=1 exited 3: it broke" in problem
    found = differences({"a.md": b"x", "b.md": b"y"}, {"a.md": b"x"}, ("1", "2"))
    assert found == ["b.md: built under PYTHONHASHSEED=1 and not under the other seed"]
    assert differences({"a.md": b"same"}, {"a.md": b"same"}, ("1", "2")) == []
