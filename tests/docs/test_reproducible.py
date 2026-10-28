"""W8: the pages are the same text under several hash seeds, and ``mime docs check`` says so.

``mime docs check`` builds every page once per seed in ``SEEDS`` (six), each build in a fresh process under its own
``PYTHONHASHSEED``, and fails on any byte that differs from the first build's, naming the page. The control plants a
generator that writes a set's iteration order: twenty or more strings, which the six seeds order differently, so the
plant's builds differ on every run (the seeds are fixed; no detection rate over repeated runs is informative, and the
rates measured here are over sets of strings).

What it detects, measured: a set of two to four strings is the case two seeds miss about as often as they catch
(two seeds order a pair alike for half of the pairs); the test draws a hundred sets of each size with a fixed
generator and requires the seeds to order each size differently in at least the shares below, and requires the first
two seeds alone to do worse, so the control discriminates between six seeds and two.

What it does not detect: a clock. The builds start together, so a timestamp to the second is the same in all of them;
a test holds instead that no generator module imports a clock, a random source or a machine's identity, nor takes the
address of an object. Nor address-dependent ordering, directory-listing order, or locale and machine differences."""

import ast
import random
import string
import subprocess
import sys
from pathlib import Path

import pytest
from checkout import ROOT, docs_check, failures

from mime.docs._reproducible import SEEDS, build_command, differences, reproducibility_problems

#: The plant's strings: twenty-four, so that the seeds order them differently with a certainty the test checks.
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


def under_seed(seed: str, program: str) -> str:
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True,
                          env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}).stdout


def test_the_seeds_give_the_plant_different_orders():
    """Checked once, as the control requires: were the orders the same, the plant would pass and prove nothing."""
    orders = [under_seed(seed, ORDER) for seed in SEEDS]
    assert len(SEEDS) >= 4 and len(set(orders)) >= 4
    assert orders[0] != orders[1] and all(sorted(order.strip().split(",")) == WORDS for order in orders)


def test_w8_the_planted_set_order_makes_the_builds_differ_naming_the_page_once():
    problems = reproducibility_problems(ROOT, planted_command)
    (problem,) = problems
    assert problem.startswith("pages/planted.md: differs between PYTHONHASHSEED=1 and PYTHONHASHSEED=")
    assert "sort it" in problem


def test_w8_the_real_generator_passes_the_same_comparison():
    """The passing arm, which must pass on every run: the real command under the same seeds."""
    assert reproducibility_problems(ROOT, build_command) == []


def test_w8_mime_docs_check_fails_on_a_planted_hash_order_dependence_and_passes_without_it():
    """Through the real command: a prelude replaces the command the check builds with, in the fresh process that runs
    it, and the check fails, naming the page; the same command without the plant passes."""
    prelude = ("import sys\nimport mime.docs._reproducible as reproducible\n"
               "sys.path.insert(0, " + repr(str(Path(__file__).parent)) + ")\n"
               "import test_reproducible\nreproducible.build_command = test_reproducible.planted_command\n")
    done = docs_check(prelude=prelude, hash_seeds=True)
    (problem,) = failures(done)
    assert problem.startswith("pages/planted.md: differs between PYTHONHASHSEED=1 and PYTHONHASHSEED=")
    assert f"pages built in fresh processes under {len(SEEDS)} hash seeds: not equal" in done.stdout
    passing = docs_check(hash_seeds=True)
    assert passing.returncode == 0 and f"under {len(SEEDS)} hash seeds: byte-equal" in passing.stdout


def test_a_build_that_fails_is_a_problem_and_a_page_in_one_build_only_is_named():
    def failing(root: Path, out: Path) -> list[str]:
        return [sys.executable, "-c", "import sys; print('it broke'); sys.exit(3)"]

    (problem,) = reproducibility_problems(ROOT, failing)[:1]
    assert "PYTHONHASHSEED=1 exited 3: it broke" in problem
    found = differences({"a.md": b"x", "b.md": b"y"}, {"a.md": b"x"}, ("1", "2"))
    assert found == ["b.md: built under PYTHONHASHSEED=1 and not under the other seed"]
    assert differences({"a.md": b"same"}, {"a.md": b"same"}, ("1", "2")) == []


# -- the detection of a small set: measured over random sets -----------------------------------------------------------

DRAWS = 100
#: What the seeds must find, as a share of the sets of each size drawn: about 97, 99.99 and 100 per cent are expected.
REQUIRED = {2: 0.90, 3: 0.97, 4: 0.99}


def drawn_sets() -> dict[int, list[list[str]]]:
    rng = random.Random(20261028)
    return {size: [["".join(rng.choices(string.ascii_lowercase, k=7)) for _ in range(size)] for _ in range(DRAWS)]
            for size in REQUIRED}


def orders_under(seed: str, sets: dict[int, list[list[str]]]) -> dict[int, list[str]]:
    program = f"sets = {sets!r}\nfor size, drawn in sets.items():\n    for one in drawn:\n        print(size, ','.join(set(one)))"
    found: dict[int, list[str]] = {size: [] for size in sets}
    for line in under_seed(seed, program).splitlines():
        size, order = line.split(" ", 1)
        found[int(size)].append(order)
    return found


def test_w8_the_seeds_order_a_small_set_differently_for_nearly_every_set_and_two_seeds_do_not():
    sets = drawn_sets()
    by_seed = {seed: orders_under(seed, sets) for seed in SEEDS}
    for size, floor in REQUIRED.items():
        found = sum(len({by_seed[seed][size][at] for seed in SEEDS}) > 1 for at in range(DRAWS))
        assert found / DRAWS >= floor, f"sets of {size}: {found} of {DRAWS} ordered differently under the six seeds"
    pairs = sum(by_seed[SEEDS[0]][2][at] != by_seed[SEEDS[1]][2][at] for at in range(DRAWS))
    assert pairs / DRAWS < 0.75, f"two seeds order {pairs} of {DRAWS} pairs differently; the control cannot tell them from six"


# -- a clock is invisible to W8, so no generator reads one ---------------------------------------------------------------

#: What a page's text must not come from: a clock, a random source, the identity of the machine, an object's address.
BANNED_MODULES = frozenset({"time", "datetime", "calendar", "random", "secrets", "uuid", "socket", "getpass", "platform",
                            "locale"})
BANNED_CALLS = frozenset({"os.getcwd", "os.getpid", "os.urandom", "os.getlogin"})
#: The build and the check time themselves for their summaries, never for a page, through ``time.perf_counter`` alone.
TIMERS = frozenset({"_build.py", "_check.py"})


def nondeterminism(source: str, timer: bool = False) -> list[str]:
    """Why a module's text may differ between two runs: an import of a clock or random source (``time`` is allowed
    for ``timer`` when only ``perf_counter`` is taken from it), a call that reads the machine, or ``id``."""
    found: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"imports {alias.name}" for alias in node.names
                      if alias.name.split(".")[0] in BANNED_MODULES and not (timer and alias.name == "time")]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in BANNED_MODULES:
            found.append(f"imports from {node.module}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            dotted = f"{node.value.id}.{node.attr}"
            if dotted in BANNED_CALLS or (node.value.id in BANNED_MODULES and not (
                    timer and dotted == "time.perf_counter")):
                found.append(f"uses {dotted}")
        elif isinstance(node, ast.Name) and node.id == "id":
            found.append("uses id")
    return found


def test_no_module_of_the_documentation_tools_reads_a_clock_a_random_source_or_the_machine():
    modules = sorted((ROOT / "src" / "mime" / "docs").glob("*.py"))
    assert len(modules) >= 15
    found = {module.name: nondeterminism(module.read_text(encoding="utf-8"), module.name in TIMERS)
             for module in modules}
    assert {name: why for name, why in found.items() if why} == {}
    assert all((ROOT / "src" / "mime" / "docs" / name).is_file() for name in TIMERS)


@pytest.mark.parametrize(("source", "timer", "named"), [
    ("import time\nstamp = time.time()\n", False, "imports time"),
    ("import time\nstamp = time.strftime('%H:%M')\n", True, "uses time.strftime"),
    ("from datetime import datetime\n", False, "imports from datetime"),
    ("import datetime as dt\n", True, "imports datetime"),
    ("import random\n", False, "imports random"),
    ("import os\npath = os.getcwd()\n", False, "uses os.getcwd"),
    ("order = sorted(items, key=id)\n", False, "uses id"),
], ids=["a-timestamp", "a-timestamp-in-a-timer-module", "from-datetime", "datetime-in-a-timer-module", "a-random-source",
        "the-working-directory", "an-address"])
def test_the_clock_check_finds_each_planted_source_of_nondeterminism(source, timer, named):
    assert named in nondeterminism(source, timer)


def test_the_clock_check_passes_a_timer_that_takes_only_perf_counter_and_a_module_with_none():
    assert nondeterminism("import time\nstarted = time.perf_counter()\n", timer=True) == []
    assert nondeterminism("import re\nname = re.sub('a', 'b', 'abc')\n") == []
