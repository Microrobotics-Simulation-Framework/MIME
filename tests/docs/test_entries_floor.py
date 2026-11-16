"""The entries' floor: no MIME entry leaves the registry unrecorded, with no count for two changes to both edit.

The registry is held to the source by ``test_refusal_table.py`` (the ``register_<kind>`` decorators and calls counted by
kind against a fresh interpreter's entries), which sees an entry registered in one place and not the other. It cannot
see an entry that leaves both: its decorator removed, or the import that registers it dropped, so that source and
registry lose it together. ``entries_floor.txt`` lists MIME's entries (kind, name, version; MIME's namespace only, so the
lab fixture and the toys never move it) as they were when the count was dropped; an entry on it that a fresh
interpreter does not hold fails unless ``entries_removed.txt`` logs it with a reason, and a logged entry that is back
fails. An entry added later needs no line.

The controls run the real check on a copy of the package, in a fresh process: the decorator removed from ``Exposure``
(the review's plant), the import that registers the camera's family dropped, and a floor entry planted that nothing
registers; each fails by name on every run, and the same logged as removed passes. There is no draw."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Collection
from pathlib import Path

import pytest
from checkout import ROOT, edit
from sorted_lists import entry_key, order_problems

from mime.docs import RatchetList
from mime.docs._export import load_namespaces
from mime.docs._lists import list_from_text, read_list
from mime.testing import installed_entries
from mime.vocabulary import MIME_NAMESPACE, namespace_of

FLOOR = Path(__file__).parent / "entries_floor.txt"
REMOVED = Path(__file__).parent / "entries_removed.txt"
LIST_ENTRIES = """
import json
import mime
from mime.registry import registered
print(json.dumps([[r.kind.value, r.name, r.version] for r in registered()]))
"""


def floor_entries(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip() and not line.startswith("#")]


def held_by(entries: Collection[tuple[str, str, str]]) -> set[str]:
    """MIME's own entries as the floor writes them. The namespaces are loaded first: a name is MIME's unless its first
    word is a registered namespace, so a toy or a lab entry is told apart only once those namespaces are registered
    (a test run alone, or in a stress run's order, has not loaded them)."""
    load_namespaces()
    return {f"{kind} {name} {version}" for kind, name, version in entries if namespace_of(name) == MIME_NAMESPACE}


def entries_of_a_checkout(root: Path) -> set[str]:
    """MIME's entries of the package under ``root``, read in a fresh process that imports it and nothing installed."""
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    done = subprocess.run([sys.executable, "-c", LIST_ENTRIES], capture_output=True, text=True, check=True,
                          env=environment, cwd=root)
    return held_by([tuple(row) for row in json.loads(done.stdout.splitlines()[-1])])


def removal_problems(held: Collection[str], floor: Collection[str], removed: RatchetList) -> list[str]:
    """Why the registry does not hold the floor: an entry of the floor that is not held and not logged as removed, an
    entry logged as removed that is held, and what is malformed in the log."""
    found = list(removed.problems)
    found += [f"{removed.path}:{line.number}: {line.item!r} is logged as removed and the registry holds it; delete "
              f"this line" for line in removed.lines if line.item in held]
    found += [f"{entry}: is on the floor ({FLOOR.name}) and the registry does not hold it; restore it, or log the "
              f"removal as '{entry}  # <reason>' in {REMOVED.name}"
              for entry in sorted(set(floor) - set(held) - removed.items())]
    return found


def test_the_registry_holds_every_entry_of_its_floor_or_logs_its_removal():
    load_namespaces()
    floor = floor_entries(FLOOR.read_text(encoding="utf-8"))
    assert floor and removal_problems(held_by(installed_entries()), floor, read_list(REMOVED)) == []


def test_the_floor_and_the_log_are_sorted_without_a_duplicate_and_hold_only_mime_entries():
    assert order_problems(FLOOR.name, FLOOR.read_text(encoding="utf-8"), entry_key) == []
    assert order_problems(REMOVED.name, REMOVED.read_text(encoding="utf-8"), entry_key) == []
    floor = floor_entries(FLOOR.read_text(encoding="utf-8"))
    assert all(len(entry.split()) >= 3 and not entry.startswith(("toy", "labx")) for entry in floor)
    assert not any(namespace_of(entry.split()[1]) != MIME_NAMESPACE for entry in floor)


# -- the controls ---------------------------------------------------------------------------------------------------

def package_copy(into: Path) -> Path:
    """The package with its data files, under ``into/src`` (the entries are read by importing it, which needs them)."""
    shutil.copytree(ROOT / "src/mime", into / "src/mime", ignore=shutil.ignore_patterns("__pycache__"))
    return into


def planted(tmp_path: Path, path: str, old: str, new: str) -> Path:
    root = package_copy(tmp_path)
    edit(root / path, old, new)
    return root


def lost(root: Path) -> list[str]:
    return sorted(set(floor_entries(FLOOR.read_text(encoding="utf-8"))) - entries_of_a_checkout(root))


def test_the_copy_with_nothing_planted_holds_the_whole_floor(tmp_path):
    assert lost(package_copy(tmp_path)) == []


def test_control_the_decorator_removed_from_an_entry_fails_naming_it(tmp_path):
    """The review's plant: source and registry lose the entry together, so the set equality of the refusal table's
    test stays green; only the floor sees it."""
    root = planted(tmp_path, "src/mime/observations/_exposure.py", "@register_observation\n", "")
    gone = lost(root)
    assert gone == ["observation exposure 0.1.0"]
    log = list_from_text(REMOVED, "")
    assert removal_problems(entries_of_a_checkout(root), floor_entries(FLOOR.read_text(encoding="utf-8")), log) == [
        f"observation exposure 0.1.0: is on the floor ({FLOOR.name}) and the registry does not hold it; restore it, or "
        f"log the removal as 'observation exposure 0.1.0  # <reason>' in {REMOVED.name}"]


def test_control_the_import_that_registers_a_family_dropped_fails_naming_its_entries(tmp_path):
    root = planted(tmp_path, "src/mime/observations/__init__.py", "from mime.observations._exposure import Exposure\n",
                   "")
    assert "observation exposure 0.1.0" in lost(root)


def test_control_a_logged_removal_passes_and_a_logged_entry_that_is_back_fails(tmp_path):
    root = planted(tmp_path, "src/mime/observations/_exposure.py", "@register_observation\n", "")
    floor = floor_entries(FLOOR.read_text(encoding="utf-8"))
    log = list_from_text(REMOVED, "observation exposure 0.1.0  # removed with the planted change\n")
    assert removal_problems(entries_of_a_checkout(root), floor, log) == []
    back = list_from_text(REMOVED, "# header\nobservation exposure 0.1.0  # removed once\n")
    assert removal_problems(held_by(installed_entries()), floor, back) == [
        f"{REMOVED}:2: 'observation exposure 0.1.0' is logged as removed and the registry holds it; delete this line"]


def test_control_a_floor_entry_nothing_registers_fails_and_a_log_line_with_no_reason_is_a_problem():
    floor = [*floor_entries(FLOOR.read_text(encoding="utf-8")), "observation planted 0.1.0"]
    held = held_by(installed_entries())
    assert removal_problems(held, floor, list_from_text(REMOVED, ""))[0].startswith("observation planted 0.1.0: is on")
    unreasoned = removal_problems(held, floor, list_from_text(REMOVED, "observation planted 0.1.0\n"))
    assert unreasoned and "has no reason" in unreasoned[0]


def test_control_a_lab_or_toy_entry_is_not_counted_and_a_floor_line_out_of_order_is_named():
    assert held_by([("interaction", "toy.spring", "0.1.0"), ("interaction", "labx.far", "0.1.0"),
                    ("interaction", "drag", "0.4.0")]) == {"interaction drag 0.4.0"}
    lines = FLOOR.read_text(encoding="utf-8").splitlines()
    first = next(n for n, line in enumerate(lines, start=1) if not line.startswith("#"))
    planted = "\n".join([*lines[:first - 1], *lines[first:], lines[first - 1]]) + "\n"
    assert order_problems(FLOOR.name, planted, entry_key) == [
        f"{FLOOR.name}:{len(lines)}: sorts before the line above it; move it up to just above line {first}"]
