"""W8: the pages are the same text under several hash seeds. ``mime docs check`` builds every page once per seed, each
build in a fresh process under its own ``PYTHONHASHSEED`` and all of them at once, and fails on any byte that differs from
the first build's, naming the page.

This bounds one thing: byte-equal pages across the builds under the seeds below, in one environment. A set of ``n``
strings that a generator writes in its iteration order is found unless every seed orders it as the first does, which
for ``k`` seeds is about ``(1/n!)^(k-1)`` of the sets of ``n`` strings: 1 in 100 for two items under eight seeds, and
fewer for more (the control measures it over random sets, ``test_reproducible.py``). It does not bound a clock (the
two builds start together, so a timestamp to the second is the same in both; a test holds that the generators read no
clock instead), an ordering that follows an object's address (``id()``, a set of objects), the order a directory lists,
or a difference of locale or machine; none of those is planted here."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

#: The seeds. They are fixed, so a build's orders are a fixed fact and a control that plants a set's order fails on
#: every run; eight of them, since two seeds order a pair of strings the same way about as often as not (the pair is
#: missed by eight seeds in about one draw in a hundred).
SEEDS = ("1", "2", "3", "4", "5", "6", "7", "8")


def build_command(root: Path, out: Path) -> list[str]:
    """The command that builds the pages of ``root`` into ``out``, as the CLI runs it."""
    return [sys.executable, "-m", "mime.cli", "docs", "build", "--root", str(root), "--out", str(out)]


def _read(directory: Path) -> dict[str, bytes]:
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def differences(first: dict[str, bytes], second: dict[str, bytes], seeds: tuple[str, str]) -> list[str]:
    """The pages that differ between two builds, each named, with the first byte at which two texts part."""
    found: list[str] = []
    for page in sorted(first.keys() | second.keys()):
        if page not in second or page not in first:
            held = seeds[0] if page in first else seeds[1]
            found.append(f"{page}: built under PYTHONHASHSEED={held} and not under the other seed")
        elif first[page] != second[page]:
            at = next((i for i, (a, b) in enumerate(zip(first[page], second[page], strict=False)) if a != b),
                      min(len(first[page]), len(second[page])))
            found.append(f"{page}: differs between PYTHONHASHSEED={seeds[0]} and PYTHONHASHSEED={seeds[1]}, from "
                         f"byte {at}; a generator is writing something that follows the hash seed (a set's order, "
                         f"most often): sort it")
    return found


def reproducibility_problems(root: Path, command: Callable[[Path, Path], list[str]] = build_command, *,
                             seeds: tuple[str, ...] = SEEDS) -> list[str]:
    """Build the pages of ``root`` with ``command`` in one fresh process per seed, all at once, and compare what each
    wrote with the first seed's; a page is named once, against the first seed it differs under. A build that fails is a
    problem too (it wrote nothing to compare)."""
    with tempfile.TemporaryDirectory(prefix="mime-docs-w8-") as scratch:
        runs: list[tuple[str, Path, subprocess.Popen[str]]] = []
        for seed in seeds:
            out = Path(scratch) / f"seed-{seed}"
            process = subprocess.Popen(command(root, out), env={**os.environ, "PYTHONHASHSEED": seed},
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            runs.append((seed, out, process))
        built: list[dict[str, bytes]] = []
        failed: list[str] = []
        for seed, out, process in runs:
            output, _ = process.communicate()
            if process.returncode != 0:
                tail = " | ".join(line for line in output.strip().splitlines()[-3:])
                failed.append(f"the build under PYTHONHASHSEED={seed} exited {process.returncode}: {tail}")
            built.append(_read(out) if out.is_dir() else {})
        if failed:
            return failed
        named: dict[str, str] = {}
        for seed, other in zip(seeds[1:], built[1:], strict=True):
            for problem in differences(built[0], other, (seeds[0], seed)):
                named.setdefault(problem.split(":", 1)[0], problem)
        return [named[page] for page in sorted(named)]


__all__ = ["SEEDS", "build_command", "differences", "reproducibility_problems"]
