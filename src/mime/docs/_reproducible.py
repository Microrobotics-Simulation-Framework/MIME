"""W8: the pages are the same text under two hash seeds. ``mime docs check`` builds every page twice, each build in a
fresh process under its own ``PYTHONHASHSEED``, and fails on any byte that differs, naming the page.

This bounds one thing: byte-equal pages across two builds under two hash seeds, in one environment. It does not bound a
timestamp, an ordering that follows an object's address (``id()``, a set of objects), the order a directory lists, or a
difference of locale or machine; none of those is planted or tested here."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

#: The two seeds. They are fixed, so a build's two orders are a fixed fact and a control that plants a set's order
#: fails on every run.
SEEDS = ("1", "2")


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
                             seeds: tuple[str, str] = (SEEDS[0], SEEDS[1])) -> list[str]:
    """Build the pages of ``root`` with ``command`` in two fresh processes, one under each seed, and compare what they
    wrote. A build that fails is a problem too (it wrote nothing to compare)."""
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
        return differences(built[0], built[1], seeds)


__all__ = ["SEEDS", "build_command", "differences", "reproducibility_problems"]
