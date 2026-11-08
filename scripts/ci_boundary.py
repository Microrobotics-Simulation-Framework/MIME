#!/usr/bin/env python
"""Which of CI's lanes a run starts, written to ``$GITHUB_OUTPUT`` for the ``shards`` job's other jobs
(``.github/workflows/ci.yml``). Standard library only.

Three lanes run on a push, a manual run and the nightly run (``.github/workflows/nightly.yml``), and on a pull request
only when the pull request can change by itself whether MIME works with them (a decision of 2026-11-08 the maintainer can
reverse, CONTRIBUTING.md, "CI: which lanes a pull request runs"): ``default`` on Python 3.13, ``maddening-main`` and
``openmaterica-main``. The last two swap an upstream for its ``main``, and the first adds a Python between the two that gate
every pull request (3.12, the floor, and 3.14, the newest). A pull request runs them only in the full matrix (the label
``ready``) and only when its diff touches:

- the dependency pins (``PINS``): both lanes and Python 3.13;
- CI itself (``CI``): the workflows, the actions they use, and the scripts they call: both lanes and Python 3.13;
- MIME's explicit adapters to an upstream (``MADDENING_ADAPTERS``, ``OPENMATERICA_ADAPTERS``): that upstream's lane
  alone. The adapters are the files that name the upstream's classes on MIME's behalf. MADDENING is imported by 41 of
  ``src/mime``'s 354 files (about 12 %), spread over the build, the nodes, the testing harness and the docs, so no
  directory stands for the boundary; the rest of that surface is covered by the push and nightly runs, and
  ``tests/test_ci_boundary.py`` fails if MADDENING comes to be imported by most of the package, since this filter would
  then stand for little.

Everything here lives in this file, and ``tests/test_ci_boundary.py`` holds it against the workflows and the code.

The outputs:

- ``maddening``, ``openmaterica``: ``true`` when that lane runs;
- ``exclude``: the matrix entries the ``default`` lane's shard jobs and its completion job leave out (JSON): Python 3.13
  unless it runs, and 3.14 too in a reduced run (a pull request without ``ready``);
- ``lane_exclude``: the entries the ``default`` job under its own name leaves out: Python 3.13 unless it runs. Python
  3.14 stays in a reduced run, where its job refuses by name (a required check never passes by being skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass

#: The dependency pins.
PINS = ("pyproject.toml", "uv.lock")
#: The scripts the workflows call (the Python and shell files under ``scripts/`` that a ``run`` of any workflow names),
#: and the one test file a workflow runs as a script.
CI_SCRIPTS = (
    "scripts/check_confidential.py", "scripts/check_no_skips.py",
    "scripts/check_shard_jobs.py", "scripts/check_shards.py", "scripts/ci_boundary.py", "scripts/ci_ready.py",
    "scripts/compilation_cache_mode.py", "scripts/install_shells.sh", "scripts/latest_attempt.py",
    "scripts/platforms_smoke.py", "scripts/prune_jax_cache.py", "scripts/release.py",
    "scripts/report_test_durations.py", "scripts/run_jobs.py", "scripts/shard_matrix.py", "scripts/stress_select.py",
    "scripts/stress_summary.py", "scripts/typing_check.py",
    "tests/test_ci_lane.py",
)
#: CI: a directory (ending in ``/``) or a file.
CI = (".github/workflows/", ".github/actions/", *CI_SCRIPTS)
#: MIME's adapters to MADDENING: the file that wraps its graph, edge and parameter calls at MIME's boundary, and the
#: node base class every MIME node derives from.
MADDENING_ADAPTERS = ("src/mime/build/_maddening.py", "src/mime/core/_node.py")
#: MIME's adapters to OpenMaterica: the three modules that import ``materica``, which read a record's field at an
#: experiment's conditions.
OPENMATERICA_ADAPTERS = ("src/mime/materials/_read.py", "src/mime/materials/_records.py",
                         "src/mime/materials/_resolver.py")
PYTHON_313 = {"python-version": "3.13"}
PYTHON_314 = {"python-version": "3.14"}


def touches(changed: list[str], paths: tuple[str, ...]) -> bool:
    """Whether any changed path is one of ``paths`` or under one that ends in ``/``."""
    return any(each == path or (path.endswith("/") and each.startswith(path)) for each in changed for path in paths)


@dataclass(frozen=True)
class Lanes:
    maddening: bool
    openmaterica: bool
    python313: bool
    exclude: list[dict[str, str]]
    lane_exclude: list[dict[str, str]]
    why: str


def decide(event: str, full: bool, changed: list[str] | None) -> Lanes:
    """The lanes a run starts: its event, whether it is the full matrix (``ready``, or not a pull request), and the paths
    its pull request changes (``None`` when they could not be read, which counts as everything changed)."""
    if event != "pull_request":
        maddening = openmaterica = python313 = True
        why = f"a {event} runs every lane"
    elif not full:
        maddening = openmaterica = python313 = False
        why = "a pull request without ready runs the reduced set"
    elif changed is None:
        maddening = openmaterica = python313 = True
        why = "the pull request's changed paths could not be read, so every lane runs"
    else:
        pins, ci = touches(changed, PINS), touches(changed, CI)
        madd, openm = touches(changed, MADDENING_ADAPTERS), touches(changed, OPENMATERICA_ADAPTERS)
        maddening, openmaterica, python313 = pins or ci or madd, pins or ci or openm, pins or ci
        touched = [name for name, hit in (("the pins", pins), ("CI", ci), ("MADDENING's adapters", madd),
                                          ("OpenMaterica's adapters", openm)) if hit]
        why = f"the pull request touches {', '.join(touched)}" if touched else (
            "the pull request touches no pin, CI file or adapter")
    exclude = ([] if python313 else [PYTHON_313]) + ([] if full else [PYTHON_314])
    return Lanes(maddening, openmaterica, python313, exclude, [] if python313 else [PYTHON_313], why)


def changed_paths() -> list[str] | None:
    """The files a pull request changes, as ``compilation_cache_mode.py`` reads them: the merge commit against its base's
    tip (the checkout is two deep). A rename counts as the file leaving and the file arriving."""
    done = subprocess.run(["git", "diff", "--name-only", "--no-renames", "HEAD^1", "HEAD"], capture_output=True,
                          text=True, check=False)
    return done.stdout.split() if done.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--full", required=True, choices=("true", "false"), help="the run's mode (the shards job's)")
    args = parser.parse_args(argv)
    lanes = decide(args.event, args.full == "true", changed_paths() if args.event == "pull_request" else [])
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"maddening={str(lanes.maddening).lower()}\nopenmaterica={str(lanes.openmaterica).lower()}\n"
                  f"exclude={json.dumps(lanes.exclude, separators=(',', ':'))}\n"
                  f"lane_exclude={json.dumps(lanes.lane_exclude, separators=(',', ':'))}\n")
    print(f"maddening-main {'runs' if lanes.maddening else 'does not run'}, openmaterica-main "
          f"{'runs' if lanes.openmaterica else 'does not run'}, default on Python 3.13 "
          f"{'runs' if lanes.python313 else 'does not run'}: {lanes.why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
