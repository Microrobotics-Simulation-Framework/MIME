# SPDX-License-Identifier: Apache-2.0
"""Compare a command's arrays between a base ref and this checkout, bit for bit, in one job.

    python .github/probe/bit_compare.py --base origin/release/v0.3.0 --programs a b c \
        --command 'python {head}/probe/dump.py {program} {out}' --output probe/out/bit_compare.json

For each named program the command runs twice, once in a worktree of the base ref with its own virtual environment
(`uv sync --locked --extra verify`, then the lab fixture) and once in this checkout (HEAD), each time pinned to the same
CPU set with `taskset` and given one BLAS thread. The command is run through the shell in the tree's root with the
tree's `.venv` first on `PATH`, and must write an NPZ file at `{out}`. Placeholders: `{program}` the program's name,
`{out}` the NPZ path to write, `{head}` this checkout's root (a script of the probe branch lives there and not in the
base's tree), `{tree}` the root the command runs in. Both runs happen in the same job, on the same machine, so the
one-core-type rule of MIME-ANO-125 holds by construction: a difference between them is not a difference between two
CPU models.

The JSON written holds, per program and per array name: the dtype and shape on each side, `equal` (the same dtype and
shape and `numpy.array_equal`) and `values_equal` (`numpy.array_equal` alone, so a dtype change reads apart from a value
change), and the largest absolute difference where both are numeric of the same shape. NaN is unequal to itself under
`array_equal`; `equal_nan` says whether the two agree with NaNs matching. An array in one NPZ and not the other, or a
program whose command failed on a side, is reported and counts against `all_equal`; the exit status is 0 when the
comparison ran, whatever it found (`--strict` makes a difference exit 1).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: One thread in every library that threads on CPU; XLA's CPU backend by its flags.
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
XLA_SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def run(command: list[str] | str, *, cwd: Path, env: dict[str, str] | None = None, shell: bool = False,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, shell=shell, check=check, text=True)


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def resolve_base(root: Path, base: str) -> str:
    """The base's commit; where the ref is not in the checkout (a probe branch is checked out shallow) it is fetched
    from `origin` by its name without the `origin/` prefix."""
    probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"], cwd=root,
                           capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        return probe.stdout.strip()
    name = base.removeprefix("origin/")
    run(["git", "fetch", "--depth", "1", "origin", name], cwd=root)
    return git(root, "rev-parse", "FETCH_HEAD^{commit}")


def prepare_base(root: Path, commit: str, where: Path) -> Path:
    """A worktree of `commit` at `where/base` with a virtual environment of its own, installed as CI installs a lane."""
    tree = where / "base"
    run(["git", "worktree", "add", "--detach", str(tree), commit], cwd=root)
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(tree / ".venv")}
    run(["uv", "sync", "--locked", "--extra", "verify"], cwd=tree, env=environment)
    run(["uv", "pip", "install", "--no-deps", "--python", str(tree / ".venv" / "bin" / "python"), "./lab_fixture"],
        cwd=tree, env=environment)
    return tree


def tree_env(tree: Path, threads: dict[str, str]) -> dict[str, str]:
    environment = {**os.environ, **threads}
    environment["VIRTUAL_ENV"] = str(tree / ".venv")
    environment["PATH"] = f"{tree / '.venv' / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    environment.pop("PYTHONHOME", None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(tree / ".venv")
    environment["XLA_FLAGS"] = f"{environment.get('XLA_FLAGS', '')} {XLA_SINGLE_THREAD}".strip()
    return environment


def run_program(tree: Path, head: Path, program: str, template: str, cpus: str, out: Path,
                threads: dict[str, str]) -> str | None:
    """Run the command for one program in `tree`, pinned; None where it succeeded, else why it did not."""
    command = template.format(program=program, out=shlex.quote(str(out)), head=shlex.quote(str(head)),
                              tree=shlex.quote(str(tree)))
    pinned = f"taskset -c {shlex.quote(cpus)} sh -c {shlex.quote(command)}"
    done = run(pinned, cwd=tree, env=tree_env(tree, threads), shell=True, check=False)
    if done.returncode != 0:
        return f"the command exited {done.returncode}"
    if not out.is_file():
        return f"the command wrote no file at {out}"
    return None


def compare_arrays(base: np.lib.npyio.NpzFile, head: np.lib.npyio.NpzFile) -> dict[str, object]:
    arrays: dict[str, object] = {}
    for name in sorted(set(base.files) | set(head.files)):
        if name not in base.files or name not in head.files:
            arrays[name] = {"equal": False, "only_in": "base" if name in base.files else "head"}
            continue
        a, b = base[name], head[name]
        entry: dict[str, object] = {
            "dtype_base": str(a.dtype), "dtype_head": str(b.dtype),
            "shape_base": list(a.shape), "shape_head": list(b.shape),
            "values_equal": bool(np.array_equal(a, b)),
        }
        entry["equal"] = bool(entry["values_equal"] and a.dtype == b.dtype and a.shape == b.shape)
        numeric = a.dtype.kind in "fiuc" and b.dtype.kind in "fiuc"
        entry["equal_nan"] = bool(np.array_equal(a, b, equal_nan=True)) if numeric else entry["values_equal"]
        if numeric and a.shape == b.shape and a.size:
            with np.errstate(all="ignore"):
                difference = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
            finite = difference[np.isfinite(difference)]
            entry["max_abs_difference"] = float(finite.max()) if finite.size else None
        arrays[name] = entry
    return arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="the ref to compare against (fetched from origin if absent)")
    parser.add_argument("--programs", nargs="+", required=True, help="the programs' names")
    parser.add_argument("--command", required=True, help="the shell command, with {program}, {out}, {head}, {tree}")
    parser.add_argument("--output", required=True, type=Path, help="where the JSON goes")
    parser.add_argument("--cpus", default=None, help="the CPU list for taskset (default: the lowest CPU this process may use)")
    parser.add_argument("--head", type=Path, default=Path.cwd(), help="this checkout (default: the working directory)")
    parser.add_argument("--keep", action="store_true", help="keep the base worktree and the NPZ files")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any array differs")
    arguments = parser.parse_args(argv)

    head = arguments.head.resolve()
    cpus = arguments.cpus or str(min(os.sched_getaffinity(0)))
    commit = resolve_base(head, arguments.base)
    work = Path(tempfile.mkdtemp(prefix="bit_compare_"))
    base_tree = prepare_base(head, commit, work)
    sides = {"base": base_tree, "head": head}
    report: dict[str, object] = {"base_ref": arguments.base, "base_commit": commit, "head_commit": git(head, "rev-parse", "HEAD"),
                                 "cpus": cpus, "threads": SINGLE_THREAD, "xla_flags": XLA_SINGLE_THREAD, "programs": {}}
    all_equal = True
    programs: dict[str, object] = {}
    for program in arguments.programs:
        outs = {side: work / f"{program}.{side}.npz" for side in sides}
        failures = {side: run_program(tree, head, program, arguments.command, cpus, outs[side], SINGLE_THREAD)
                    for side, tree in sides.items()}
        if any(failures.values()):
            programs[program] = {"status": "command failed", "failures": {s: f for s, f in failures.items() if f}}
            all_equal = False
            continue
        with np.load(outs["base"], allow_pickle=False) as base, np.load(outs["head"], allow_pickle=False) as head_arrays:
            arrays = compare_arrays(base, head_arrays)
        equal = all(entry["equal"] for entry in arrays.values())  # type: ignore[index]
        programs[program] = {"status": "compared", "all_equal": equal, "arrays": arrays}
        all_equal = all_equal and equal
    report["programs"] = programs
    report["all_equal"] = all_equal
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not arguments.keep:
        run(["git", "worktree", "remove", "--force", str(base_tree)], cwd=head, check=False)
        shutil.rmtree(work, ignore_errors=True)
    print(f"bit_compare: {'every array equal' if all_equal else 'a difference or a failure'}; wrote {arguments.output}")
    return 0 if all_equal or not arguments.strict else 1


if __name__ == "__main__":
    sys.exit(main())
