#!/usr/bin/env python3
"""mem_estimate.py: the memory a probe will need and the route that follows, before the probe runs.

A probe is a measurement too heavy for a laptop's queue; some are heavier than a runner. This script says how much memory a
case needs and where it can run, from the compiled program and not from a guess, so that a probe over a runner's memory stops
at its first step with its route instead of being killed an hour in (a standard runner has 16 GB; 6 of 113 probe runs were
killed by memory).

    python .github/probe/mem_estimate.py --case module:function [--path DIR] [--arg name=value]... [--dtype float32]
        [--steps 10] [--wrt node.leaf]... [--history N] [--output-json FILE] [--require actions|local]
    python .github/probe/mem_estimate.py --from-file probe/estimate.txt --arm "$PROBE_ARM" --require actions

``--case`` is a factory returning an ``Experiment`` or a built experiment (as ``node_measure.py`` takes it). The estimate is

* the **program's**: XLA's compiled memory analysis (``jax.jit(f).lower(...).compile().memory_analysis()``) of a chunk of
  ``--steps`` steps of the graph, and with ``--wrt`` of the reverse-mode derivative of that chunk in the named leaves
  (``node.leaf``; ``differentiated_run`` with checkpointing where the tree has it, else the plain chunk), on abstract shapes:
  nothing of the run is allocated. Its bytes are temporaries, arguments and outputs less what aliases, plus code;
* with the **host's**: the process's resident set after the build (the build's own matrices and tables stay), the build's
  own peak, and ``--history N`` recorded steps of the state;
* with the nodes' **own estimates** where a node has one (``memory_estimate()``, past the fixed part);
* the **fixed part** of a process (the interpreter, JAX and MIME imported, about 0.7 GiB).

The estimate is the largest of the build's peak and the run's (resident set after the build plus the program's bytes), and
of the nodes' own. **It is multiplied by the calibration factor** of ``mem_calibration.json`` before the route is read: the
cases the estimate was checked against, each with the factor by which its measured peak lay over its estimate, and the
largest of them (never under 1: an underestimate is the failure that matters). The route:

* up to 12 GiB: ``actions`` (a ``probe/<name>`` branch, the probe workflow);
* up to 48 GiB: ``local`` (the maintainer's heavy slot, one at a time, under a 48 GiB cap, for memory-bound correctness and
  never timing claims);
* over: ``modal`` (the maintainer's word only).

What it bounds and what it does not: the calibration cases are few and of one family (the MRS drag and its near field on a
CPU); a case of another family is outside what the factor was measured on, and the output says how many cases it rests on.
It does not see memory a node holds off the XLA program (a callback's, a host-side table made during the run). An estimate
that jumps unexpectedly is a reason to look for the probe's own bug before sending it to a bigger machine.

``--from-file`` reads a probe's ``probe/estimate.txt``: one line per case, ``<arm or *>: <the arguments above>``, blank lines
and ``#`` comments skipped; the lines of the arm in ``--arm`` and of ``*`` are run, each in a fresh process (its own peak),
and the route is the highest of them. Exit: 0 the route is within ``--require`` (or none was asked), 1 beyond it (the route
is printed), 2 usage or an error in the case.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

GIB = 2**30
#: The fixed part of a process: the interpreter with JAX and MIME imported (0.65 to 0.73 GiB measured, W3b's fit).
FIXED_GIB = 0.7
#: The routes' limits in GiB: a standard runner has 16 GB and keeps room for its system; the laptop's heavy slot is capped.
ACTIONS_GIB = 12.0
LOCAL_GIB = 48.0
ROUTES = ("actions", "local", "modal")
CALIBRATION = Path(__file__).with_name("mem_calibration.json")


def route_of(gib: float) -> str:
    return "actions" if gib <= ACTIONS_GIB else "local" if gib <= LOCAL_GIB else "modal"


def kib_of(field: str) -> int:
    """A kiB field of /proc/self/status (VmHWM, VmRSS); 0 off Linux."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def calibration() -> dict[str, Any]:
    """The calibration table: ``factor`` (never under 1) and ``cases``, each with its estimate and measured peak."""
    try:
        table = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"factor": 1.0, "cases": [], "missing": True}
    return table


def program_bytes(program: Any, *args: Any) -> dict[str, int]:
    """XLA's buffer assignment of ``program`` on abstract ``args``, compiled and not run, in bytes."""
    import jax

    stats = jax.jit(program).lower(*args).compile().memory_analysis()
    parts = {name: int(getattr(stats, f"{name}_size_in_bytes")) for name in ("temp", "argument", "output", "alias",
                                                                              "generated_code")}
    parts["total"] = parts["temp"] + parts["argument"] + parts["output"] - parts["alias"] + parts["generated_code"]
    return parts


def abstract(tree: Any) -> Any:
    """``tree`` with each array leaf as its shape and dtype only."""
    import jax
    import jax.numpy as jnp

    def shape_of(leaf: Any) -> Any:
        dtype = leaf.dtype if hasattr(leaf, "dtype") else jnp.result_type(leaf)
        return jax.ShapeDtypeStruct(jnp.shape(leaf), dtype)

    return jax.tree_util.tree_map(shape_of, tree)


def nbytes(tree: Any) -> int:
    import jax

    return sum(int(math.prod(leaf.shape)) * leaf.dtype.itemsize for leaf in jax.tree_util.tree_leaves(tree))


def split_leaf(params: Any, name: str) -> tuple[str, str]:
    """`node.leaf` as the node and the leaf it names in `params["nodes"]` (a leaf's own name may hold a dot)."""
    for index, character in enumerate(name):
        if character == "." and name[:index] in params["nodes"] and name[index + 1:] in params["nodes"][name[:index]]:
            return name[:index], name[index + 1:]
    raise SystemExit(f"no leaf {name!r} (node.leaf) in the parameter tree")


def reverse_function(graph: Any, steps: int, wrt: list[str], maddening: Any, checkpoint: bool = True,
                     params: Any = None) -> Any:
    """The value and gradient in the leaves ``wrt`` of the sum of the final state's floating fields, through a chunk of
    ``steps`` steps: ``(values, state, params) -> (value, gradient)``."""
    import jax
    import jax.numpy as jnp

    keys = [split_leaf(params, name) for name in wrt]
    if checkpoint and hasattr(maddening, "differentiated_run"):
        run = maddening.differentiated_run(graph, steps, (), (), checkpoint=True)

        def chunk(state: Any, params: Any) -> Any:
            return run(state, params)[0]
    else:
        chunk = maddening.chunk_function(graph, steps)

    def loss(values: Any, state: Any, params: Any) -> Any:
        nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
        for (node, leaf), value in zip(keys, values, strict=True):
            nodes[node][leaf] = value
        final = chunk(state, {**params, "nodes": nodes})
        return sum((jnp.sum(x) for x in jax.tree_util.tree_leaves(final)
                    if jnp.issubdtype(x.dtype, jnp.floating)), start=jnp.zeros(()))

    return jax.value_and_grad(loss)


def parse_arguments(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        key, _, text = item.partition("=")
        try:
            parsed[key] = json.loads(text)
        except json.JSONDecodeError:
            parsed[key] = text
    return parsed


def estimate_case(arguments: argparse.Namespace) -> dict[str, Any]:
    """The estimate of one case, in GiB, with its parts. Builds the case in this process (so run it in its own)."""
    import jax

    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    if arguments.path:
        sys.path.insert(0, str(Path(arguments.path).resolve()))
    module, _, name = arguments.case.partition(":")
    factory = getattr(importlib.import_module(module), name)
    started_kib = kib_of("VmRSS")
    made = factory(**parse_arguments(arguments.arg))
    built = made.build() if hasattr(made, "build") else made
    graph = built.graph
    from mime.build import _maddening as maddening

    after_build_kib = kib_of("VmRSS")
    build_peak_kib = kib_of("VmHWM")
    graph.reset_state()
    state = abstract(maddening.full_state(graph))
    params = abstract(maddening.graph_params(graph))
    parts: dict[str, Any] = {"forward": program_bytes(maddening.chunk_function(graph, arguments.steps), state, params)}
    run_bytes = parts["forward"]["total"]
    if arguments.wrt:
        values = [params["nodes"][node][leaf] for node, leaf in (split_leaf(params, w) for w in arguments.wrt)]
        parts["reverse"] = program_bytes(reverse_function(graph, arguments.steps, arguments.wrt, maddening,
                                                                  not arguments.plain, params),
                                         values, state, params)
        run_bytes = parts["reverse"]["total"]
    history = nbytes(state) * arguments.history
    own: dict[str, float] = {}
    for node_name in graph.node_names:
        node = graph.get_node(node_name)
        method = getattr(node, "memory_estimate", None)
        if callable(method):
            own[node_name] = float(method()) + FIXED_GIB * GIB
    resident = after_build_kib * 1024
    candidates = {"build peak": build_peak_kib * 1024,
                  "run (resident after the build + the program's bytes + history)": resident + run_bytes + history,
                  **{f"node {name}'s own estimate + fixed part": value for name, value in own.items()}}
    largest = max(candidates, key=lambda key: candidates[key])
    return {
        "case": arguments.case, "args": parse_arguments(arguments.arg), "steps": arguments.steps, "dtype": arguments.dtype,
        "wrt": arguments.wrt, "history_steps": arguments.history,
        "program_bytes": parts, "state_bytes": nbytes(state), "history_bytes": history,
        "rss_before_build_gib": started_kib / 2**20, "rss_after_build_gib": after_build_kib / 2**20,
        "build_peak_gib": build_peak_kib / 2**20, "node_estimates_gib": {k: v / GIB for k, v in own.items()},
        "candidates_gib": {k: v / GIB for k, v in candidates.items()}, "dominant": largest,
        "estimate_gib": max(candidates.values()) / GIB,
        "estimator_peak_gib": kib_of("VmHWM") / 2**20,
    }


def judge(found: dict[str, Any]) -> dict[str, Any]:
    """The estimate with the calibration factor applied, and the route."""
    table = calibration()
    factor = max(1.0, float(table.get("factor", 1.0)))
    gib = found["estimate_gib"] * factor
    found.update(calibration_factor=factor, calibration_cases=len(table.get("cases", [])),
                 calibration_missing=bool(table.get("missing")), bound_gib=gib, route=route_of(gib))
    return found


def describe(found: dict[str, Any]) -> str:
    lines = [f"mem_estimate: {found['case']} {found['args'] or ''} {found['dtype']}, {found['steps']} steps"
             + (f", reverse in {', '.join(found['wrt'])}" if found["wrt"] else ", forward")]
    for kind, parts in found["program_bytes"].items():
        lines.append(f"  {kind} program (XLA, compiled not run): {parts['total'] / GIB:.2f} GiB "
                     f"(temporaries {parts['temp'] / GIB:.2f}, arguments {parts['argument'] / GIB:.2f}, "
                     f"outputs {parts['output'] / GIB:.2f})")
    lines.append(f"  host: {found['rss_after_build_gib']:.2f} GiB resident after the build (peak {found['build_peak_gib']:.2f})"
                 + (f", history {found['history_bytes'] / GIB:.2f} GiB" if found["history_bytes"] else ""))
    for name, value in found["node_estimates_gib"].items():
        lines.append(f"  node {name}'s own estimate: {value:.2f} GiB (with the fixed part)")
    lines.append(f"  estimate {found['estimate_gib']:.2f} GiB (dominant: {found['dominant']})")
    note = "NO CALIBRATION TABLE, factor 1" if found["calibration_missing"] else \
        f"calibration factor {found['calibration_factor']:.2f} over {found['calibration_cases']} cases"
    lines.append(f"  x {note} = {found['bound_gib']:.2f} GiB; route: {found['route']}"
                 + ("" if found["route"] == "actions" else
                    " (Modal: needs his word)" if found["route"] == "modal" else " (the local heavy slot, under the cap)"))
    return "\n".join(lines)


def lines_of(path: Path, arm: str) -> list[list[str]]:
    """The argument lists of ``path``'s lines for ``arm`` and ``*``."""
    chosen: list[list[str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        who, colon, rest = text.partition(":")
        if not colon or not rest.strip() or not re.fullmatch(r"\*|[A-Za-z0-9._-]{1,40}", who.strip()):
            raise SystemExit(f"mem_estimate: {path}:{number}: expected '<arm or *>: <arguments>'")
        if who.strip() in (arm, "*"):
            chosen.append(shlex.split(rest))
    return chosen


def from_file(path: Path, arm: str, require: str | None) -> int:
    worst = 0
    for index, line in enumerate(lines_of(path, arm), 1):
        done = subprocess.run([sys.executable, __file__, *line], text=True,
                              capture_output=True, check=False)
        route = next((w for w in ROUTES if f"route: {w}" in done.stdout), None)
        print(done.stdout, end="")
        if done.returncode >= 128 or done.returncode < 0 or route is None:
            reason = (f"was killed (signal {abs(done.returncode) % 128})" if done.returncode >= 128 or done.returncode < 0
                      else f"failed ({done.returncode}): {done.stderr.strip().splitlines()[-1:] or ''}")
            print(f"mem_estimate: line {index} {reason}: "
                  f"{'it ran out of memory, so the case is beyond a runner' if done.returncode in (-9, 137) else 'see above'}",
                  file=sys.stderr)
            if done.returncode in (-9, 137):
                route = "local"
            else:
                return 2
        worst = max(worst, ROUTES.index(route))
    print(f"mem_estimate: {path} for arm {arm or '(none)'}: route {ROUTES[worst]}")
    if require is not None and worst > ROUTES.index(require):
        print(f"mem_estimate: this probe needs the {ROUTES[worst]} route, beyond {require}"
              + (": Modal: needs his word" if ROUTES[worst] == "modal" else
                 ": run it through the local heavy slot (probe_launch.sh), not on a runner"), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", help="module:function returning an Experiment (or a built one)")
    parser.add_argument("--path", help="a directory to put on sys.path before importing the case")
    parser.add_argument("--arg", action="append", default=[], help="name=value for the factory (JSON value, else string)")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--steps", type=int, default=10, help="steps of the chunk the program is compiled for (default 10)")
    parser.add_argument("--wrt", action="append", default=[], help="node.leaf to differentiate in (reverse mode; repeatable)")
    parser.add_argument("--plain", action="store_true", help="reverse mode through the plain chunk, without checkpointing")
    parser.add_argument("--history", type=int, default=0, help="recorded steps of the state held on the host")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--from-file", type=Path, help="a probe's estimate.txt (lines '<arm or *>: <arguments>')")
    parser.add_argument("--arm", default=os.environ.get("PROBE_ARM", ""), help="the arm whose lines --from-file runs")
    parser.add_argument("--require", choices=("actions", "local"), help="exit 1 if the route is beyond this one")
    arguments = parser.parse_args(argv)
    if arguments.from_file is not None:
        return from_file(arguments.from_file, arguments.arm, arguments.require)
    if not arguments.case:
        parser.error("give --case or --from-file")
    found = judge(estimate_case(arguments))
    print(describe(found))
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if arguments.require is not None and ROUTES.index(found["route"]) > ROUTES.index(arguments.require):
        print(f"mem_estimate: route {found['route']} is beyond {arguments.require}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
