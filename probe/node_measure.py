#!/usr/bin/env python3
"""node_measure.py: the node-level measurement template ("measure through the real node").

A measurement a decision or a PR's claim rests on runs through the built graph (`built.run`, or the node's own step),
never through a stand-alone bench of a solver function: a bench builds matrices the node never builds and measures the
bare solve's memory. This script builds an experiment, runs it, and records, for ONE size per process (so that the peak
memory is that size's and no earlier one's):

  * seconds to build, the first run (it compiles), and the steady runs (the same run again, compiled): time per step is
    the steady run's seconds over its steps, the first run's extra is the compile;
  * the process's peak memory: VmHWM from /proc/self/status, and, because VmHWM is absent off Linux and cannot be
    attributed to a phase, a 5 ms VmRSS sampler thread (its maximum is reported beside it, and is the peak where VmHWM is
    absent). Where /proc/self/clear_refs accepts a reset, each phase's own peak is read as well;
  * the outputs a claim rests on: `--final node.field` (the state after the last step) and `--output
    observation.output` (an observation's output, e.g. `swim_speed:ball.speed`; name it in `--record` for a per-step
    series): shape, dtype, finiteness, min/max/mean/last and, up to `--max-values`, the values themselves;
  * the environment: the device, the JAX version, the dtype, the threads and the CPU set, so that a number is read at its
    operating point.

  * the **trace counter** (item 25, widened after the review): every JAX trace and XLA compile is counted per phase and per
    function name (`jax.monitoring`'s `jaxpr_trace_duration` and `backend_compile_duration`, which carry the function's
    name), and MADDENING's `scan_trace_count` is read at the end. The run FAILS (exit 3, "RE-TRACED") if
      - a steady run (forward or reverse) traced or compiled anything: a timing that re-traces each repeat measures the tracer;
      - a FIRST run (forward or reverse) traced one function name more than `MAX_SAME_NAME_TRACES` (2) times: a first run
        legitimately traces the graph's scan twice (a chunk and its remainder), so a function traced three times or more in it
        was traced again by mistake (a jit built per call, a closure over a new value). Its resolution is stated: ONE
        re-trace of a function in the first run is indistinguishable from the same function used at two shapes, and is not
        seen; the plants it is measured against are in the README;
      - the graph's scan program was traced more than `MAX_SCAN_PROGRAMS` (2) times in all, or again after the first run.
    On the scan path (`built.run`) the graph traces its step INSIDE the scan program, and MADDENING's `trace_count` counts
    only direct step calls: it reads 0 there and is reported but not judged; the count that means "the step was traced again"
    on this path is `scan_trace_count`, with the function-name counts of the global counter;
  * the **reverse mode** (`--reverse node.leaf`, repeatable): the gradient, in those leaves, of the sum of the final state's
    floating fields through `--reverse-steps` steps of MIME's differentiated run (`differentiated_run`, checkpointed unless
    `--no-checkpoint`; the plain chunk where the tree has none), jitted once; first run (compile), steady runs and peak in
    phases of their own (`reverse_first_run`, `reverse_steady_run`; `forward_peak_mib` and `reverse_peak_mib` at the top of
    `peak_memory`), so that a forward timing is never read as a reverse one;
  * the **memory check** (`--estimate-json FILE`, a `mem_estimate.py --output-json` or `--summary-json`): the run's process peak
    against the estimate and the bound the route was read from; recorded as `memory_check` and, if the peak is over the bound,
    printed as UNDERESTIMATE and the exit status is 4 (a run that exceeds what it was routed by is flagged, never silent);
  * the **environment of the timing**: load average at the start and the end, the CPU model, the threads; a load per CPU
    over 0.5 at either end is printed as a warning and recorded (`timing_on_loaded_machine`): a time on it is not a timing
    claim.

JSON out (`--output-json`); one summary line on stdout. It runs locally (in the repository's environment, never with the
repository's folder as the working directory) or as a probe's script (`python probe/node_measure.py ...` on a branch
`probe/<name>`; copy it there).

    node_measure.py --case sphere_settling:sphere_settling --path examples --arg radius_um=10 --steps 30 \\
        --final ball.position --output-json out.json
    node_measure.py --case mrs_cases:mrs_tube --path tests/saved --steps 8 --dtype float32 --repeats 3 ...

`--case module:function` is a factory returning an `Experiment` (built here) or an already built experiment;
`--arg name=value` passes keyword arguments (value parsed as JSON, else a string); `--path` adds a directory to
`sys.path`. `--runner module:function` replaces `built.run(duration)` by `function(built, steps)` returning a result with
the same reading interface, for a case that drives a node's own step; the timing then covers the whole call.

Declared limits: the time is wall clock on a shared machine (take the min over `--repeats`, not the mean); peak memory
is the process's, so it includes the build, the compile and the interpreter, and is an upper bound on the node's own.
Run one size per process and compare sizes across processes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

SAMPLE_INTERVAL = 0.005


def status_kib(field: str) -> int | None:
    """A kiB field of /proc/self/status (VmHWM, VmRSS), None off Linux."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def reset_peak() -> bool:
    """Reset the kernel's high-water mark so the next VmHWM read is a phase's own; False where that is not allowed."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


class RssSampler:
    """The process's resident set sampled every 5 ms in a thread: its maximum is a peak where VmHWM is absent."""

    def __init__(self) -> None:
        self.peak_kib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = status_kib("VmRSS")
            if rss is not None and rss > self.peak_kib:
                self.peak_kib = rss
            time.sleep(SAMPLE_INTERVAL)

    def start(self) -> RssSampler:
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.peak_kib


TRACE_EVENTS = ("/jax/core/compile/jaxpr_trace_duration", "/jax/core/compile/backend_compile_duration")
#: A first run traces the graph's scan twice (a chunk and its remainder); a function traced more often than this in one
#: phase was traced again by mistake. One re-trace (a second trace of a function) is not seen: it looks like a second shape.
MAX_SAME_NAME_TRACES = 2
#: The graph's scan program is traced at most twice in a build (the chunk and its remainder).
MAX_SCAN_PROGRAMS = 2


class TraceCounter:
    """Counts JAX traces and XLA compiles process-wide (a listener on `jax.monitoring`), traces also by function name; a
    phase reads the difference."""

    def __init__(self) -> None:
        self.traces = 0
        self.compiles = 0
        self.by_name: dict[str, int] = {}
        import jax

        jax.monitoring.register_event_duration_secs_listener(self._seen)

    def _seen(self, event: str, duration: float, **kwargs: Any) -> None:
        if event == TRACE_EVENTS[0]:
            self.traces += 1
            name = str(kwargs.get("fun_name", "?"))
            self.by_name[name] = self.by_name.get(name, 0) + 1
        elif event == TRACE_EVENTS[1]:
            self.compiles += 1


COUNTER: TraceCounter | None = None


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def load_state() -> dict[str, Any]:
    """The load average now and what a CPU's share of it is; a share over 0.5 is a loaded machine."""
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return {}
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return {"load_1m": one, "load_5m": five, "load_15m": fifteen, "cpus_usable": cpus, "load_per_cpu_1m": one / cpus}


def split_leaf(params: Any, name: str) -> tuple[str, str]:
    """`node.leaf` as the node and the leaf it names in `params["nodes"]` (a leaf's own name may hold a dot)."""
    for index, character in enumerate(name):
        if character == "." and name[:index] in params["nodes"] and name[index + 1:] in params["nodes"][name[:index]]:
            return name[:index], name[index + 1:]
    raise SystemExit(f"no leaf {name!r} (node.leaf) in the parameter tree")


def reverse_function(graph: Any, steps: int, wrt: list[str], checkpoint: bool, params: Any) -> Any:
    """`(values, state, params) -> (value, gradient)`: the sum of the final state's floating fields through `steps` steps
    of the differentiated run, in the leaves `wrt` (`node.leaf`)."""
    import jax
    import jax.numpy as jnp
    from mime.build import _maddening as maddening

    keys = [split_leaf(params, name) for name in wrt]
    if hasattr(maddening, "differentiated_run"):
        run = maddening.differentiated_run(graph, steps, (), (), checkpoint=checkpoint)

        def chunk(state: Any, params: Any) -> Any:
            return run(state, params)[0]
    else:
        chunk = maddening.chunk_function(graph, steps)

    def loss(values: Any, state: Any, params: Any) -> Any:
        nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
        for (node, leaf), value in zip(keys, values, strict=True):
            nodes[node][leaf] = value
        final = chunk(state, {**params, "nodes": nodes})
        return sum((jnp.sum(x) for x in jax.tree_util.tree_leaves(final) if jnp.issubdtype(x.dtype, jnp.floating)),
                   start=jnp.zeros(()))

    return jax.jit(jax.value_and_grad(loss))


def mib(kib: int | None) -> float | None:
    return None if kib is None else round(kib / 1024.0, 1)


class Phase:
    """One timed phase: seconds, and its own memory peak (VmHWM after a reset, else the sampler's maximum)."""

    def __init__(self, name: str, report: dict[str, Any], can_reset: bool) -> None:
        self.name, self.report, self.can_reset = name, report, can_reset

    def __enter__(self) -> Phase:
        self.reset = self.can_reset and reset_peak()
        self.sampler = RssSampler().start()
        self.before = (COUNTER.traces, COUNTER.compiles) if COUNTER else (0, 0)
        self.names_before = dict(COUNTER.by_name) if COUNTER else {}
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        seconds = time.perf_counter() - self.started
        sampled = self.sampler.stop()
        entry: dict[str, Any] = {"seconds": seconds, "rss_sampled_peak_mib": mib(sampled), "rss_end_mib": mib(status_kib("VmRSS")),
                                 "traces": (COUNTER.traces - self.before[0]) if COUNTER else None,
                                 "compiles": (COUNTER.compiles - self.before[1]) if COUNTER else None}
        if COUNTER:
            entry["traces_by_name"] = {name: count - self.names_before.get(name, 0) for name, count in COUNTER.by_name.items()
                                       if count > self.names_before.get(name, 0)}
        if self.reset:
            entry["hwm_phase_mib"] = mib(status_kib("VmHWM"))
        self.report.setdefault(self.name, []).append(entry)


def load(spec: str, path: str | None) -> Any:
    if path:
        sys.path.insert(0, str(Path(path).resolve()))
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def parse_arguments(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        key, _, text = item.partition("=")
        try:
            parsed[key] = json.loads(text)
        except json.JSONDecodeError:
            parsed[key] = text
    return parsed


def describe(array: Any, max_values: int) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(array)
    entry: dict[str, Any] = {"shape": list(values.shape), "dtype": str(values.dtype)}
    if values.dtype.kind not in "fiub" or values.size == 0:
        return entry
    entry["finite"] = bool(np.isfinite(values).all())
    flat = values.astype(np.float64).ravel()
    entry.update(min=float(flat.min()), max=float(flat.max()), mean=float(flat.mean()), last=float(flat[-1]))
    if values.size <= max_values:
        entry["values"] = values.astype(np.float64).tolist()
    return entry


def as_array(thing: Any) -> Any:
    """The numbers of a series (`.raw`) or a reading (`.value`), else the thing itself."""
    for name in ("raw", "value"):
        if hasattr(thing, name):
            inner = getattr(thing, name)
            return getattr(inner, "magnitude", inner)
    return thing


def read_outputs(result: Any, finals: list[str], outputs: list[str], max_values: int) -> dict[str, Any]:
    read: dict[str, Any] = {}
    for spec in finals:
        node, _, field = spec.rpartition(".")
        read[f"final:{spec}"] = describe(result.final[node][field], max_values)
    for spec in outputs:
        observation, _, name = spec.rpartition(".")
        read[f"output:{spec}"] = describe(as_array(result[observation][name]), max_values)
    return read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="module:function returning an Experiment (or a built one)")
    parser.add_argument("--path", help="a directory to put on sys.path before importing the case")
    parser.add_argument("--arg", action="append", default=[], help="name=value for the factory (JSON value, else string)")
    parser.add_argument("--runner", help="module:function(built, steps) replacing built.run (the node's own step)")
    parser.add_argument("--steps", type=int, required=True, help="base steps per run (one size per process)")
    parser.add_argument("--repeats", type=int, default=2, help="steady runs after the first, which compiles (default 2)")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--history", action="store_true", help="run with history=True")
    parser.add_argument("--record", action="append", default=[], help="a per-step output to record (repeatable)")
    parser.add_argument("--final", action="append", default=[], help="node.field of the state after the last step")
    parser.add_argument("--output", action="append", default=[], help="observation.output to read, e.g. swim_speed:ball.speed")
    parser.add_argument("--reverse", action="append", default=[], help="node.leaf to differentiate in (reverse mode; repeatable)")
    parser.add_argument("--reverse-steps", type=int, help="steps of the reverse run (default: --steps)")
    parser.add_argument("--no-checkpoint", action="store_true", help="reverse mode through the unchecked chunk")
    parser.add_argument("--estimate-json", type=Path, help="a mem_estimate.py output: the peak is checked against its bound")
    parser.add_argument("--max-values", type=int, default=16, help="write the values of an array up to this size")
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args(argv)

    import jax

    global COUNTER
    COUNTER = TraceCounter()
    load_start = load_state()
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    from mime.units import s

    report: dict[str, Any] = {
        "case": arguments.case, "args": parse_arguments(arguments.arg), "steps": arguments.steps, "dtype": arguments.dtype,
        "history": arguments.history, "runner": arguments.runner,
    }
    can_reset = reset_peak()
    baseline = status_kib("VmRSS")
    factory = load(arguments.case, arguments.path)
    phases: dict[str, Any] = {}
    with Phase("build", phases, can_reset):
        made = factory(**report["args"])
        built = made.build() if hasattr(made, "build") else made
    dt = float(built.report.dt)
    runner = load(arguments.runner, arguments.path) if arguments.runner else None

    def run_once() -> Any:
        if runner is not None:
            result = runner(built, arguments.steps)
        else:
            result = built.run(s(arguments.steps * dt), history=arguments.history, record=list(arguments.record))
        jax.block_until_ready(result.final)
        return result

    with Phase("first_run", phases, can_reset):
        result = run_once()
    scan_traces_after_first = int(getattr(built.graph, "scan_trace_count", 0))
    for _ in range(arguments.repeats):
        with Phase("steady_run", phases, can_reset):
            result = run_once()
    steady = [entry["seconds"] for entry in phases.get("steady_run", [])]
    reverse_report: dict[str, Any] | None = None
    if arguments.reverse:
        from mime.build import _maddening as maddening

        reverse_steps = arguments.reverse_steps or arguments.steps
        program = reverse_function(built.graph, reverse_steps, arguments.reverse, not arguments.no_checkpoint,
                                   maddening.graph_params(built.graph))

        def reverse_once() -> Any:
            built.graph.reset_state()
            state, params = maddening.full_state(built.graph), maddening.graph_params(built.graph)
            values = [params["nodes"][node][leaf] for node, leaf in (split_leaf(params, w) for w in arguments.reverse)]
            found = program(values, state, params)
            jax.block_until_ready(found)
            return found

        with Phase("reverse_first_run", phases, can_reset):
            value, gradient = reverse_once()
        for _ in range(arguments.repeats):
            with Phase("reverse_steady_run", phases, can_reset):
                value, gradient = reverse_once()
        import numpy as np

        reverse_steady = [entry["seconds"] for entry in phases["reverse_steady_run"]]
        reverse_report = {
            "wrt": arguments.reverse, "steps": reverse_steps, "checkpoint": not arguments.no_checkpoint,
            "seconds_per_step_steady_min": (min(reverse_steady) / reverse_steps) if reverse_steady else None,
            "seconds_per_step_first": phases["reverse_first_run"][0]["seconds"] / reverse_steps,
            "value": describe(value, arguments.max_values),
            "gradient": [describe(g, arguments.max_values) for g in gradient],
            "gradient_finite": bool(all(np.isfinite(np.asarray(g)).all() for g in gradient)),
        }

    def phase_peak(names: tuple[str, ...]) -> float | None:
        peaks = [entry.get("hwm_phase_mib") or entry["rss_sampled_peak_mib"] or 0.0
                 for name in names for entry in phases.get(name, [])]
        return max(peaks) if peaks else None

    peak_hwm, peak_sampled = status_kib("VmHWM"), max(
        [entry["rss_sampled_peak_mib"] or 0.0 for runs in phases.values() for entry in runs] or [0.0])
    # The trace counter (see the docstring): a steady run that traced or compiled anything; a first run that traced one
    # function name more than MAX_SAME_NAME_TRACES times; the graph's scan program traced more than MAX_SCAN_PROGRAMS times
    # or again after the first run.
    re_traced = [f"{name} #{index}: {entry['traces']} traces, {entry['compiles']} compiles"
                 for name in ("steady_run", "reverse_steady_run") for index, entry in enumerate(phases.get(name, []))
                 if entry["traces"] or entry["compiles"]]
    for name in ("first_run", "reverse_first_run"):
        for entry in phases.get(name, []):
            for function, count in sorted(entry.get("traces_by_name", {}).items()):
                if count > MAX_SAME_NAME_TRACES:
                    re_traced.append(f"{name}: '{function}' was traced {count} times (a first run traces a function at most "
                                     f"{MAX_SAME_NAME_TRACES} times: the scan's chunk and remainder)")
    scan_traces, step_traces = int(getattr(built.graph, "scan_trace_count", 0)), int(getattr(built.graph, "trace_count", 0))
    if scan_traces > scan_traces_after_first:
        re_traced.append(f"the graph's scan trace count grew from {scan_traces_after_first} (after the first run) to {scan_traces}")
    if scan_traces > MAX_SCAN_PROGRAMS:
        re_traced.append(f"the graph's scan program was traced {scan_traces} times in this build (at most {MAX_SCAN_PROGRAMS}: "
                         f"the chunk and its remainder)")
    load_end = load_state()
    loaded = any(state.get("load_per_cpu_1m", 0.0) > 0.5 for state in (load_start, load_end))
    report.update(
        dt_s=dt, phases=phases,
        seconds_per_step_steady_min=(min(steady) / arguments.steps) if steady else None,
        seconds_per_step_first=phases["first_run"][0]["seconds"] / arguments.steps,
        reverse=reverse_report,
        trace_check={"passed": not re_traced, "re_traced": re_traced, "graph_scan_trace_count": scan_traces,
                     "graph_step_trace_count": step_traces, "graph_scan_trace_count_after_first_run": scan_traces_after_first,
                     "limits": {"same_name_traces_in_a_phase": MAX_SAME_NAME_TRACES, "scan_programs_per_build": MAX_SCAN_PROGRAMS},
                     "total_traces": COUNTER.traces, "total_compiles": COUNTER.compiles,
                     "note": "steady runs must trace and compile nothing; a first run traces a function at most twice; "
                             "graph_step_trace_count counts direct step calls and reads 0 on the scan path (not judged)"},
        peak_memory={"vmhwm_mib": mib(peak_hwm), "rss_sampled_peak_mib": peak_sampled, "rss_at_start_mib": mib(baseline),
                     "phase_peaks_from_reset": can_reset,
                     "forward_peak_mib": phase_peak(("build", "first_run", "steady_run")),
                     "reverse_peak_mib": phase_peak(("reverse_first_run", "reverse_steady_run")),
                     "peak_mib": mib(peak_hwm) if peak_hwm is not None else peak_sampled,
                     "source": "VmHWM" if peak_hwm is not None else "5 ms VmRSS sampler"},
        outputs=read_outputs(result, arguments.final, arguments.output, arguments.max_values),
        timing_on_loaded_machine=loaded,
        environment={"devices": [str(device) for device in jax.devices()], "jax": jax.__version__,
                     "python": platform.python_version(), "machine": platform.machine(), "cpu_model": cpu_model(),
                     "cpus_total": os.cpu_count(), "load_at_start": load_start, "load_at_end": load_end,
                     "cpus_usable": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                     "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS",
                                                                      "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION")}},
        note="peak memory is the whole process (build, compile, interpreter): an upper bound on the node's own; "
             "time is wall clock on a shared machine, so take the minimum of the steady runs",
    )
    underestimated = False
    if arguments.estimate_json is not None:
        estimate = json.loads(arguments.estimate_json.read_text(encoding="utf-8"))
        peak_gib = (report["peak_memory"]["peak_mib"] or 0.0) / 1024.0
        raw, bound = float(estimate["estimate_gib"]), float(estimate["bound_gib"])
        underestimated = peak_gib > bound
        report["memory_check"] = {"peak_gib": peak_gib, "raw_estimate_gib": raw, "bound_gib": bound,
                                  "peak_over_raw": peak_gib / raw if raw else None, "exceeds_bound": underestimated,
                                  "exceeds_raw": peak_gib > raw, "families": estimate.get("families"),
                                  "source": "this process's peak (VmHWM), whole run"}
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finite = all(entry.get("finite", True) for entry in report["outputs"].values())
    if reverse_report is not None:
        finite = finite and reverse_report["gradient_finite"]
    print(f"node_measure: {arguments.case} {arguments.steps} steps {arguments.dtype}: "
          f"{report['seconds_per_step_steady_min']} s/step steady (min of {arguments.repeats}), first run "
          f"{report['seconds_per_step_first']:.4g} s/step, peak {report['peak_memory']['peak_mib']} MiB "
          f"({report['peak_memory']['source']}); outputs {'finite' if finite else 'NOT FINITE'}; wrote {arguments.output_json}")
    if reverse_report is not None:
        print(f"node_measure: reverse in {', '.join(arguments.reverse)}: {reverse_report['seconds_per_step_steady_min']} s/step "
              f"steady, first {reverse_report['seconds_per_step_first']:.4g} s/step, peak {report['peak_memory']['reverse_peak_mib']} "
              f"MiB (forward {report['peak_memory']['forward_peak_mib']} MiB)")
    if loaded:
        print(f"node_measure: WARNING the machine was loaded (load per CPU {load_start.get('load_per_cpu_1m', 0):.2f} at the "
              f"start, {load_end.get('load_per_cpu_1m', 0):.2f} at the end): the times are not a timing claim")
    if arguments.estimate_json is not None:
        check = report["memory_check"]
        print(f"node_measure: memory {check['peak_gib']:.2f} GiB against estimate {check['raw_estimate_gib']:.2f}, bound "
              f"{check['bound_gib']:.2f}" + (f": UNDERESTIMATE, the peak is over the bound the route was read from "
                                             f"(ratio to the raw estimate {check['peak_over_raw']:.2f}); add it to the calibration"
                                             if underestimated else ""))
    if re_traced:
        print("node_measure: RE-TRACED, the measurement is invalid: " + "; ".join(re_traced))
        return 3
    if underestimated:
        return 4
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
