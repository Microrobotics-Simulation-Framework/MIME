"""Harnesses for the saved form and the compiled run: the round trip through a fresh process, the list of installed
entries a registry walk covers, and the check that every parameter leaf reaches the compiled step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from jax import monitoring
from maddening.core.params import ParamSpec

from mime.build import BuildRefused
from mime.build._maddening import device_array, parameter_leaves, parameter_specs, run_scan
from mime.experiment import BuiltExperiment, Experiment
from mime.saved import DEFAULT_STEPS, DOCUMENT_FILE, ExperimentDocument, dump_document
from mime.testing._graph import history_differences
from mime.units import s

type History = dict[str, dict[str, np.ndarray]]

_LIST_ENTRIES = """
import importlib.metadata, json
import mime
for point in importlib.metadata.entry_points(group="mime.entries"):
    point.load()
from mime.registry import registered
print(json.dumps([[r.kind.value, r.name, r.version] for r in registered()]))
"""


def _fresh_environment() -> dict[str, str]:
    """This process's environment, with its precision passed on: x64 is process-wide, and a fresh process must
    build in the precision this one did for its bits to be comparable."""
    return {**os.environ, "JAX_ENABLE_X64": "1" if cast(Any, jax.config).jax_enable_x64 else "0"}


def installed_entries() -> list[tuple[str, str, str]]:
    """``(kind, name, version)`` of every entry registered once MIME and every package providing a ``mime.entries``
    namespace are imported, read in a fresh process so that entries a test registered on its own do not count."""
    done = subprocess.run([sys.executable, "-c", _LIST_ENTRIES], capture_output=True, text=True, check=True,
                          env=_fresh_environment())
    return [(kind, name, version) for kind, name, version in json.loads(done.stdout.splitlines()[-1])]


def covered(document: ExperimentDocument) -> tuple[set[tuple[str, str, str]], set[str]]:
    """The entries ``(kind, name, version)`` and the kinds of thing a document's experiment uses, a wall kind an
    interaction names by its ``wall_kind`` setting among them."""
    entries = {("interaction", d["name"], d["version"]) for d in document["interactions"]}
    entries |= {key for d in document["interactions"] for key in _wall_kind(d.get("settings", {}).get("wall_kind"))}
    entries |= {("observation", d["name"], d["version"]) for d in document["observations"]}
    if document["controller"] is not None:
        entries.add(("controller", document["controller"]["name"], document["controller"]["version"]))
    kinds: set[str] = set()
    for thing in document["things"].values():
        kinds.add(thing["kind"])
        entries |= {("dynamics", e["name"], e["version"]) for e in thing.get("dynamics", {}).values()}
        entries |= {("field provider", e["name"], e["version"]) for e in thing.get("fields", {}).values()}
        entries |= {("drive", e["name"], e["version"]) for e in thing.get("commands", {}).values()}
        if "geometry" in thing:
            entries.add(("geometry", thing["geometry"]["name"], thing["geometry"]["version"]))
        entries |= {("geometry", p["solid"]["name"], p["solid"]["version"]) for p in thing.get("parts", {}).values()}
        entries |= {("dynamics", s["name"], s["version"]) for s in thing.get("chain", [])}
    return entries, kinds


def _wall_kind(written: object) -> set[tuple[str, str, str]]:
    """The wall kind a setting ``wall_kind`` names, ``"<name>"`` (its one registered version) or ``"<name>
    <version>"``, as an entry key; none where it names none."""
    from mime.registry import EntryKind, versions_of

    if not isinstance(written, str) or not written.split():
        return set()
    name, *version = written.split()
    held = version or versions_of(EntryKind.WALL_KIND, name)
    return {(EntryKind.WALL_KIND.value, name, held[0])} if len(held) == 1 else set()


@dataclass(frozen=True)
class FreshLoad:
    """What a fresh process found: its exit status and output, and when the load succeeded the derived hash it
    rebuilt and the first steps it ran."""

    returncode: int
    output: str
    derived_hash: str | None
    history: History | None


def load_in_fresh_process(folder: Path, *, document_only: bool = False) -> FreshLoad:
    """Load ``folder`` in a new Python process (``python -m mime.testing._fresh_load``), rebuild, and run the lock's
    reference steps. With ``document_only`` it reads the document alone and builds it, for an experiment whose
    build is refused and so has no lock."""
    command = [sys.executable, "-m", "mime.testing._fresh_load", str(folder)]
    done = subprocess.run(command + (["--document-only"] if document_only else []), capture_output=True, text=True,
                          check=False, env=_fresh_environment())
    output = done.stdout + done.stderr
    result = folder / "fresh_load.json"
    if done.returncode != 0 or not result.exists():
        return FreshLoad(done.returncode, output, None, None)
    found = json.loads(result.read_text())
    history: History = {}
    with np.load(folder / "fresh_load.npz", allow_pickle=False) as held:
        for key in held.files:
            node, name = key.split("/", 1)
            history.setdefault(node, {})[name] = np.asarray(held[key])
    return FreshLoad(done.returncode, output, found["derived_hash"], history)


def first_steps(built: BuiltExperiment, steps: int) -> History:
    """The first ``steps`` steps of ``built``, run in whole chunks when its run is chunked, the rest dropped."""
    chunk = built.report.run.chunk or 1
    history = built.run(s(-(-steps // chunk) * chunk * cast(float, built.report.dt)), history=True).history
    assert history is not None
    return {node: {name: value[:steps] for name, value in fields.items()} for node, fields in history.items()}


def round_trip_failures(experiment: Experiment, folder: Path, steps: int = DEFAULT_STEPS) -> list[str]:
    """Build, save into ``folder``, load in a fresh process, rebuild: every way the loaded experiment differs from
    the built one in its derived hash or, bit for bit, its first ``steps`` steps. Empty when it round-trips."""
    built = experiment.build()
    lock = built.save(folder, steps=steps)
    reference = first_steps(experiment.build(), steps)
    fresh = load_in_fresh_process(folder)
    if fresh.returncode != 0 or fresh.history is None:
        return [f"the fresh process failed (exit {fresh.returncode}):\n{fresh.output}"]
    failures: list[str] = []
    if fresh.derived_hash != lock["hashes"]["derived"]:
        failures.append(f"derived hash {fresh.derived_hash} after the load, {lock['hashes']['derived']} saved")
    failures += history_differences(reference, fresh.history)
    return failures


def refused_round_trip_failures(experiment: Experiment, folder: Path, codes: tuple[str, ...]) -> list[str]:
    """For an experiment whose build is refused: its build refuses with ``codes``, and its document, written and
    read back in a fresh process, describes an experiment refused for the same reasons."""
    try:
        experiment.build()
    except BuildRefused as refused:
        found = tuple(refused.codes())
    else:
        return ["the build was not refused"]
    if found != codes:
        return [f"the build is refused with {list(found)}, not {list(codes)}"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DOCUMENT_FILE).write_text(dump_document(experiment.document()), encoding="utf-8")
    fresh = load_in_fresh_process(folder, document_only=True)
    expected = f"refused: {json.dumps(list(codes))}"
    return [] if expected in fresh.output else [f"the fresh process did not refuse with {list(codes)}:\n"
                                                f"{fresh.output}"]


@contextmanager
def _compilations() -> Generator[list[str]]:
    """Every XLA compilation while the block runs, as JAX's own monitoring reports them."""
    seen: list[str] = []

    def listen(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            seen.append(event)

    monitoring.register_event_duration_secs_listener(listen)
    try:
        yield seen
    finally:
        monitoring.unregister_event_duration_listener(listen)


def _equal(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> bool:
    return all(np.array_equal(np.asarray(a[n][f]), np.asarray(b[n][f])) for n in a for f in a[n])


def param_reach_failures(built: BuiltExperiment, steps: int = 5) -> list[str]:
    """Every value a study may change is a ``params`` leaf the compiled step reads. For each trainable leaf of the
    built graph (an initial condition, ``initial_*``, reaches the run through the state instead), write a new value
    and run ``steps`` steps from the initial state, every external input fed its controller's held write: the run's
    output must change, and nothing may be compiled again. Each leaf that changes nothing, and any recompilation, is
    a failure. Leaves the graph reset."""
    graph = built.graph
    leaves = parameter_leaves(graph)
    specs = parameter_specs(graph)
    checked = [(owner, key) for owner, held in sorted(leaves.items()) for key in sorted(held)
               if specs.get(owner, {}).get(key, ParamSpec()).trainable]
    fed = built.held_inputs()
    graph.reset_state()
    reference = run_scan(graph, steps, fed=fed)
    changed = {(owner, key): device_array(np.asarray(leaves[owner][key]) * 1.5 + 0.25) for owner, key in checked}
    graph.reset_state()
    traces = graph.scan_trace_count
    failures: list[str] = []
    with _compilations() as compiled:
        for (owner, key), value in changed.items():
            graph.reset_state()
            found = run_scan(graph, steps, {"nodes": {owner: {key: value}}}, fed=fed)
            if _equal(reference, found):
                failures.append(f"writing params['nodes'][{owner!r}][{key!r}] changed nothing over {steps} steps: "
                                f"the compiled step does not read it, so a study changing it changes nothing (a "
                                f"value baked into the trace as a constant)")
    if compiled or graph.scan_trace_count != traces:
        failures.append(f"writing {len(checked)} parameter leaves compiled {len(compiled)} program(s) and traced "
                        f"{graph.scan_trace_count - traces} scan(s): a leaf is an argument of the compiled step, so "
                        f"a new value must not recompile it")
    graph.reset_state()
    return failures
