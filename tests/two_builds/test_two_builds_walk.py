"""The two-builds check's false-positive arm: the registry walk's cases (``tests/saved/cases.py``), nothing planted.
Each case's document is written (beside its saved folder where it reads one: an asset, a record, a description), and
each float of it (settings, declared values, parts' settings, initial state, gravity; not the run's step, a quaternion,
an integer or a zero) is moved by a relative 1e-6 in turn; the base and the moved document are each assembled as a
study's point is (planned, the controller's law made, assembled, compiled by MADDENING) and the trace form compares
their steps as MIME's chunked run executes them (``mime.testing.graph_program``). A move the build refuses (a rate no
longer a multiple of the step) is skipped. On the tip before S-F0-1 the arm fired 21 times on the MRS drag's held
matrix (``held_or_made`` compared the leaves with the build's values as constants) and 5 times on the hull's search
values (Python floats made from its shape): both are arguments of the step now, and **no pair fires**.

The control, on the cases those fires came from: the same pairs traced with the parameters closed over (the step's
default, MADDENING's compile-time snapshot), which makes every leaf a constant of the program, fire on every pair
whose leaves move. So a quiet pair is the leaves being arguments, not a form that cannot see them."""

from __future__ import annotations

import copy
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from cases import CASES

from mime.build import BuildRefused, plan_experiment
from mime.build._construct import assemble
from mime.build._in_graph import program_of
from mime.build._maddening import chunk_step, full_state, graph_params, parameter_leaves
from mime.control import law_and_models
from mime.experiment import Experiment
from mime.saved import dump_document
from mime.saved._pins import keeping_pins
from mime.testing._two_builds import graph_program, programs_differ

RELATIVE = 1e-6
#: The keys whose numbers are not moved: a direction, a version, the seed.
SKIPPED = {"orientation", "up", "schema", "mime", "seed"}
WALKED = [case for case in CASES if not case.refused]


def numbers(document: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """The path to every float, or list of floats, of a document that the walk moves."""
    found: list[tuple[Any, ...]] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key not in SKIPPED and path[:1] != ("run",):
                found += numbers(value, (*path, key))
    elif isinstance(document, list):
        if document and all(isinstance(x, float) for x in document):
            found.append(path)
        else:
            for index, value in enumerate(document):
                found += numbers(value, (*path, index))
    elif isinstance(document, float):
        found.append(path)
    return found


def at(document: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        document = document[key]
    return document


def moved(document: Any, path: tuple[Any, ...]) -> Any:
    held = copy.deepcopy(document)
    parent = at(held, path[:-1])
    value = parent[path[-1]]
    parent[path[-1]] = [x * (1.0 + RELATIVE) for x in value] if isinstance(value, list) else value * (1.0 + RELATIVE)
    return held


def assembled(experiment: Experiment) -> Any:
    settings, declarations, saved = experiment.declared_as()
    with keeping_pins(saved):
        plans = plan_experiment(settings, list(declarations))
        law, _, _ = law_and_models(plans)
        graph = assemble(plans, law).graph
    graph.compile()
    return graph


def loader(case: Any, folder: Path) -> tuple[Any, Callable[[Any, str], Any]]:
    """The case's document and how to assemble a document of it: written alone where that assembles, else beside
    its saved folder, which holds what it reads."""
    experiment = case.make()
    document = copy.deepcopy(experiment.document())

    def written(where: Path) -> Callable[[Any, str], Any]:
        def load(held: Any, tag: str) -> Any:
            path = where / f"{tag}.mime.yaml"
            path.write_text(dump_document(held), encoding="utf-8", newline="\n")
            return assembled(Experiment.from_document(path))
        return load

    alone = written(folder)
    try:
        alone(document, "probe")
        return document, alone
    except BuildRefused:
        raise
    except Exception:  # noqa: BLE001 - a document that needs its folder: save it, and write beside it
        experiment.save(folder / "saved", steps=1)
        return document, written(folder / "saved")


def closed_program(graph: Any) -> tuple[str, list[np.ndarray]]:
    """The step traced with the parameters closed over: every leaf a constant of the program (the control)."""
    step = chunk_step(graph)
    graph.reset_state()
    state = jax.tree_util.tree_map(lambda leaf: leaf, full_state(graph))
    params = graph_params(graph)
    from mime.build._maddening import external_inputs, inputs

    fed: dict[str, dict[str, Any]] = {}
    for node, field in external_inputs(graph):
        spec = inputs(graph.get_node(node))[field]
        fed.setdefault(node, {})[field] = np.zeros(tuple(spec.shape), dtype=np.dtype(spec.dtype or np.float64))
    return program_of(jax.make_jaxpr(lambda s: step(s, fed, params))(state))


def walk(case: Any, folder: Path, control: bool = False) -> list[dict[str, Any]]:
    document, load = loader(case, folder)
    base = load(document, "base")
    program = closed_program(base) if control else graph_program(base)
    leaves = {o: {k: np.asarray(v) for k, v in f.items()} for o, f in parameter_leaves(base).items()}
    rows: list[dict[str, Any]] = []
    for path in numbers(document):
        value = at(document, path)
        if (isinstance(value, list) and not any(value)) or value == 0.0:
            continue
        try:
            other = load(moved(document, path), "moved")
        except BuildRefused:
            rows.append({"path": path, "refused": True})
            continue
        moved_leaves = [(o, k) for o, f in parameter_leaves(other).items() for k, v in f.items()
                        if k in leaves.get(o, {}) and np.asarray(v).shape == leaves[o][k].shape
                        and not np.array_equal(np.asarray(v), leaves[o][k])]
        found = closed_program(other) if control else graph_program(other)
        rows.append({"path": path, "how": programs_differ(program, found), "moved": moved_leaves})
    return rows


@pytest.mark.parametrize("case", WALKED, ids=[case.name for case in WALKED])
def test_no_pair_of_the_registry_walks_cases_fires_the_trace_form(case: Any, tmp_path: Path):
    if case.lab and importlib.util.find_spec("mime_lab_fixture") is None:
        pytest.skip("the lab fixture is not installed (every CI lane installs it)")
    rows = walk(case, tmp_path)
    fired = [(row["path"], row["how"]) for row in rows if row.get("how")]
    assert not fired
    assert any(row.get("moved") for row in rows) or not rows, "the walk moved a leaf, so a quiet arm is evidence"


CONTROLS = ("mrs_free_space", "hull_contact_looped")


@pytest.mark.parametrize("name", CONTROLS)
def test_control_the_leaves_closed_over_fire_on_every_pair_whose_leaves_move(name: str, tmp_path: Path):
    case = next(case for case in CASES if case.name == name)
    rows = [row for row in walk(case, tmp_path, control=True) if row.get("moved")]
    assert rows, "pairs whose leaves move"
    assert all(row["how"] for row in rows), [row["path"] for row in rows if not row["how"]]
