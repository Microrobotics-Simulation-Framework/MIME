"""The two-builds check's false-positive arm: the registry walk's cases (``tests/saved/cases.py``), nothing planted.
Each case's document is written (beside its saved folder where it reads one: an asset, a record, a description), and
each float of it (settings, declared values, parts' settings, initial state, gravity; not the run's step, a quaternion,
an integer or a zero) is moved by a relative 1e-6 in turn; the base and the moved document are each assembled as a
study's point is (planned, the controller's law made, assembled, compiled by MADDENING) and the trace form compares
their steps as MIME's chunked run executes them (``mime.testing.graph_program``, whose constants include those of every
nested sub-program: an array a node's step holds as built is seen, not only a Python float). A move the build refuses
(a rate no longer a multiple of the step) is skipped. On the tip before S-F0-1 the arm fired 21 times on the MRS drag's held
matrix (``held_or_made`` compared the leaves with the build's values as constants) and 5 times on the hull's shape
(Python floats made from it): the matrix's leaves and the hull's search values are arguments of the step now. What
fires is pinned below, pair by pair (:data:`HELD`), and every such pair is ``held``: a moved leaf of it is one the step
cannot read (MADDENING's record of the step's reads), a value its node consumes when it is built and declares so, which
MADDENING refuses to write into a built graph and a study refuses to vary. Any other fire fails.

The control, on the cases those fires came from: the same pairs traced with the parameters closed over (the step's
default, MADDENING's compile-time snapshot), which makes every leaf a constant of the program, fire on every pair
whose leaves move. So a quiet pair is the leaves being arguments, not a form that cannot see them. On R1's merge the
arm fired on `rft_tube`'s helix turns: the rft rung's measured family and gap, decided at build from leaves the step
reads, which the step now judges from its leaves; its control plants them back as the build made them, and the same
pair fires.

**"The walk is quiet" bounds what it walked**: the trace form over the floats of these cases' documents, nonzero ones,
each moved by 1e-6 in turn. It does not reach a node no case builds, a setting that is a direction, a count or a choice,
a zero (skipped), the run form (not run over the walk: it compiles each build's chunk), a value that reaches the run
only through a host callback or a derivative rule's lazily traced function, or float32 (the check refuses it)."""

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
from mrs_cases import mrs_solids
from robot_asset import bound_robot_with_parts

from mime.build import ArgumentRefused, BuildRefused, plan_experiment
from mime.build._construct import assemble
from mime.build._maddening import chunk_step, full_state, graph_params, parameter_leaves, unreadable_leaves
from mime.control import law_and_models
from mime.experiment import Experiment
from mime.nodes.hydro import RFTDragNode
from mime.saved import dump_document
from mime.saved._pins import keeping_pins
from mime.study import Study, grid
from mime.testing import graph_program
from mime.testing._two_builds import nested_program, programs_differ
from mime.units import mm, mPa_s, ms, mT

RELATIVE = 1e-6
#: The pairs that fire, each held: a vector setting of a part's kind (an ellipsoid's semi-axes, a box's size), which
#: the node made from the part holds as built (the rigid body's hull, ``hull_shape_semi_axes``; the MRS drag's samples,
#: ``shape_size`` and ``shape_semi_axes``; each declared so and unread by the step), and which no run varies: an
#: override and a sweep refuse it (below), so a new value is a rebuild. A declared geometry's vectors become leaves its
#: step reads with declared ranges (lane S's next phase).
HELD: dict[str, set[str]] = {
    "bound_robot_with_parts": {"things/UMR/parts/Shell/solid/settings/semi_axes/value"},
    "mrs_solids": {"things/brick/parts/shell/solid/settings/size/value",
                   "things/egg/parts/shell/solid/settings/semi_axes/value"},
    # The grid's spacing, which the point body's drag consumes at build (its exchange kernel's tables and the
    # self-field it removes); a field's setting is written where it is, never overridden, so a new box is a rebuild.
    "point_body_flow": {"things/water/fields/flow/settings/domain/settings/length/value"},
}
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
    except Exception:  # noqa: BLE001 - a document that needs its folder (an asset, a record): save it, write beside it
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
    return nested_program(jax.make_jaxpr(lambda s: step(s, fed, params))(state))


def walk(case: Any, folder: Path, control: bool = False) -> list[dict[str, Any]]:
    document, load = loader(case, folder)
    base = load(document, "base")
    program = closed_program(base) if control else graph_program(base)
    leaves = {o: {k: np.asarray(v) for k, v in f.items()} for o, f in parameter_leaves(base).items()}
    unread = set(unreadable_leaves(base))
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
        rows.append({"path": path, "how": programs_differ(program, found), "moved": moved_leaves,
                     "unread": [each for each in moved_leaves if each in unread]})
    return rows


@pytest.mark.parametrize("case", WALKED, ids=[case.name for case in WALKED])
def test_no_pair_of_the_registry_walks_cases_fires_the_trace_form(case: Any, tmp_path: Path):
    if case.lab and importlib.util.find_spec("mime_lab_fixture") is None:
        pytest.skip("the lab fixture is not installed (every CI lane installs it)")
    rows = walk(case, tmp_path)
    fired = {"/".join(map(str, row["path"])) for row in rows if row.get("how")}
    assert fired == HELD.get(case.name, set())
    assert all(row["unread"] for row in rows if row.get("how")), "every pair that fires is held as built"
    assert any(row.get("moved") for row in rows) or not rows, "the walk moved a leaf, so a quiet arm is evidence"


#: Each held pair as a study or an override names it: its case, the value's path, its base and a moved value, the leaf
#: the step cannot read, and a control's value the step reads (an axis, its base and a moved value).
REBUILDS = {
    "bound_robot_with_parts": (bound_robot_with_parts, "UMR.parts.Shell.semi_axes", mm((1.56, 1.56, 1.56)),
                               mm((1.5601, 1.56, 1.56)), ("UMR", "hull_shape_semi_axes"), "coils.commands.amplitude",
                               mT(1.2), mT(1.3)),
    "mrs_solids/brick": (mrs_solids, "brick.parts.shell.size", mm((0.6, 0.9, 1.3)), mm((0.6001, 0.9, 1.3)),
                         ("drag:brick", "shape_size"), "water.viscosity", mPa_s(1.0), mPa_s(1.1)),
    "mrs_solids/egg": (mrs_solids, "egg.parts.shell.semi_axes", mm((0.5, 0.7, 1.1)), mm((0.5001, 0.7, 1.1)),
                       ("drag:egg", "shape_semi_axes"), "water.viscosity", mPa_s(1.0), mPa_s(1.1)),
}


@pytest.mark.parametrize("name", list(REBUILDS))
def test_a_held_setting_takes_a_rebuild_an_override_is_refused_and_a_sweep_builds_each_value(name: str):
    """What makes the walk's held fires acceptable: no run varies a leaf the step cannot read. An override of the
    setting is refused (``override.structural``: a kind's setting that is not one number is built per value, so a run
    on the built step cannot take another), and a sweep over it builds each value, one group a value split by the
    axis, so no group's members differ in the leaf. The control: a value the step reads is overridden and planned as
    one group of two members, nothing refused."""
    make, path, base, moved, leaf, axis, own, other = REBUILDS[name]
    exp = make()
    built = exp.build()
    assert leaf in unreadable_leaves(built.graph)
    with pytest.raises(BuildRefused) as raised:
        built.run(ms(2.0), override={path: moved})
    assert raised.value.codes() == ["override.structural"]
    swept = Study(exp).sweep(grid({path: [base, moved]}), duration=ms(2.0)).plan()
    assert [group.members for group in swept.groups] == [[0], [1]] and not swept.refused, swept.text()
    assert swept.groups[1].reasons[0].startswith(f"its value of {path}, a kind's setting"), swept.groups[1].reasons

    built.run(ms(2.0), override={axis: other})
    control = Study(exp).sweep(grid({axis: [own, other]}), duration=ms(2.0)).plan()
    assert len(control.groups) == 1 and len(control.points) == 2 and not control.refused, control.text()


CONTROLS = ("mrs_free_space", "hull_contact_looped")


@pytest.mark.parametrize("name", CONTROLS)
def test_control_the_leaves_closed_over_fire_on_every_pair_whose_leaves_move(name: str, tmp_path: Path):
    case = next(case for case in CASES if case.name == name)
    rows = [row for row in walk(case, tmp_path, control=True) if row.get("moved")]
    assert rows, "pairs whose leaves move"
    assert all(row["how"] for row in rows), [row["path"] for row in rows if not row["how"]]


def test_control_the_rft_family_decided_at_build_fires_on_rft_tubes_helix_turns(monkeypatch: pytest.MonkeyPatch,
                                                                                 tmp_path: Path):
    """The rft rung's family and its measured gap, judged in the step from its leaves: planted back as R1 built them,
    decided at build and held as constants of the step, they fire on rft_tube's helix turns, whose move crosses the
    family's one measured slenderness (held to ``SHAPE_TOLERANCE``, a relative 1e-6 as the walk's move is); the
    planted values are the step's own at the built leaves, so nothing else differs. So the quiet rft cases are the
    family being judged in the step, not a walk that cannot see it."""
    judged = RFTDragNode.measured_here

    def at_build(self: RFTDragNode, p: Any) -> tuple[Any, Any]:
        with jax.enable_x64(True), jax.ensure_compile_time_eval():
            measured, gap = judged(self, self._built_leaves())  # noqa: SLF001 - the leaves as built
            held = bool(measured), float(gap)
        return jax.numpy.asarray(held[0]), jax.numpy.asarray(held[1], dtype=jax.numpy.asarray(p["viscosity"]).dtype)

    monkeypatch.setattr(RFTDragNode, "measured_here", at_build)
    case = next(case for case in CASES if case.name == "rft_tube")
    rows = walk(case, tmp_path)
    assert {"/".join(map(str, row["path"])) for row in rows if row.get("how")} == {
        "things/helix/parts/shell/solid/settings/turns"}
