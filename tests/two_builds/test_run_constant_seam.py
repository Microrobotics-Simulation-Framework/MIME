"""The run-constant seam (``mime.core.MimeNode.run_constants``, made once per chunk by MIME's chunked runs) on the MRS
drag, whose held body-frame matrix is its run-constant (MIME-ANO-147), and the hull's search values held as leaves
(MIME-ANO-145).

- **Reverse mode.** FL-9 in free space at 222 force points (the bake-check spike's case), float64: the gradient of a
  loss of the final pose in the shape's radius moved 1 % (so the matrix is made from it), through the chunked run (the
  matrix made once, before the loop: ``chunk_function``), equals the one through MADDENING's scan of the step (whose
  own ``held_or_made`` makes the matrix again at every step of the backward pass) to 1e-12 of its size, at 20 steps;
  in the chunk's compiled program, value and gradient, the 666-wide LU sits outside the scan's loops only. Control: in
  the per-step form's compiled program it sits inside the loop, every run.
- **Forward, bit for bit.** MIME's chunked scan with flags (the seam), MADDENING's scan of the step (its own
  ``held_or_made``, against its ``held_from`` leaves) and that step against the build's values as constants (the
  program before the change: ``_held_from`` patched to the build's values) give the same arrays, in free space and with
  the tube's wall, float64 and float32.
- **The hull.** A hull contact's step given its search values as leaves gives the arrays of the step reading them as the
  build's Python numbers (the program before the change: the step's constants without the leaves), bit for bit, on the
  contact 0.5.0 and the 0.4.0 kept registered, float64 and float32.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hull_cases import hull_contact
from mrs_cases import BALL, free_space, mrs, mrs_tube
from script_a import fl9

from mime.build import plan_experiment
from mime.build._construct import assemble
from mime.build._maddening import (
    FlagTest,
    chunk_function,
    fed_inputs,
    full_state,
    graph_params,
    node_state,
    parameter_leaves,
    resolved_inputs,
    run_scan,
    scan_chunk_tallied,
    step_function,
)
from mime.control import law_and_models
from mime.experiment import Experiment
from mime.interactions import KEPT
from mime.nodes.body._rigid_inertial import SEARCH_LEAVES, RigidInertialNode
from mime.nodes.hydro.mrs import _node as mrs_node
from mime.saved._pins import keeping_pins
from mime.units import kg, kg_m2, m_per_s, mm, rad_per_s

NODE, BODY = "drag:screw", "screw"
#: The spike's spacing: 222 force points on FL-9, a 666-wide dense solve.
SPACING_MM = 0.829
STEPS = 20
WEIGHTS = jnp.asarray([0.3, -0.5, 0.7])
PRECISIONS = ("float64", "float32")


@pytest.fixture(autouse=True)
def precision_of(request: pytest.FixtureRequest) -> Iterator[None]:
    """A test parametrised by ``precision`` runs in it; any other in float64."""
    spec = getattr(request.node, "callspec", None)
    precision = spec.params.get("precision", "float64") if spec is not None else "float64"
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", precision == "float64")
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module")
def screw() -> Any:
    built = free_space("seam_fl9", {BODY: fl9()}, mrs(mm(SPACING_MM)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                       spin=rad_per_s((3.0, -1.0, 2.0))).build()
    graph = built.graph
    graph.reset_state()
    held = np.asarray(parameter_leaves(graph)[NODE][mrs_node.HELD])
    assert held.shape == (6, 6), "free space holds the resistance"
    assert graph.get_node(NODE).sample_count == 222, "the spike's 222 force points"
    return built


def _with_radius(params: Any, radius: Any) -> Any:
    nodes = dict(params["nodes"])
    held = dict(nodes[NODE])
    held["shape_radius"] = jnp.asarray(radius, dtype=jnp.asarray(held["shape_radius"]).dtype)
    nodes[NODE] = held
    return {**params, "nodes": nodes}


def _loss(final: Any) -> Any:
    pose = final[BODY]["pose"]
    return 1e3 * jnp.dot(pose[:3], WEIGHTS) + jnp.sum(pose[3:])


def seam_loss(built: Any, steps: int) -> Any:
    graph = built.graph
    run = chunk_function(graph, steps, built.held_inputs())
    state, params = full_state(graph), graph_params(graph)
    return lambda radius: _loss(run(state, _with_radius(params, radius)))


def per_step_loss(built: Any, steps: int) -> Any:
    graph = built.graph
    step = step_function(graph)
    state, params = full_state(graph), graph_params(graph)
    external = fed_inputs(graph, built.held_inputs())

    def loss(radius: Any) -> Any:
        moved = _with_radius(params, radius)

        def body(carry: Any, _: Any) -> tuple[Any, None]:
            return step(carry, external, moved), None

        final, _ = jax.lax.scan(body, state, None, length=steps)
        return _loss(final)

    return loss


_CALL = re.compile(r"(?:body|condition|to_apply|calls|true_computation|false_computation)=%?([\w.\-]+)")
_BRANCHES = re.compile(r"branch_computations=\{([^}]*)\}")
_HEAD = re.compile(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$")
_FACTOR = re.compile(r'= \(?f(?:64|32)\[(\d+),(\d+)\][^=]*custom_call_target="([^"]*getrf[^"]*)"', re.I)


def lu_places(text: str, steps: int, width: int) -> tuple[int, int]:
    """How many ``width``-wide LU factorisations an optimised HLO module computes inside a loop of ``steps`` trips
    (reachable from the body of a while whose known trip count is ``steps``), and how many outside every such loop."""
    comps: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        head = _HEAD.match(line)
        if head and not line.startswith(" "):
            current = head.group(2)
            comps[current] = []
            continue
        if line.startswith("}"):
            current = None
        elif current is not None:
            comps[current].append(line)
    calls = {name: set[str]() for name in comps}
    bodies: list[str] = []
    for name, lines in comps.items():
        for line in lines:
            calls[name].update(found for found in _CALL.findall(line) if found in comps)
            for group in _BRANCHES.findall(line):
                calls[name].update(each.strip().lstrip("%") for each in group.split(",")
                                   if each.strip().lstrip("%") in comps)
            if " while(" in line and f'"n":"{steps}"' in line:
                body = re.search(r"body=%?([\w.\-]+)", line)
                if body:
                    bodies.append(body.group(1))
    inside: set[str] = set()
    todo = list(bodies)
    while todo:
        name = todo.pop()
        if name not in inside:
            inside.add(name)
            todo.extend(calls.get(name, ()))
    assert bodies, "the program holds the scan's loop"
    counts = [0, 0]
    for name, lines in comps.items():
        for line in lines:
            for rows, columns, _ in _FACTOR.findall(line):
                if int(rows) == width and int(columns) == width:
                    counts[0 if name in inside else 1] += 1
    return counts[0], counts[1]


@pytest.fixture(scope="module")
def gradients(screw: Any) -> dict[str, Any]:
    radius = float(np.asarray(parameter_leaves(screw.graph)[NODE]["shape_radius"])) * 1.01
    found: dict[str, Any] = {}
    for name, make in (("seam", seam_loss), ("per step", per_step_loss)):
        program = jax.jit(jax.value_and_grad(make(screw, STEPS)))
        compiled = program.lower(radius).compile()
        found[name] = (compiled(radius), compiled.as_text())
    return found


def test_reverse_mode_through_the_chunk_equals_the_per_step_form(gradients: dict[str, Any]):
    (value, gradient), _ = gradients["seam"]
    (reference, slope), _ = gradients["per step"]
    assert abs(float(value) - float(reference)) <= 1e-12 * abs(float(reference))
    assert abs(float(gradient) - float(slope)) <= 1e-12 * abs(float(slope))
    assert float(slope) != 0.0, "the radius moves the loss, so equal gradients are evidence"


def test_the_chunks_lu_sits_outside_its_loops(gradients: dict[str, Any]):
    _, text = gradients["seam"]
    inside, outside = lu_places(text, STEPS, 666)
    assert inside == 0
    assert outside >= 1, "the matrix is made, once"


def test_control_the_per_step_forms_lu_sits_inside_its_backward_loop(gradients: dict[str, Any]):
    _, text = gradients["per step"]
    inside, _ = lu_places(text, STEPS, 666)
    assert inside >= 1


def free_ball() -> Experiment:
    return free_space("free_ball", {"ball": BALL}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


FORWARD = {"free_ball": free_ball, "mrs_tube": mrs_tube}


def _arrays(state: Any) -> dict[str, np.ndarray]:
    return {f"{node}.{name}": np.asarray(value) for node, fields in state.items() if node != "_meta"
            for name, value in fields.items()}


def forward_runs(case: str) -> list[dict[str, np.ndarray]]:
    built = FORWARD[case]().build()
    graph = built.graph
    drag = next(n for n in graph.node_names if isinstance(graph.get_node(n), mrs_node.MRSDragNode))
    graph.reset_state()
    seam, _ = scan_chunk_tallied(graph, 8, fed_inputs(graph, built.held_inputs()), [FlagTest(drag, "resistance")])
    graph.reset_state()
    own = run_scan(graph, 8, fed=built.held_inputs())
    with pytest.MonkeyPatch.context() as patch:
        def as_built(self: mrs_node.MRSDragNode, p: Any) -> dict[str, Any]:
            return {key: jnp.asarray(value, dtype=jnp.asarray(p[key]).dtype)
                    for key, value in self._built_theta().items()}  # noqa: SLF001

        patch.setattr(mrs_node.MRSDragNode, "_held_from", as_built)
        before = FORWARD[case]().build()
        before.graph.reset_state()
        literal = run_scan(before.graph, 8, fed=before.held_inputs())
    return [_arrays(seam), _arrays(own), _arrays(literal)]


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("case", list(FORWARD))
def test_forward_the_seam_the_steps_own_comparison_and_the_literal_one_are_bit_for_bit(case: str, precision: str):
    seam, own, literal = forward_runs(case)
    assert seam.keys() == own.keys() == literal.keys()
    for key in seam:
        assert np.array_equal(seam[key], own[key]), key
        assert np.array_equal(own[key], literal[key]), key


def assembled_hull(experiment: Experiment) -> tuple[Any, str]:
    settings, declarations, saved = experiment.declared_as()
    with keeping_pins(saved):
        plans = plan_experiment(settings, list(declarations))
        law, _, _ = law_and_models(plans)
        graph = assemble(plans, law).graph
    graph.compile()
    graph.reset_state()
    name = next(n for n in graph.node_names
                if isinstance(node := graph.get_node(n), RigidInertialNode) and node.holds_hull)
    return graph, name


HULLS = {"hull_contact_looped": lambda: hull_contact("hull_contact_looped"),
         "hull_contact_kept": lambda: hull_contact("hull_contact_kept", contact=KEPT["contact", "0.4.0"][0])}


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("case", list(HULLS))
def test_the_hulls_search_values_as_leaves_are_the_literal_steps_bit_for_bit(case: str, precision: str):
    graph, name = assembled_hull(HULLS[case]())
    node = graph.get_node(name)
    leaves = parameter_leaves(graph)[name]
    assert set(SEARCH_LEAVES) <= set(leaves)
    with_leaves = node._step_constants(leaves)  # noqa: SLF001
    literal = {key: value for key, value in with_leaves.items() if key not in SEARCH_LEAVES}
    inputs = resolved_inputs(graph, name)
    held = dict(node_state(graph, name))
    runs = []
    for constants in (with_leaves, literal):
        step = jax.jit(node.advance, static_argnums=3)
        state, found = held, []
        for _ in range(4):
            state = step(constants, state, inputs, node.delta_t)
            found.append({key: np.asarray(value) for key, value in state.items()})
        runs.append(found)
    assert any(np.any(np.asarray(step["contact_forces"]) != 0) for step in runs[0]), "the hull is held"
    for a, b in zip(*runs, strict=True):
        for key in a:
            assert np.array_equal(a[key], b[key]), key


def test_a_run_constant_that_is_not_a_leaf_of_its_own_shape_is_refused_as_mimes_defect():
    """``run.run-constant``: a node whose run-constant has another shape than the leaf it replaces, or replaces none."""
    from unit_fields import FIELD, UnitFieldSpringNode, spring_experiment

    from mime.build._refusals import InternalError

    built = spring_experiment()[0].build()
    graph = built.graph
    for wrong in ({FIELD: jnp.zeros(3)}, {"no_such_leaf": jnp.zeros(())}):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(UnitFieldSpringNode, "run_constants", lambda self, params, wrong=wrong: wrong)
            with pytest.raises(InternalError) as raised:
                chunk_function(graph, 2, built.held_inputs())(full_state(graph), graph_params(graph))
        assert raised.value.codes() == ["run.run-constant"]
