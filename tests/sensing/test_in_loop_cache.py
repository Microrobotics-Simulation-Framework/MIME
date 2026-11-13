"""The static structure, a leaf: read as the tree holds it, never made in a frame step, and never read at other numbers
than the run's.

The node holds the structure (``static_structure``) beside every number it was made at (``static_structure_from``: the
nine intrinsics, the scene's numbers, the camera's pose). A frame step reads it. It is made when the node's parameters are
made (the build, an override's point, a study's point, the save's rerun); once a forward run, on the host, before its
steps, where the tree's was made at other numbers than its leaves (``held_made_again``); and, where a derivative may be
taken, once a run from the run's leaves, outside its steps (``run_constants``). Five claims:

* a run continued with another leaf draws the whole frame at it (the review of #266 found the tube drawn at the old
  ``fy`` and the ball at the new one), whether the leaf comes by an override (its tree is a point's, made at its own
  leaves), written into the graph's tree leaf by leaf, or in a tree given to the run with the leaf written leaf by leaf;
* the structure a tree holds is read, on both routes (a run whose chunks make the run-constants, and MADDENING's own scan
  steps, a recorded run): it is *served* (a camera that renders the static structure in every frame is correct and
  dearer, and no image-level test of a served value can tell it), and made again once a run where any of its numbers
  differs;
* a step that meets a structure made at other numbers, outside MIME's runs, draws a frame that is not a number;
* no frame step makes it: the step's program, as a chunk runs it, as MADDENING's scan runs it and inside a differentiated
  run's loop, reaches none of the functions that make it (a camera on a moving mount, which renders it in every frame,
  is the test's own control that the inspection sees them), and a differentiated run makes it once, outside its loop.
  The served case cannot see a make in a branch the step does not take (#284's review: a remake in every frame off the
  build point); this one can;
* the state carries nothing of it, so a held step holds the image alone.

**The controls** (``CAMERA_PLANT``): ``read-as-is`` turns the run's remake and the step's guard off, so a forward run reads
the tree's structure as it is; ``remake-off`` the remake alone; ``guard-off`` the guard alone; ``cache-every-frame`` makes
the structure in every frame and reads nothing held; ``made-branch`` puts the make back in the frame step as #284 first
had it (a branch taken where the numbers differ). Each fails the cases of its claim on every run; an override's tree is
built at its own leaves, so ``read-as-is`` does not reach the override cases, only the leaf-by-leaf ones."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted
from test_in_loop_mounts import moving_gantry

from mime.build import RunFailed, _maddening
from mime.docs._switches import equation_frames, equations, step_function
from mime.observations import Mounted
from mime.sensing._node import STRUCTURE, STRUCTURE_FROM, CameraNode
from mime.units import ms

lax: Any = jax.lax

if planted("read-as-is") or planted("remake-off"):
    CameraNode.held_made_again = lambda self, params: {}  # type: ignore[method-assign]

if planted("read-as-is") or planted("guard-off"):
    # still reading the numbers (a step that read none would have its leaves' writes refused, a failure of another kind)
    CameraNode.holds_at = lambda self, p, numbers: jnp.all(numbers == p[STRUCTURE_FROM]) | True  # type: ignore[method-assign]

if planted("cache-every-frame"):
    CameraNode.structure_of = lambda self, p, numbers: self._make(numbers)  # type: ignore[method-assign]  # noqa: SLF001

if planted("made-branch"):
    def _made_branch(self: CameraNode, p: Any, numbers: Any) -> Any:
        return lax.cond(jnp.all(numbers == p[STRUCTURE_FROM]), lambda: jnp.asarray(p[STRUCTURE]),
                        lambda: self._make(numbers).astype(jnp.asarray(p[STRUCTURE]).dtype))  # noqa: SLF001

    CameraNode.structure_of = _made_branch  # type: ignore[method-assign]

MARKER = 0.25
CAMERA = "camera:"


def case(fy: float = 750.0, hide: tuple[str, ...] = (), period=ms(3)):
    return ballistic(period, hide=hide, absorbers=(("tube", 3e5),), detector=dataclasses.replace(DETECTOR, fy=fy))


def expected(built) -> np.ndarray:
    return np.asarray(built.current_state()[CAMERA]["expected"], dtype=np.float64).copy()


@contextmanager
def camera_leaves(built, **leaves: Any) -> Iterator[None]:
    """For the block, the camera's ``leaves`` written into the graph's parameter tree, which its compiled programs take as
    an argument (nothing is traced again); the tree put back after."""
    held = _maddening._held(built.graph)  # noqa: SLF001
    tree = held.params
    nodes = dict(tree["nodes"])
    nodes[CAMERA] = {**nodes[CAMERA], **leaves}
    held.params = {**tree, "nodes": nodes}
    try:
        yield
    finally:
        held.params = tree


#: How a continued run is given another fy: the override (its tree is a point's, built at its own leaves), fy written into
#: the graph's tree leaf by leaf, and a tree given to the run with fy written leaf by leaf (as a study's member's is given).
LEAF_ROUTES = ("override", "written", "given")


@pytest.mark.parametrize("route", LEAF_ROUTES)
@pytest.mark.parametrize("hide", [("ball",), ()], ids=["tube-only", "ball-and-tube"])
def test_a_continued_run_with_another_leaf_draws_the_frame_at_it(dtype, hide, route):
    built = case(hide=hide).build()
    built.run(ms(6))
    before = expected(built)
    if route == "override":
        built.run(ms(6), override={"camera:.detector.fy": 800.0})
    else:
        tree = _maddening.graph_params(built.graph)
        fy = jnp.asarray(800.0, dtype=jnp.asarray(tree["nodes"][CAMERA]["fy"]).dtype)
        if route == "written":
            with camera_leaves(built, fy=fy):
                built.run(ms(6))
        else:
            given = {**tree, "nodes": {**tree["nodes"], CAMERA: {**tree["nodes"][CAMERA], "fy": fy}}}
            built._runs.run(6, history=False, params=given)  # noqa: SLF001 - a tree given to a forward run
    fresh = case(800.0, hide).build()
    fresh.run(ms(12))
    assert np.abs(expected(built) - before).max() > 100.0  # the leaf moved the image (the tube's edge), so this can tell
    # exactly equal: the tree's structure is made at its leaves by the same compiled maker as the fresh build's (the
    # override's point's at its build, a tree written leaf by leaf's once before the run), so any difference is a stale
    # part, not rounding (measured 0.0 in both dtypes, tube only and with the ball)
    assert np.abs(expected(built) - expected(fresh)).max() == 0.0


ROUTES = {"run-constants": {}, "history-scan": {"record": ["camera:.frame_step"]}}


@pytest.mark.parametrize("route", list(ROUTES))
def test_the_structure_is_served_while_its_numbers_hold_and_made_again_once_a_run_when_any_differs(dtype, route):
    built = case(period=ms(1), hide=("ball",)).build()
    run = ROUTES[route]
    built.run(ms(2), **run)
    clean = expected(built)
    own = _maddening.graph_params(built.graph)["nodes"][CAMERA]
    structure, numbers = jnp.asarray(own[STRUCTURE]), jnp.asarray(own[STRUCTURE_FROM])
    rows, columns = clean.shape
    planes = 3 + 10 * rows * columns  # the transmission's plane follows the origin and the ray table's ten
    marked = structure.at[planes:planes + rows * columns].set(MARKER)

    with camera_leaves(built, **{STRUCTURE: marked}):
        built.run(ms(1), **run)
    assert np.allclose(expected(built), 1000.0 * MARKER), "a structure whose numbers hold is read, not rendered again"

    for index in range(numbers.shape[0]):
        with camera_leaves(built, **{STRUCTURE: marked, STRUCTURE_FROM: numbers.at[index].add(1.0)}):
            built.run(ms(1), **run)
        assert np.abs(expected(built) - clean).max() == 0.0, f"number {index} of {numbers.shape[0]} is not compared"


def test_a_step_given_a_structure_made_at_other_numbers_draws_no_frame(dtype):
    """The guard behind the forward run's remake: MADDENING's own scan called on the graph (no run of MIME's, so nothing
    makes the structure again first) with a tree whose structure was made at other numbers draws a frame that is not a
    number, never the frame at other leaves."""
    built = case(period=ms(1), hide=("ball",)).build()
    numbers = jnp.asarray(_maddening.graph_params(built.graph)["nodes"][CAMERA][STRUCTURE_FROM])
    with camera_leaves(built, **{STRUCTURE_FROM: numbers.at[1].add(1.0)}):
        _maddening.run_scan(built.graph, 1)
    assert np.all(np.isnan(np.asarray(_maddening.node_state(built.graph, CAMERA)["expected"])))


#: The functions that make the static structure: the ray table, the standing subjects' image (``static_part``, defined as
#: ``_static``) and the maker around them.
MAKERS = {"ray_table", "_static", "structure_maker.<locals>.made"}
def reached(jaxpr: Any) -> set[str]:
    """The functions of :data:`MAKERS` that ``jaxpr``'s equations were traced in, nested ones included."""
    return {frame.function for _, equation in equations(jaxpr) for frame in equation_frames(equation)
            if frame.function in MAKERS}


def calls(jaxpr: Any, name: str) -> list[Any]:
    """The compiled calls (``jit``) of the function ``name`` in ``jaxpr``, nested ones included."""
    return [equation for _, equation in equations(jaxpr)
            if equation.primitive.name in ("pjit", "jit") and str(equation.params.get("name", "")) == name]


def camera_steps(jaxpr: Any) -> list[Any]:
    """The camera's compiled steps a run's program calls (``jit(CameraNode.advance)``), as their own programs."""
    found = []
    for equation in calls(jaxpr, "advance"):
        inner = equation.params["jaxpr"]
        if any(frame.function == "CameraNode.advance" for _, each in equations(inner) for frame in equation_frames(each)):
            found.append(inner)
    return found


def camera_programs(built: Any) -> dict[str, Any]:
    """The camera's frame step as each forward route traces it, the tree its arguments: MADDENING's own step (a recorded
    run's scan, ``graph.step``) and the step as a chunk runs it, its run-constants made first."""
    graph = built.graph
    node = graph.get_node(CAMERA)
    state, params = _maddening.full_state(graph), _maddening.graph_params(graph)
    given = {**_maddening.zero_inputs(node), **_maddening.resolved_inputs(graph, CAMERA)}
    step, dt = step_function(node), float(node.delta_t)
    fed = _maddening.fed_inputs(graph, built.held_inputs())
    return {"own step": jax.make_jaxpr(lambda s, g, p: step(s, g, dt, params=p))(state[CAMERA], given,
                                                                                    params["nodes"][CAMERA]),
            "chunk step": jax.make_jaxpr(_maddening.chunk_step(graph))(state, fed, params)}


def test_no_frame_step_makes_the_structure_and_a_differentiated_run_makes_it_once_outside_its_loop(dtype):
    built = case(period=ms(1), hide=("ball",)).build()
    for route, program in camera_programs(built).items():
        assert reached(program) == set(), f"{route}: a frame step makes the structure ({reached(program)})"

    graph = built.graph
    state, params = _maddening.full_state(graph), _maddening.graph_params(graph)
    run = _maddening.differentiated_run(graph, 3, (), (), checkpoint=False)
    theta = jnp.asarray(params["nodes"][CAMERA]["fy"])

    def wall(fy: Any) -> Any:
        camera = {**params["nodes"][CAMERA], "fy": fy}
        final, _, _ = run(state, {**params, "nodes": {**params["nodes"], CAMERA: camera}})
        return jnp.sum(final[CAMERA]["expected"])

    for mode, program in (("forward", jax.make_jaxpr(wall)(theta)),
                          ("jvp", jax.make_jaxpr(lambda t: jax.jvp(wall, (t,), (jnp.ones_like(t),)))(theta))):
        steps = camera_steps(program)
        assert steps, f"{mode}: the inspection finds no step of the camera in the run"
        in_steps = set().union(*(reached(step) for step in steps))
        assert in_steps == set(), f"{mode}: a step of the differentiated run makes the structure ({in_steps})"
        made = len(calls(program, "made")) + len(calls(program, "jvp(made)"))
        assert reached(program) == MAKERS and made == 1, f"{mode}: the run makes the structure {made} times, not once"

    moving = ballistic(ms(1), mount=Mounted(frame="stage.gantry.carriage", orientation=LOOK_ACROSS))
    moving_gantry(moving)
    control = camera_programs(moving.build())
    assert all({"ray_table", "_static"} <= reached(program) for program in control.values()), \
        "the inspection does not see a frame step render the structure (a camera on a moving mount does in every frame)"


def test_the_state_carries_nothing_of_the_structure(dtype):
    """The held step's carry: the camera's state is its clock, the image, the ring and the flags, nothing image-sized
    beyond the two frames, so a held step does not move the structure."""
    built = case().build()
    state = built.current_state()[CAMERA]
    rows, columns = np.shape(state["image"])
    sized = sorted(name for name, value in state.items() if np.size(value) >= rows * columns)
    assert sized == ["expected", "image"], sized
    assert STRUCTURE in _maddening.graph_params(built.graph)["nodes"][CAMERA]
