"""The static structure, a leaf: read as the tree holds it, never made in a frame step of MIME's runs, and never read at
other numbers than the run's.

The node holds the structure (``static_structure``) beside every number it was made at (``static_structure_from``: the
nine intrinsics, the scene's numbers, the camera's pose). A frame step reads it. It is made when the node's parameters are
made (the build, an override's point, a study's point, the save's rerun); once a forward run, on the host, before its
steps, where the tree's was made at other numbers than its leaves (``refreshed_held``); and, where a derivative may be
taken, once a run from the run's leaves, outside its steps (``run_constants``). Five claims:

* a run continued with another leaf draws the whole frame at it (the review of #266 found the tube drawn at the old
  ``fy`` and the ball at the new one), whether the leaf comes by an override, written into the graph's tree leaf by leaf,
  or in a tree given to the run with the leaf written leaf by leaf; the run's remake makes the structure again for the
  two leaf-by-leaf routes and nothing for the override, whose tree is a point's, built at its own leaves (asserted);
* the structure a tree holds is read, on both routes (a run whose chunks are given the run-constants, and MADDENING's own
  scan, which a recorded run uses): it is *served* (a camera that renders the static structure in every frame is correct
  and dearer, and no image-level test of a served value can tell it), and made again once a run where any of its numbers
  differs;
* a step not given its run-constants (MADDENING's own scan called on the graph), which takes whatever tree it is given,
  makes a structure made at other numbers again in its frame; a step given them that meets one draws a frame that is not
  a number;
* no step given its run-constants makes it: the step's program as such a chunk runs it, and inside a differentiated
  run's loop, reaches none of the functions that make it (a camera on a moving mount, which renders it in every frame, is
  the test's own control that the inspection sees them), and a differentiated run makes it once, outside its loop. The
  step not given them holds the choice per frame, by design (MADDENING's verification injects a moved leaf into it). The
  served case cannot see a make in a branch the step does not take (#284's review: a remake in every frame off the build
  point); this one can;
* the state carries nothing of it, so a held step holds the image alone.

**The controls** (``CAMERA_PLANT``): ``read-as-is`` turns the run's remake and the step's comparison off, so a forward run
reads the tree's structure as it is; ``remake-off`` the remake alone; ``guard-off`` the comparison alone (the guard of a
step given its run-constants and the per-frame choice of one not given them); ``cache-every-frame`` makes the structure
in every frame and reads nothing held; ``made-branch`` puts the make back in the frame step given its run-constants as
#284 first had it (a branch taken where the numbers differ). Each fails the cases of its claim on every run. No plant of
the run's reading reaches the override's route, whose tree holds no stale structure to read: that case holds the frame
at the leaf and the remake's premise, not a stale read."""

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
    CameraNode.refreshed_held = lambda self, params: {}  # type: ignore[method-assign]

if planted("read-as-is") or planted("guard-off"):
    # still reading the numbers (a step that read none would have its leaves' writes refused, a failure of another kind)
    CameraNode.holds_at = lambda self, p, numbers: jnp.all(numbers == p[STRUCTURE_FROM]) | True  # type: ignore[method-assign]

if planted("cache-every-frame"):
    # still taking the leaf as an input, as guard-off still reads the numbers
    CameraNode.structure_of = lambda self, p, numbers: self._make(numbers) + 0 * p[STRUCTURE]  # type: ignore[method-assign]  # noqa: SLF001

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


def marked(structure: Any, rows: int, columns: int) -> Any:
    """``structure`` with the standing subjects' transmission set to :data:`MARKER`, so a frame drawn over it reads
    ``1000 * MARKER`` everywhere: a frame read from it is told from one drawn over a structure made again."""
    planes = 3 + 10 * rows * columns  # the transmission's plane follows the origin and the ray table's ten
    return jnp.asarray(structure).at[planes:planes + rows * columns].set(MARKER)


@contextmanager
def remakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[bool]]:
    """For the block, whether each call of the camera's ``refreshed_held`` (once a forward run) made the structure
    again, in order."""
    made: list[bool] = []
    own = CameraNode.refreshed_held

    def spied(self: CameraNode, params: Any) -> dict[str, Any]:
        found = own(self, params)
        made.append(bool(found))
        return found

    with monkeypatch.context() as patch:
        patch.setattr(CameraNode, "refreshed_held", spied)
        yield made


#: How a continued run is given another fy: the override (its tree is a point's, built at its own leaves), fy written into
#: the graph's tree leaf by leaf, and a tree given to the run with fy written leaf by leaf (as a study's member's is given).
LEAF_ROUTES = ("override", "written", "given")


@pytest.mark.parametrize("route", LEAF_ROUTES)
@pytest.mark.parametrize("hide", [("ball",), ()], ids=["tube-only", "ball-and-tube"])
def test_a_continued_run_with_another_leaf_draws_the_frame_at_it(dtype, hide, route, monkeypatch):
    built = case(hide=hide).build()
    built.run(ms(6))
    before = expected(built)
    with remakes(monkeypatch) as made:
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
    # the override's tree is its point's, built at its own leaves, so its run makes nothing again and no plant of the
    # run's reading can reach it; a tree written leaf by leaf is made again once, before the run
    assert made == [route != "override"], made


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
    held = marked(structure, rows, columns)

    with camera_leaves(built, **{STRUCTURE: held}):
        built.run(ms(1), **run)
    assert np.allclose(expected(built), 1000.0 * MARKER), "a structure whose numbers hold is read, not rendered again"

    for index in range(numbers.shape[0]):
        with camera_leaves(built, **{STRUCTURE: held, STRUCTURE_FROM: numbers.at[index].add(1.0)}):
            built.run(ms(1), **run)
        assert np.abs(expected(built) - clean).max() == 0.0, f"number {index} of {numbers.shape[0]} is not compared"


def test_a_stale_structure_is_made_again_in_a_step_not_given_run_constants_and_is_no_frame_in_one_given_them(dtype):
    """Behind the forward run's remake. A tree whose structure is marked and whose numbers say it was made elsewhere:
    MADDENING's own scan called on the graph (a step not given its run-constants, which takes whatever tree it is given,
    as MADDENING's verification does) draws the frame over one made from the step's numbers, not the marked one; the step
    as a chunk given its run-constants runs it draws a frame that is not a number, never the marked one."""
    built = case(period=ms(1), hide=("ball",)).build()
    built.run(ms(1))
    clean = expected(built)  # nothing moves: every frame is the structure's
    graph = built.graph
    tree = _maddening.graph_params(graph)
    own = tree["nodes"][CAMERA]
    stale = {STRUCTURE: marked(own[STRUCTURE], *clean.shape),
             STRUCTURE_FROM: jnp.asarray(own[STRUCTURE_FROM]).at[1].add(1.0)}
    with camera_leaves(built, **stale):
        _maddening.run_scan(graph, 1)
    assert np.abs(np.asarray(_maddening.node_state(graph, CAMERA)["expected"]) - clean).max() == 0.0, \
        "a step not given its run-constants read a structure made at other numbers"
    given = {**tree, "nodes": {**tree["nodes"], CAMERA: {**own, **stale}}}
    fed = _maddening.fed_inputs(graph, built.held_inputs())
    final = jax.jit(_maddening.chunk_step(graph))(_maddening.full_state(graph), fed, given)
    assert np.all(np.isnan(np.asarray(final[CAMERA]["expected"]))), \
        "a step given its run-constants drew over a structure made at other numbers"


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
    """The camera's frame step traced with the tree as arguments: as a chunk given its run-constants runs it (made
    first) and as MADDENING's own scan runs it (not given them: a recorded run's, ``graph.step``)."""
    graph = built.graph
    node = graph.get_node(CAMERA)
    state, params = _maddening.full_state(graph), _maddening.graph_params(graph)
    given = {**_maddening.zero_inputs(node), **_maddening.resolved_inputs(graph, CAMERA)}
    step, dt = step_function(node), float(node.delta_t)
    fed = _maddening.fed_inputs(graph, built.held_inputs())
    return {"chunk step": jax.make_jaxpr(_maddening.chunk_step(graph))(state, fed, params),
            "own step": jax.make_jaxpr(lambda s, g, p: step(s, g, dt, params=p))(state[CAMERA], given,
                                                                                    params["nodes"][CAMERA])}


def test_no_step_given_run_constants_makes_the_structure_and_a_differentiated_run_makes_it_once_outside_its_loop(dtype):
    built = case(period=ms(1), hide=("ball",)).build()
    programs = camera_programs(built)
    assert reached(programs["chunk step"]) == set(), \
        f"a step given its run-constants makes the structure ({reached(programs['chunk step'])})"
    # a step not given them takes any tree, so the choice per frame lives there, by design
    assert reached(programs["own step"]) == MAKERS, "a step not given its run-constants cannot make a stale one again"

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
