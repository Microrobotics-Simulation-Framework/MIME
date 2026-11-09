"""The save's resolution rerun on an MRS drag: the nudged pass (the reference steps run again from a start one ulp off,
the parameters the step reads moved too) makes each drag's held matrix once, for the moved leaves, and gives it to the
step with the leaves it was made at (``MimeNode.made_again``; the ``held_from`` leaves), and the step keeps it there
rather than making it at each step.

Each arm is a fresh build, run four steps chunked by one and saved, in float64: its nodes are its own, so each step is
traced with the arm's patches. The resolution the save records, and its reference, are bit for bit those of the step
that makes the matrix itself at every nudged step: given a stale matrix with the build's leaves as the ones it was made
at, which no nudged leaf equals, the step ignores it and makes the matrix, as the step before the change did. A step
that kept the given matrix would record the stale arm's resolution, so equality is evidence that the step made it. The
control of the bits: a nudged pass given a matrix made from the un-nudged leaves, with the moved leaves as the ones it
was made at, records another resolution. The count is a callback in the drag's own matrix, so the counted builds are a
program of their own and are judged only by their counts; their control, the step that makes the matrix at every
step, counts one per nudged step, so the count sees the step's making.

The step compares its leaves with the ``held_from`` leaves beside the matrix, never with the build's values, so the
stale read the earlier key allowed (a step given every body leaf exactly as the rerun's, and the matrix as built, kept
the build's matrix) cannot be reached: given those leaves and the matrix with its own leaves as built, the step makes
the matrix at them.

The module runs the bits tests (and the unreachable read) in float64 and in float32; the counts are the same in
either, so they run in float64 alone."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from mrs_cases import BALL, free_space, mrs, mrs_tube

from mime.build import _construct, _maddening
from mime.core import jit, jnp, one_ulp_up_leaf
from mime.experiment import Experiment
from mime.nodes.hydro.mrs import _node as mrs_node
from mime.saved import LOCK_FILE
from mime.units import kg, kg_m2, m_per_s, mm, rad_per_s, s


def free_ball() -> Experiment:
    """A ball in free space, moving and turning from the start, its drag holding its resistance."""
    return free_space("free_ball", {"ball": BALL}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


PRECISIONS = ("float64", "float32")


@pytest.fixture(autouse=True)
def x64(request: pytest.FixtureRequest) -> Iterator[None]:
    """The run's precision: a test parametrised by ``precision`` runs in it, any other in float64 (the ``tests/saved``
    default, put back after). Replaces the folder's fixture, which fixes float64 for the module."""
    spec = getattr(request.node, "callspec", None)
    precision = spec.params.get("precision", "float64") if spec is not None else "float64"
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", precision == "float64")
    yield
    jax.config.update("jax_enable_x64", held)


#: Free space (the drag holds its resistance) and the tube's wall (it holds its free part's matrix), one drag each.
CASES: dict[str, Callable[[], Experiment]] = {"free_ball": free_ball, "mrs_tube": mrs_tube}
RUN, STEPS = 4, 8


def saved_reference(make: Callable[[], Experiment], folder: Path) -> dict[str, Any]:
    """The case built afresh, run ``RUN`` steps chunked by one and saved: its lock's reference."""
    built = make().build()
    dt = float(built.report.dt)  # type: ignore[arg-type]
    built.run(s(RUN * dt), history=True, hook=lambda chunk: None, chunk=s(dt))
    built.save(folder, steps=STEPS)
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))["reference"]


def _stale(self: mrs_node.MRSDragNode, params: dict[str, Any], moved: dict[str, Any]) -> Any:
    """The drag's matrix made, as ``made_again`` makes it, from the build's leaves, not the moved ones."""
    built = {key: jnp.asarray(value, dtype=jnp.asarray(params[key]).dtype)
             for key, value in self._built_theta().items()}  # noqa: SLF001
    return jit(self._make())(built).astype(moved[mrs_node.HELD].dtype)  # noqa: SLF001


def made_at_every_step(patch: pytest.MonkeyPatch) -> None:
    """The nudged pass given a stale matrix with the leaves it was made at, the build's: no nudged leaf equals them, so
    the step makes the matrix at each nudged step and ignores the one given, as before the change."""
    real = mrs_node.MRSDragNode.made_again

    def at_every_step(self: mrs_node.MRSDragNode, params: dict[str, Any]) -> dict[str, Any]:
        moved = real(self, params)
        if not moved:
            return moved
        built = {f"{mrs_node.HELD_FROM}{key}": jnp.asarray(value, dtype=jnp.asarray(params[key]).dtype)
                 for key, value in self._built_theta().items()}  # noqa: SLF001
        return {**moved, **built, mrs_node.HELD: _stale(self, params, moved)}

    patch.setattr(mrs_node.MRSDragNode, "made_again", at_every_step)


def made_from_the_unnudged_leaves(patch: pytest.MonkeyPatch) -> None:
    """The nudged pass given a stale matrix with the moved leaves as the ones it was made at: the step keeps it."""
    real = mrs_node.MRSDragNode.made_again

    def stale(self: mrs_node.MRSDragNode, params: dict[str, Any]) -> dict[str, Any]:
        moved = real(self, params)
        if not moved:
            return moved
        return {**moved, mrs_node.HELD: _stale(self, params, moved)}

    patch.setattr(mrs_node.MRSDragNode, "made_again", stale)


type References = Callable[[str, str], dict[str, Any]]


@pytest.fixture(scope="module")
def references(tmp_path_factory: pytest.TempPathFactory) -> Iterator[References]:
    """Each case's saved reference under each arm and precision, built when a test first asks for it."""
    found: dict[tuple[str, str, bool], dict[str, Any]] = {}
    arms: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], ...]] = {
        "shipped": (), "made by the step": (made_at_every_step,), "stale": (made_from_the_unnudged_leaves,)}

    def get(case: str, arm: str) -> dict[str, Any]:
        key = (case, arm, bool(jax.config.jax_enable_x64))
        if key not in found:
            with pytest.MonkeyPatch.context() as patch:
                for plant in arms[arm]:
                    plant(patch)
                found[key] = saved_reference(CASES[case], tmp_path_factory.mktemp(case) / arm.replace(" ", "_"))
        return found[key]

    yield get


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("case", list(CASES))
def test_the_resolution_and_reference_are_the_steps_that_make_the_matrix_at_every_step_bit_for_bit(
        case: str, precision: str, references: References):
    shipped, before = references(case, "shipped"), references(case, "made by the step")
    assert shipped["sha256"] == before["sha256"]
    assert shipped["resolution"] == before["resolution"]
    assert any(value for held in shipped["resolution"].values() for value in held.values()), \
        "the rerun moved something, so equal resolutions are evidence"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("case", list(CASES))
def test_control_a_nudged_pass_given_a_matrix_made_from_the_unnudged_leaves_records_another_resolution(
        case: str, precision: str, references: References):
    shipped, stale = references(case, "shipped"), references(case, "stale")
    assert stale["sha256"] == shipped["sha256"], "the plant moves the rerun alone, never the reference"
    assert stale["resolution"] != shipped["resolution"]


def counted_save(make: Callable[[], Experiment], folder: Path, plant: Callable[[pytest.MonkeyPatch], None] | None
                 ) -> dict[str, int]:
    """The matrices made, counted: in the step (a callback in the drag's matrix, the build's check of host callbacks
    patched away for this program) during the run and during the save, and by ``made_again``."""
    counts = {"run": 0, "save": 0, "made_again": 0}
    where = {"now": "run"}

    def tick() -> None:
        counts[where["now"]] += 1

    with pytest.MonkeyPatch.context() as patch:
        if plant is not None:
            plant(patch)
        held_or_made = mrs_node.held_or_made

        def counting(make: Callable[..., Any], built: Any) -> Any:
            def counted(theta: Any) -> Any:
                found = make(theta)
                jax.debug.callback(tick)
                return found
            return held_or_made(counted, built)

        real_again = mrs_node.MRSDragNode.made_again

        def again(self: mrs_node.MRSDragNode, params: dict[str, Any]) -> dict[str, Any]:
            found = real_again(self, params)
            counts["made_again"] += int(mrs_node.HELD in found)
            return found

        patch.setattr(mrs_node, "held_or_made", counting)
        patch.setattr(mrs_node.MRSDragNode, "made_again", again)
        patch.setattr(_construct, "node_host_callbacks", lambda graph, name: set())
        built = make().build()
        dt = float(built.report.dt)  # type: ignore[arg-type]
        built.run(s(RUN * dt), history=True, hook=lambda chunk: None, chunk=s(dt))
        jax.effects_barrier()
        where["now"] = "save"
        built.save(folder, steps=STEPS)
        jax.effects_barrier()
    return counts


@pytest.mark.parametrize("case", list(CASES))
def test_the_nudged_pass_makes_the_drags_matrix_once_and_the_step_none(case: str, tmp_path: Path):
    assert counted_save(CASES[case], tmp_path / "saved", None) == {"run": 0, "save": 0, "made_again": 1}


@pytest.mark.parametrize("case", list(CASES))
def test_control_the_step_that_makes_the_matrix_at_every_step_is_counted_at_each_nudged_step(
        case: str, tmp_path: Path):
    counted = counted_save(CASES[case], tmp_path / "saved", made_at_every_step)
    assert counted == {"run": 0, "save": STEPS, "made_again": 1}


def history_with(built: Any, name: str, overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Two steps of the built graph (a graph step, as a run takes it) with ``overrides`` written over the drag's
    parameters, the graph's parameters put back after."""
    held = _maddening._held(built.graph)  # noqa: SLF001
    tree = held.params
    nodes = {owner: dict(leaves) for owner, leaves in tree["nodes"].items()}
    nodes[name].update(overrides)
    held.params = {**tree, "nodes": nodes}
    try:
        built.graph.reset_state()
        return built._history_on(built.graph, 2, None)  # noqa: SLF001
    finally:
        held.params = tree


def same(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> bool:
    return all(np.array_equal(np.asarray(left[node][field]), np.asarray(right[node][field]))
               for node in left for field in left[node])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_the_stale_read_is_unreachable_a_step_given_every_body_leaf_as_the_rerun_moves_it_makes_the_matrix(
        precision: str):
    """A step given every body leaf one ulp up (as the rerun moves them) and the matrix as built, with the leaves it was
    made at (``held_from``) as built, makes the matrix at the moved leaves: its history is the one given the matrix made
    there with its leaves, and not the build's. The earlier key, value equality with the rerun's leaves, kept the
    build's matrix here. Given the matrix made there and its leaves, the step keeps it; one leaf alone moved, it makes
    the matrix too."""
    built = free_ball().build()
    drag = next(name for name in built.graph.node_names if isinstance(built.graph.get_node(name), mrs_node.MRSDragNode))
    node = built.graph.get_node(drag)
    leaves = node.body_leaf_names()
    params = _maddening._held(built.graph).params["nodes"][drag]  # noqa: SLF001
    up = {key: _maddening.device_array(held if (held := one_ulp_up_leaf(params[key])) is not None
                                       else np.asarray(params[key])) for key in leaves}
    moved = {key for key in leaves if not np.array_equal(np.asarray(up[key]), np.asarray(params[key]))}
    assert moved == {"shape_radius", "part_rotation", "blob"}, "the ball's centre is zero, which the rerun keeps"
    dtype = np.asarray(params[mrs_node.HELD]).dtype

    def made_at(theta: dict[str, Any]) -> dict[str, Any]:
        return {mrs_node.HELD: _maddening.device_array(np.asarray(jit(node._make())(theta)).astype(dtype)),  # noqa: SLF001
                **{f"{mrs_node.HELD_FROM}{key}": value for key, value in theta.items()}}

    assert not np.array_equal(np.asarray(made_at(up)[mrs_node.HELD]), np.asarray(params[mrs_node.HELD])), \
        "the matrix differs at the moved leaves, so the comparison can tell"

    as_built = history_with(built, drag, {})
    every_leaf_up = history_with(built, drag, up)
    assert same(every_leaf_up, history_with(built, drag, {**up, **made_at(up)})), "the step makes the matrix there"
    assert not same(every_leaf_up, as_built), "and does not keep the build's"

    radius_only = {"shape_radius": up["shape_radius"]}
    radius_made = made_at({k: radius_only.get(k, params[k]) for k in leaves})
    assert same(history_with(built, drag, radius_only), history_with(built, drag, {**radius_only, **radius_made})), \
        "one leaf moved: the step makes the matrix"
    assert not same(history_with(built, drag, radius_only), as_built)
