"""The save's resolution rerun on an MRS drag: the nudged pass (the reference steps run again from a start one ulp off,
the parameters the step reads moved too) makes each drag's held matrix once, for the moved leaves, and gives it to the
step, which keeps it there rather than making it at each step (``MimeNode.made_again``, ``held_or_made``'s second key).

Each arm is a fresh build, run four steps chunked by one and saved, in float64: its nodes are its own, so each step is
traced with the arm's patches. The resolution the save records, and its reference, are bit for bit those of the step
that makes the matrix itself at every nudged step (the second key taken away, the save before the change), given a stale
matrix it must then ignore: a step that kept the given matrix would record the stale arm's resolution, so equality is
evidence that the step made it. The control of the bits: a nudged pass given a matrix made from the un-nudged leaves
records another resolution. The count is a callback in the drag's own matrix, so the counted builds are a program of
their own and are judged only by their counts; their control, the step that makes the matrix at every step, counts one
per nudged step, so the count sees the step's making."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import pytest
from mrs_cases import BALL, free_space, mrs, mrs_tube

from mime.build import _construct
from mime.core import jit, jnp
from mime.experiment import Experiment
from mime.nodes.hydro.mrs import _node as mrs_node
from mime.nodes.hydro.mrs import _quadrature
from mime.saved import LOCK_FILE
from mime.units import kg, kg_m2, m_per_s, mm, rad_per_s, s


def free_ball() -> Experiment:
    """A ball in free space, moving and turning from the start, its drag holding its resistance."""
    return free_space("free_ball", {"ball": BALL}, mrs(mm(1.5)), mass=kg(1e-7), inertia=kg_m2((1e-14, 2e-14, 3e-14)),
                      velocity=m_per_s((1e-3, -2e-3, 0.5e-3)), spin=rad_per_s((3.0, -1.0, 2.0)))


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


def made_at_every_step(patch: pytest.MonkeyPatch) -> None:
    """The step given no second key: the save before the change, which makes the matrix at each nudged step."""
    patch.setattr(mrs_node, "held_or_made", lambda make, built, given=(): _quadrature.held_or_made(make, built))


def made_from_the_unnudged_leaves(patch: pytest.MonkeyPatch) -> None:
    """The nudged pass given a stale matrix: made, as ``made_again`` makes it, from the build's leaves, not the moved
    ones."""
    real = mrs_node.MRSDragNode.made_again

    def stale(self: mrs_node.MRSDragNode, params: dict[str, Any]) -> dict[str, Any]:
        moved = real(self, params)
        if not moved:
            return moved
        built = {key: jnp.asarray(value, dtype=jnp.asarray(params[key]).dtype)
                 for key, value in self._built_theta().items()}  # noqa: SLF001
        return {mrs_node.HELD: jit(self._make())(built).astype(moved[mrs_node.HELD].dtype)}  # noqa: SLF001

    patch.setattr(mrs_node.MRSDragNode, "made_again", stale)


type References = Callable[[str, str], dict[str, Any]]


@pytest.fixture(scope="module")
def references(tmp_path_factory: pytest.TempPathFactory) -> Iterator[References]:
    """Each case's saved reference under each arm, built when a test first asks for it."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    arms: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], ...]] = {
        "shipped": (), "made by the step": (made_at_every_step, made_from_the_unnudged_leaves),
        "stale": (made_from_the_unnudged_leaves,)}

    def get(case: str, arm: str) -> dict[str, Any]:
        if (case, arm) not in found:
            with pytest.MonkeyPatch.context() as patch:
                for plant in arms[arm]:
                    plant(patch)
                found[case, arm] = saved_reference(CASES[case], tmp_path_factory.mktemp(case) / arm.replace(" ", "_"))
        return found[case, arm]

    yield get


@pytest.mark.parametrize("case", list(CASES))
def test_the_resolution_and_reference_are_the_steps_that_make_the_matrix_at_every_step_bit_for_bit(
        case: str, references: References):
    shipped, before = references(case, "shipped"), references(case, "made by the step")
    assert shipped["sha256"] == before["sha256"]
    assert shipped["resolution"] == before["resolution"]
    assert any(value for held in shipped["resolution"].values() for value in held.values()), \
        "the rerun moved something, so equal resolutions are evidence"


@pytest.mark.parametrize("case", list(CASES))
def test_control_a_nudged_pass_given_a_matrix_made_from_the_unnudged_leaves_records_another_resolution(
        case: str, references: References):
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

        def counting(make: Callable[..., Any], built: Any, given: Any = ()) -> Any:
            def counted(theta: Any) -> Any:
                found = make(theta)
                jax.debug.callback(tick)
                return found
            return held_or_made(counted, built, given)

        real_again = mrs_node.MRSDragNode.made_again

        def again(self: mrs_node.MRSDragNode, params: dict[str, Any]) -> dict[str, Any]:
            found = real_again(self, params)
            counts["made_again"] += len(found)
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
