"""The static structure, a leaf: read while the numbers beside it are the step's, made again whenever any differs.

The node holds the structure (``static_structure``) beside every number it was made at (``static_structure_from``: the
nine intrinsics, the scene's numbers, the camera's pose); a frame step reads it where those are the step's and makes it
where any differs. A run whose chunks make the run-constants and one MADDENING's own scan steps (a recorded run) take the
same choice: the run-constant is the held leaf itself, carrying the make's derivative (a choice there would copy the
structure at every chunk). Three claims:

* a run continued with another leaf draws the whole frame at it (the review of #266 found the tube drawn at the old
  ``fy`` and the ball at the new one);
* a held number that differs, whichever of them, makes the structure again, and one that does not leaves the held one
  read (it is *served*: a camera that renders the static structure in every frame is correct and dearer, and no
  image-level test can tell it), on both routes;
* the state carries nothing of it, so a held step holds the image alone.

**The controls.** ``CAMERA_PLANT=numbers-not-compared`` reads the held structure whatever its numbers;
``CAMERA_PLANT=cache-every-frame`` renders it in every frame and reads nothing held. Each fails the cases of its claim on
every run."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import DETECTOR, ballistic, planted

from mime.build import _maddening
from mime.sensing._node import STRUCTURE, STRUCTURE_FROM, CameraNode
from mime.units import ms

if planted("numbers-not-compared"):
    CameraNode.holds_at = lambda self, p, numbers: jnp.asarray(True)  # type: ignore[method-assign]

if planted("cache-every-frame"):
    CameraNode.holds_at = lambda self, p, numbers: jnp.asarray(False)  # type: ignore[method-assign]

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


@pytest.mark.parametrize("hide", [("ball",), ()], ids=["tube-only", "ball-and-tube"])
def test_a_continued_run_with_another_leaf_draws_the_frame_at_it(dtype, hide):
    built = case(hide=hide).build()
    built.run(ms(6))
    before = expected(built)
    built.run(ms(6), override={"camera:.detector.fy": 800.0})
    fresh = case(800.0, hide).build()
    fresh.run(ms(12))
    assert np.abs(expected(built) - before).max() > 100.0  # the leaf moved the image (the tube's edge), so this can tell
    # exactly equal: the override's tree holds the structure made at its leaves by the same compiled maker as the fresh
    # build's (measured 0.0 in both dtypes, tube only and with the ball), so any difference is a stale part, not rounding
    assert np.abs(expected(built) - expected(fresh)).max() == 0.0


ROUTES = {"run-constants": {}, "history-scan": {"record": ["camera:.frame_step"]}}


@pytest.mark.parametrize("route", list(ROUTES))
def test_the_structure_is_served_while_its_numbers_hold_and_made_again_when_any_differs(dtype, route):
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


def test_the_state_carries_nothing_of_the_structure(dtype):
    """The held step's carry: the camera's state is its clock, the image, the ring and the flags, nothing image-sized
    beyond the two frames, so a held step does not move the structure."""
    built = case().build()
    state = built.current_state()[CAMERA]
    rows, columns = np.shape(state["image"])
    sized = sorted(name for name, value in state.items() if np.size(value) >= rows * columns)
    assert sized == ["expected", "image"], sized
    assert STRUCTURE in _maddening.graph_params(built.graph)["nodes"][CAMERA]
