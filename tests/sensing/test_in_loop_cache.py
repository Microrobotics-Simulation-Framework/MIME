"""The static structure's cache: made again whenever what it was made from differs, and served while it does not.

The node holds, beside the cache, every number the static part reads (the nine intrinsics, the scene's numbers, the
camera's pose); at a step where any differs from the step's own it makes the cache again. Three claims:

* a run continued with another leaf draws the whole frame at it (the review of #266 found the tube drawn at the old
  ``fy`` and the ball at the new one);
* a held number that differs, whichever of them, erases a cache written into the state, and one that does not leaves
  it standing (the cache is *served*: a camera that renders the static structure in every frame is correct and dearer,
  and no image-level test can tell it);
* the same holds for a state carried into a run with other numbers, which is the first claim at the state.

**The controls.** ``CAMERA_PLANT=numbers-not-compared`` is the old guard (the cache made at the first step only);
``CAMERA_PLANT=cache-every-frame`` makes it at every frame. Each fails the cases of its claim on every run."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import DETECTOR, ballistic, planted

from mime.build._maddening import set_node_state
from mime.sensing._node import CameraNode
from mime.units import ms

if planted("numbers-not-compared"):
    original = CameraNode._static_part  # noqa: SLF001

    def not_compared(self, intrinsics, numbers, pose, dtype, count, held):
        return original(self, intrinsics, held[4], pose, dtype, count, held)

    CameraNode._static_part = not_compared  # type: ignore[method-assign]  # noqa: SLF001

if planted("cache-every-frame"):
    original_served = CameraNode._static_part  # noqa: SLF001

    def every_frame(self, intrinsics, numbers, pose, dtype, count, held):
        return original_served(self, intrinsics, numbers, pose, dtype, count * 0, held)

    CameraNode._static_part = every_frame  # type: ignore[method-assign]  # noqa: SLF001

MARKER = 0.25


def case(fy: float = 750.0, hide: tuple[str, ...] = (), period=ms(3)):
    return ballistic(period, hide=hide, absorbers=(("tube", 3e5),), detector=dataclasses.replace(DETECTOR, fy=fy))


def expected(built) -> np.ndarray:
    return np.asarray(built.current_state()["camera:"]["expected"], dtype=np.float64).copy()


@pytest.mark.parametrize("hide", [("ball",), ()], ids=["tube-only", "ball-and-tube"])
def test_a_continued_run_with_another_leaf_draws_the_frame_at_it(dtype, hide):
    built = case(hide=hide).build()
    built.run(ms(6))
    before = expected(built)
    built.run(ms(6), override={"camera:.detector.fy": 800.0})
    fresh = case(800.0, hide).build()
    fresh.run(ms(12))
    assert np.abs(expected(built) - before).max() > 100.0  # the leaf moved the image (the tube's edge), so this can tell
    assert np.abs(expected(built) - expected(fresh)).max() < 1e-2


def test_the_cache_is_served_while_its_numbers_hold_and_made_again_when_any_differs(dtype):
    built = case(period=ms(1), hide=("ball",)).build()
    built.run(ms(2))
    state = {name: jnp.asarray(value) for name, value in built.current_state()["camera:"].items()}
    clean = expected(built)
    numbers = state["cache_from"]
    marked = {**state, "cache_transmission": jnp.full_like(state["cache_transmission"], MARKER)}

    set_node_state(built._graph, "camera:", marked)  # noqa: SLF001
    built.run(ms(1))
    assert np.allclose(expected(built), 1000.0 * MARKER), "a cache whose numbers hold is read, not rendered again"

    for index in range(numbers.shape[0]):
        set_node_state(built._graph, "camera:", {**marked, "cache_from": numbers.at[index].add(1.0)})  # noqa: SLF001
        built.run(ms(1))
        assert np.abs(expected(built) - clean).max() == 0.0, f"number {index} of {numbers.shape[0]} is not compared"
