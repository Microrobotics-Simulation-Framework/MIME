"""The rft rung's node through MADDENING's battery (``mime.testing.assert_nodes_verified``), on the round-trip cases'
bodies in free space and in the tube, with the drag's drawn pose a body's (``node_cases.DRAG``'s bounds and unit
quaternion); and its step traced by ``jax.jit``."""

import sys
from pathlib import Path

import jax
import pytest
from rft_cases import rft_free_space, rft_tube

from mime.build._maddening import node_state, parameter_leaves
from mime.interactions import Drag
from mime.testing import assert_nodes_verified

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "saved"))
from node_cases import DRAG  # noqa: E402


@pytest.mark.parametrize("make", [rft_free_space, rft_tube], ids=["free_space", "tube"])
def test_the_rft_node_passes_the_battery(make):
    assert_nodes_verified(make().build(), Drag, DRAG)


def test_the_step_traces():
    built = rft_free_space().build()
    node = built.graph.get_node("drag:coil")
    state, pose = node_state(built.graph, "drag:coil"), node_state(built.graph, "coil")["pose"]
    params = parameter_leaves(built.graph)["drag:coil"]
    out = jax.jit(lambda s, p: node.update(s, {"pose": pose}, 1e-3, params=p))(state, params)
    assert out["resistance"].shape == (6, 6)
