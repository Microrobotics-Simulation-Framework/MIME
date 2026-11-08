"""The rft rung's node through MADDENING's battery (``mime.testing.assert_nodes_verified``), on R6's helix in free
space, with the drag's drawn pose a body's (``node_cases.DRAG``'s bounds and unit quaternion); and its step traced by
``jax.jit``. Not with the tube's table: the battery batches and differentiates the step over its draws, and with the
table's two reads a pair of points that held 57 GiB before the kernel stopped it (a laptop, R1's build); the MRS's
walled node is not run through it either. The walled step's derivative is held by
``tests/oracles/test_rft_gradients.py`` and its stop by ``tests/oracles/test_rft.py``."""

import sys
from pathlib import Path

import jax
from rft_cases import R6_HELIX, R6_SETTINGS, helix_body, rft, rft_free_space

from mime.build._maddening import node_state, parameter_leaves
from mime.interactions import Drag
from mime.testing import assert_nodes_verified

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "saved"))
from node_cases import DRAG  # noqa: E402


def test_the_rft_node_passes_the_battery_in_free_space():
    assert_nodes_verified(helix_body("battery", R6_HELIX, R6_SETTINGS, rft(wall_kind="free_space")).build(), Drag, DRAG)


def test_the_step_traces():
    built = rft_free_space().build()
    node = built.graph.get_node("drag:coil")
    state, pose = node_state(built.graph, "drag:coil"), node_state(built.graph, "coil")["pose"]
    params = parameter_leaves(built.graph)["drag:coil"]
    out = jax.jit(lambda s, p: node.update(s, {"pose": pose}, 1e-3, params=p))(state, params)
    assert out["resistance"].shape == (6, 6)
