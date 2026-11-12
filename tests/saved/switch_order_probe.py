"""One process of ``test_switch_order.py``: the order that made the switch scan answer differently, run for real.

``python switch_order_probe.py polluted|clean`` traces the articulated arm's step (``case/dynamic_rpm``) as the scan does,
``polluted`` after a test of the tube tree's union (``tests/scene/test_tube_tree.py::test_its_normal_is_unit_and_its_nearest_point_on_the_wall``,
which ``pytest`` ran first in the order that failed) has traced the union in this process, and prints how many equations of the arm's
step carry a frame in ``scene/_union.py``: the arm runs no union, so the scan's answer is 0. The environment variable
``MIME_PLANT_STALE_FRAMES`` traces as the scan did before ``trace_fresh`` (``make_jaxpr`` on whatever JAX had cached): the plant of the
detection rate in the pull request that fixed this."""

import json
import os
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS), str(TESTS / "saved"), str(TESTS / "scene")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import mime.docs._switches as switches  # noqa: E402


def trace_the_union() -> None:
    """What the tube-tree test does: a union's normals and distances at 400 points, traced in this process."""
    from union_oracles import D, y_tree

    tree = y_tree("asymmetric", 0.1)
    lumen = tree.lumen()
    held = {name: jnp.asarray(value) for name, value in tree.parameters().items()}
    points = jnp.asarray(np.random.default_rng(6).uniform(-0.8 * D, 0.8 * D, (400, 3)))
    found = lumen.outward_normal(held, points, lumen.plan(held, points, margin=0.0))
    nearest = points - found.distance.distance[:, None] * found.normal
    lumen.signed_distance(held, nearest, lumen.plan(held, nearest, margin=0.0))


def union_frames_in_the_arms_step() -> int:
    from switch_configurations import configurations

    from mime.build._maddening import node_state, parameter_leaves, resolved_inputs, zero_inputs

    configuration, name = "case/dynamic_rpm", "rpm.arm"
    graph = configurations()[configuration]().build().graph
    node = graph.get_node(name)
    given = {**zero_inputs(node), **resolved_inputs(graph, name)}
    step = switches.traced_step(configuration, name, node, node_state(graph, name), given, parameter_leaves(graph).get(name))
    return sum(any(frame.file.replace("\\", "/").endswith("scene/_union.py") for frame in switches.equation_frames(equation))
               for _, equation in switches.equations(step.jaxpr))


if __name__ == "__main__":
    if os.environ.get("MIME_PLANT_STALE_FRAMES"):
        switches.trace_fresh = lambda function, *arguments: switches.make_jaxpr(function, *arguments)
    if sys.argv[1] == "polluted":
        trace_the_union()
    print(json.dumps({"union_frames": union_frames_in_the_arms_step()}))
