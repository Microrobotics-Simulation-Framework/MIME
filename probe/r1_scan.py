"""Camera spike R1, item 4 (report only): the switch scan over the camera node's step traced forward, as the scan traces
it, and traced in reverse mode (the step's vjp in its leaves and its state), with and without the render checkpointed.

The scan's sites come from the source (a function's raw operations) and its nesting from the traces (a switch function's
equation inside a transform). A reverse-mode trace runs the same Python, so its sites are the forward's; what it could
add is nesting: a switch function's equation under a transform the forward trace does not have. Prints the hits each
trace gives beyond the forward's and writes JSON.

    python probe/r1_scan.py OUT.json    (MIME_SPIKE_RENDER_CHECKPOINT read when the node is traced)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    import r1_core as core

    from mime.build._maddening import make_jaxpr, node_state, parameter_leaves, resolved_inputs, zero_inputs
    from mime.docs._switches import TracedStep, scan_switches, step_function, traced_step
    from mime.sensing._node import CameraNode

    import mime

    package = Path(mime.__file__).parent
    built = core.case(width=64).build()
    graph = built.graph
    node = graph.get_node(core.CAMERA)
    given = {**zero_inputs(node), **resolved_inputs(graph, core.CAMERA)}
    state = node_state(graph, core.CAMERA)
    leaves = parameter_leaves(graph).get(core.CAMERA)
    forward = traced_step("forward", core.CAMERA, node, state, given, leaves)
    step = step_function(node)
    dt = node.delta_t

    def pulled(state, given, leaves):
        def run(leaves, state):
            out = step(state, given, dt, params=leaves)
            return jax.tree_util.tree_reduce(lambda a, b: a + b, [jax.numpy.sum(x) for x in jax.tree_util.tree_leaves(out)
                                                                 if jax.numpy.issubdtype(x.dtype, jax.numpy.floating)])

        return jax.grad(run, argnums=(0, 1), allow_int=True)(leaves, state)

    reverse = TracedStep("reverse", core.CAMERA, CameraNode, make_jaxpr(pulled, state, given, leaves))
    alone = scan_switches([forward], package)
    both = scan_switches([forward, reverse], package)
    camera_hits = sorted(site for site in both.hits if site.startswith("sensing/") or site.startswith("observations/"))
    added = sorted(set(both.hits) - set(alone.hits))
    report = {"render_checkpoint": os.environ.get("MIME_SPIKE_RENDER_CHECKPOINT") == "1",
              "forward_hits": len(alone.hits), "forward_and_reverse_hits": len(both.hits),
              "added_by_reverse": added, "camera_sites": len(camera_hits),
              "nested_in_camera": [s for s in camera_hits if " nested_" in s],
              "scope_functions_forward": len(alone.scope), "scope_functions_with_reverse": len(both.scope)}
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "added_by_reverse"}, indent=1), "added:", added[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
