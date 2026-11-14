"""The node count and the compiled step's operation count of a case: the instructions of the optimised HLO of one
step of the graph as a chunk runs it (``mime.build._maddening.chunk_step``), each computation's, a fusion's body
included, and the instructions outside fusions (what XLA schedules)."""

from __future__ import annotations

import json
import re
import sys

import jax

import probe_cases
from mime.build._maddening import chunk_step, fed_inputs, full_state, graph_params

_INSTRUCTION = re.compile(r"^\s*(?:ROOT )?%[\w.\-]+ = ")


def counts(name: str, dtype: str) -> dict[str, object]:
    jax.config.update("jax_enable_x64", dtype == "float64")
    built = getattr(probe_cases, name)().build()
    graph = built.graph
    step = chunk_step(graph)
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    params = graph_params(graph)
    external = fed_inputs(graph, None)
    text = jax.jit(step).lower(state, external, params).compile().as_text()
    total, outside, inside = 0, 0, False
    for line in text.splitlines():
        if line.startswith(("%fused", "fused_computation")) or re.match(r"^%?fused", line):
            inside = True
        elif line.startswith(("ENTRY", "%")) and line.rstrip().endswith("{"):
            inside = "fused" in line.split()[0]
        if _INSTRUCTION.match(line):
            total += 1
            outside += 0 if inside else 1
    return {"case": name, "dtype": dtype, "nodes": len(graph.node_names), "node_names": list(graph.node_names),
            "hlo_instructions": total, "hlo_instructions_outside_fusions": outside,
            "derived_hash": built.report.derived_hash}


if __name__ == "__main__":
    print(json.dumps(counts(sys.argv[1], sys.argv[2])))
