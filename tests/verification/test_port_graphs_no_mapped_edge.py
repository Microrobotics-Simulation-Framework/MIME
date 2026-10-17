"""The ported graphs (FL-9, the Stokes-drag slice) hold no interface-mapped edge.

MADDENING 0.4 keeps a mapped edge's weights in ``gm.params["mappings"]``, the
tree that holds node constants.  The MADDENING 0.4.1 port leaves mappings out of
scope on the claim that these graphs have none; this makes the claim a check.
The positive control adds one mapped edge to the FL-9 graph, so the check is
shown to see one.
"""

import warnings

import jax.numpy as jnp
from maddening.core.coupling.mapping import matrix_mapping

from mime.experiments.dejongh import build_graph
from mime.experiments.stokes_drag_demo import make_experiment


def _compiled(gm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _mapped(gm) -> tuple[list[str], list[str]]:
    edges = [repr(e) for e in gm.edges if e.mapping is not None]
    return edges, sorted(gm.params["mappings"])


def test_fl9_graph_has_no_mapped_edge():
    assert _mapped(_compiled(build_graph())) == ([], [])


def test_stokes_drag_slice_has_no_mapped_edge():
    gm, _ = make_experiment().build()
    assert _mapped(_compiled(gm)) == ([], [])


def test_positive_control_one_mapped_edge_is_seen():
    gm = build_graph()
    gm.add_edge("body", "lub", "velocity", "background_velocity",
                mapping=matrix_mapping(jnp.eye(3)))
    edges, slots = _mapped(_compiled(gm))
    assert len(edges) == 1 and len(slots) == 1
