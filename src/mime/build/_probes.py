"""Build-time probes of the compiled step: what MADDENING will run, traced once and never executed.

A run is MADDENING's one jit-compiled step, scanned. Whatever MIME adds above its nodes runs at build time, and what
runs on the host between compiled calls is named in the run's structure; nothing runs on the host inside a step. A
host callback in the traced step breaks that, so the build refuses it, naming the node whose update makes it.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

from jax.extend.core import Jaxpr, subjaxprs
from maddening import GraphManager

from mime.build._maddening import make_jaxpr, node_state, node_update, parameter_leaves, traced_step, zero_inputs

#: The primitives by which traced code calls back into Python: ``jax.pure_callback``, ``jax.experimental.io_callback``,
#: ``jax.debug.callback`` and ``jax.debug.print``.
HOST_CALLBACKS = frozenset({"pure_callback", "io_callback", "debug_callback", "debug_print"})


def _jaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
    yield jaxpr
    for inner in subjaxprs(jaxpr):
        yield from _jaxprs(inner)


def host_callbacks(jaxpr: Jaxpr) -> set[str]:
    """The host-callback primitives anywhere in ``jaxpr``, the bodies of its scans, loops and branches included."""
    return {eqn.primitive.name for held in _jaxprs(jaxpr) for eqn in held.eqns if eqn.primitive.name in HOST_CALLBACKS}


def step_host_callbacks(graph: GraphManager) -> set[str]:
    return host_callbacks(traced_step(graph))


def node_host_callbacks(graph: GraphManager, name: str) -> set[str]:
    """The host callbacks in one node's ``update``, traced alone, with each input it declares at zero: what names
    the node behind a callback the whole step's trace found."""
    node = graph.get_node(name)
    zeros = zero_inputs(node)
    update = node_update(node)
    if "params" in inspect.signature(update).parameters:
        def call(state: object, given: object, leaves: object) -> object:
            return update(state, given, node.delta_t, params=leaves)

        traced = make_jaxpr(call, node_state(graph, name), zeros, parameter_leaves(graph).get(name))
    else:
        def call_plain(state: object, given: object) -> object:
            return update(state, given, node.delta_t)

        traced = make_jaxpr(call_plain, node_state(graph, name), zeros)
    return host_callbacks(traced)


__all__ = ["HOST_CALLBACKS", "host_callbacks", "node_host_callbacks", "step_host_callbacks", "traced_step"]
