"""Build-time probe of what a run executes: each node's step code, traced once and never executed.

A run is MADDENING's one jit-compiled step, scanned. Whatever MIME adds above its nodes runs at build time, and what
runs on the host between compiled calls is named in the run's structure; nothing runs on the host inside a step. A
host callback in a node's step code breaks that, so the build refuses it, naming the node.

Each node is traced alone through MADDENING's public node interface: its state, the inputs the graph resolves for it
(an unconnected input at zero), its ``update`` and its ``compute_boundary_fluxes``. So the probe reads nothing private
and needs no compiled step. It sees every node MIME or a lab writes; MADDENING's own coupling and mapping code is
MADDENING's to keep free of callbacks.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator

from jax.extend.core import Jaxpr, subjaxprs
from maddening import GraphManager

from mime.build._maddening import (
    make_jaxpr,
    node_fluxes,
    node_state,
    node_update,
    parameter_leaves,
    resolved_inputs,
    zero_inputs,
)

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


def _traced(method: Callable[..., object], state: object, given: object, dt: float, leaves: object) -> Jaxpr:
    if "params" in inspect.signature(method).parameters:
        def call(state: object, given: object, leaves: object) -> object:
            return method(state, given, dt, params=leaves)

        return make_jaxpr(call, state, given, leaves)

    def call_plain(state: object, given: object) -> object:
        return method(state, given, dt)

    return make_jaxpr(call_plain, state, given)


def node_host_callbacks(graph: GraphManager, name: str) -> set[str]:
    """The host callbacks in one node's step code, its ``update`` and its ``compute_boundary_fluxes``, each traced
    alone with the inputs the graph resolves for the node and every unconnected input it declares at zero."""
    node = graph.get_node(name)
    given = {**zero_inputs(node), **resolved_inputs(graph, name)}
    state = node_state(graph, name)
    leaves = parameter_leaves(graph).get(name)
    return {call for method in (node_update(node), node_fluxes(node))
            for call in host_callbacks(_traced(method, state, given, node.delta_t, leaves))}


__all__ = ["HOST_CALLBACKS", "host_callbacks", "node_host_callbacks"]
