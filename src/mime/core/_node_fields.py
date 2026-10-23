"""Typed views of a node's state and inputs.

MADDENING's ``SimulationNode`` annotates ``initial_state``, ``update`` and ``compute_boundary_fluxes`` with bare
``dict``. An override may not narrow a parameter (``state: ToyBodyState`` is refused by the checker's override rule,
since a plain ``dict`` is not a TypedDict), so a node keeps MADDENING's signature, written ``state: NodeFields`` and
``boundary_inputs: Mapping[str, Array]``, and takes a typed view inside: a closed TypedDict per node for its state and
one for its inputs, each value a jaxtyping array (``Float[Array, "3"]``). Both functions here are static only: at run
time a view is the mapping it was given, and nothing is checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from jax import Array

type NodeFields = dict[str, Array]


def typed_view[V: Mapping[str, Array]](kind: type[V], fields: Mapping[str, Array]) -> V:
    """``fields`` seen as ``kind``, a closed TypedDict of arrays (or one with ``extra_items`` arrays, for fields
    named at construction). A TypedDict that is not closed, or holds anything but arrays, is refused by the checker:
    it is not a ``Mapping[str, Array]``."""
    return cast(V, fields)


def node_fields(view: Mapping[str, Array]) -> NodeFields:
    """A typed state or output, ``ToyBodyState(x=..., v=...)``, as the ``dict`` MADDENING takes. Constructing the
    TypedDict is the check: a missing field, an extra one, or a value that is not an array is a type error."""
    return dict(view)
