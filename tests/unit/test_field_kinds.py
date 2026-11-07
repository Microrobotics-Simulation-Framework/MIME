"""A saved field's kind is its type in its node's typed state (``mime.core.FieldKind``), never its node's class: a new
node whose state TypedDict writes a field as ``HeldPoints`` beside one ``HeldForces`` field has its points judged only
on the rows whose force acts, with nothing declared beside it. That is the one kind ``verify`` reads; how a field is
split into blocks comes from its declared units (``tests/saved/test_verify_blocks.py``).

The controls: a TypedDict that does not name exactly the node's fields types none of them, and two that type one field
differently leave it of no kind."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float
from typing_extensions import ReadOnly, TypedDict

from mime.core import FieldKind, HeldForces, HeldPoints, MimeNode, NodeFields, field_kinds, node_fields, \
    paired_forces


class GaugeState(TypedDict, closed=True):
    held: ReadOnly[HeldPoints]
    taken: ReadOnly[HeldForces]
    count: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape


class GaugeNode(MimeNode):
    """A node no table lists: its held points and the forces they took are typed."""

    def initial_state(self) -> NodeFields:
        return node_fields(GaugeState(held=jnp.zeros((2, 3)), taken=jnp.zeros(2), count=jnp.zeros(())))

    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return dict(state)


class SpareState(TypedDict, closed=True):
    pose: ReadOnly[Float[Array, "7"]]
    points: ReadOnly[HeldPoints]


class SpareNode(MimeNode):
    """Held points with no field of held forces beside them."""

    def initial_state(self) -> NodeFields:
        return node_fields(SpareState(pose=jnp.zeros(7), points=jnp.zeros((1, 3))))

    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return dict(state)


def test_a_new_nodes_held_points_pair_with_its_one_field_of_held_forces():
    node = GaugeNode("gauge", 1e-3)
    kinds = field_kinds(node, set(node.initial_state()))
    assert kinds == {"count": FieldKind.OWN, "held": FieldKind.HELD_POINTS, "taken": FieldKind.HELD_FORCES}
    assert paired_forces(kinds) == "taken"


def test_held_points_with_no_field_of_held_forces_pair_with_nothing():
    node = SpareNode("spare", 1e-3)
    kinds = field_kinds(node, set(node.initial_state()))
    assert kinds["points"] is FieldKind.HELD_POINTS and paired_forces(kinds) is None


def test_control_two_typed_states_that_type_one_field_otherwise_leave_it_of_no_kind(monkeypatch: pytest.MonkeyPatch):
    """A second TypedDict beside the node, naming the same fields and typing ``held`` as a plain array."""
    import sys

    from mime.core import _field_kinds

    class OtherReading(TypedDict, closed=True):
        held: ReadOnly[Float[Array, "groups 3"]]  # noqa: F722 - a jaxtyping shape
        taken: ReadOnly[HeldForces]
        count: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape

    _field_kinds._kinds.cache_clear()
    node = GaugeNode("gauge", 1e-3)
    assert field_kinds(node, set(node.initial_state()))["held"] is FieldKind.HELD_POINTS, "the arm before the plant"
    monkeypatch.setattr(sys.modules[__name__], "OtherReading", OtherReading, raising=False)
    _field_kinds._kinds.cache_clear()
    try:
        assert field_kinds(node, set(node.initial_state()))["held"] is FieldKind.OWN
    finally:
        _field_kinds._kinds.cache_clear()


def test_control_a_typed_state_that_does_not_name_exactly_the_fields_types_none_of_them():
    node = GaugeNode("gauge", 1e-3)
    assert field_kinds(node, {"held", "taken", "count", "spin"}) == {
        "count": FieldKind.OWN, "held": FieldKind.OWN, "spin": FieldKind.OWN, "taken": FieldKind.OWN}
    assert field_kinds(node, {"held"}) == {"held": FieldKind.OWN}, "a required field missing"
