"""A saved field's kind is its type in its node's typed state (``mime.core.FieldKind``), never its node's class: a new
node whose state TypedDict writes a field as ``Twist`` is judged as a twist with nothing declared beside it, sized by its
radius of gyration where no hull is known; ``Ratio`` against one; ``HeldPoints`` paired with the one ``HeldForces``.

The controls: a TypedDict that does not name exactly the node's fields types none of them, and two that type one field
differently leave it judged against its own largest magnitude."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import FieldKind, HeldForces, HeldPoints, MimeNode, NodeFields, Ratio, Twist, field_kinds, node_fields
from mime.saved._scales import OWN, SourceScale, source_scales


class NewBodyState(TypedDict, closed=True):
    pose: ReadOnly[Float[Array, "7"]]
    twist: ReadOnly[Twist]


class NewBodyNode(MimeNode):
    """A body no table lists: its state's twist is typed, and it holds a mass and three moments."""

    def __init__(self, name: str, timestep: float, *, mass: float, inertia: tuple[float, float, float]) -> None:
        super().__init__(name, timestep, mass=mass, inertia=inertia)

    def initial_state(self) -> NodeFields:
        return node_fields(NewBodyState(pose=jnp.zeros(7), twist=jnp.zeros(6)))

    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return dict(state)


class GaugeState(TypedDict, closed=True):
    share: ReadOnly[Ratio]
    held: ReadOnly[HeldPoints]
    taken: ReadOnly[HeldForces]
    count: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    extra: NotRequired[ReadOnly[Twist]]


class GaugeNode(MimeNode):
    def initial_state(self) -> NodeFields:
        return node_fields(GaugeState(share=jnp.zeros(()), held=jnp.zeros((2, 3)), taken=jnp.zeros(2),
                                      count=jnp.zeros(())))

    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        return dict(state)


class OtherReading(TypedDict, closed=True):
    """Names the gauge's fields too, and types its share otherwise."""

    share: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    held: ReadOnly[HeldPoints]
    taken: ReadOnly[HeldForces]
    count: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape


def test_a_new_nodes_twist_typed_field_gets_the_twist_scale_sized_by_its_radius_of_gyration():
    node = NewBodyNode("probe", 1e-3, mass=2e-6, inertia=(1e-12, 2e-12, 3e-12))
    assert field_kinds(node, set(node.initial_state())) == {"pose": FieldKind.OWN, "twist": FieldKind.TWIST}
    scales = source_scales({"probe": node}, {"probe": ("pose", "twist")}, {})
    assert scales == {"probe.twist": SourceScale(FieldKind.TWIST, pytest.approx(np.sqrt(3e-12 / 2e-6), rel=1e-12),
                                                 "its radius of gyration about its widest axis, from its mass and "
                                                 "inertia")}


def test_a_twist_with_nothing_to_size_it_by_is_judged_against_its_own_largest_magnitude():
    class Unsized(NewBodyNode):
        def __init__(self, name: str, timestep: float) -> None:
            MimeNode.__init__(self, name, timestep)

    node = Unsized("probe", 1e-3)
    assert source_scales({"probe": node}, {"probe": ("pose", "twist")}, {}) == {"probe.twist": OWN}


def test_a_ratio_is_judged_against_one_and_held_points_pair_with_the_one_field_of_held_forces():
    """``OtherReading`` names the same fields and types ``share`` otherwise: the two disagree on it, so it is judged
    against its own largest magnitude, while both type the held points and forces alike."""
    node = GaugeNode("gauge", 1e-3)
    kinds = field_kinds(node, set(node.initial_state()))
    assert kinds == {"count": FieldKind.OWN, "held": FieldKind.HELD_POINTS, "share": FieldKind.OWN,
                     "taken": FieldKind.HELD_FORCES}
    scales = source_scales({"gauge": node}, {"gauge": tuple(node.initial_state())}, {})
    assert scales == {"gauge.held": SourceScale(FieldKind.HELD_POINTS, forces="gauge.taken")}


def test_control_without_the_disagreeing_reading_the_ratio_is_typed(monkeypatch: pytest.MonkeyPatch):
    import sys

    from mime.core import _field_kinds

    monkeypatch.delattr(sys.modules[__name__], "OtherReading")
    _field_kinds._kinds.cache_clear()
    try:
        node = GaugeNode("gauge", 1e-3)
        assert field_kinds(node, set(node.initial_state()))["share"] is FieldKind.RATIO
        assert source_scales({"gauge": node}, {"gauge": tuple(node.initial_state())}, {})["gauge.share"] == \
            SourceScale(FieldKind.RATIO)
    finally:
        _field_kinds._kinds.cache_clear()


def test_control_a_typed_state_that_does_not_name_exactly_the_fields_types_none_of_them():
    node = NewBodyNode("probe", 1e-3, mass=2e-6, inertia=(1e-12, 2e-12, 3e-12))
    assert field_kinds(node, {"pose", "twist", "spin"}) == {"pose": FieldKind.OWN, "spin": FieldKind.OWN,
                                                            "twist": FieldKind.OWN}
    assert field_kinds(node, {"twist"}) == {"twist": FieldKind.OWN}, "a required field missing"
