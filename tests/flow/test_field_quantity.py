"""SF-2's quantity rule: a field kind a medium carries declares the quantities it provides (``Dynamics.provides``, the
vocabulary's field interfaces, checked at registration), and a law finds the medium's field by quantity
(``PlanContext.fields_providing``, ``field_for``): the one field providing it, refused when none does and when several
do unless the law names one with ``field=``. The flow kinds' declarations agree with their solved-field contract's."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mime.dynamics import FVM, Womersley
from mime.dynamics.flow._fields import fvm_field
from mime.gradients import GradientStatus
from mime.registry import Dynamics, PlanContext, Problem, register_dynamics
from mime.things import ThingKind
from mime.vocabulary import FLOW_BODY_FORCE, FLOW_VELOCITY, register_namespace

V = FLOW_VELOCITY.name
register_namespace("fieldrule", "0.0.1")


def context(fields: dict[str, tuple[str, ...]]) -> PlanContext:
    return PlanContext(0.001, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), {"medium": "water"}, {"medium": None},
                       provided={"medium": fields})


def test_a_law_finds_the_one_field_providing_a_quantity_and_is_refused_where_none_or_several_do():
    one = context({"water.flow": (V,)})
    assert one.fields_providing("medium", V) == ("water.flow",)
    assert one.field_for("medium", V) == "water.flow"
    none = context({"water.flow": ()}).field_for("medium", V)
    assert isinstance(none, Problem) and none.code == "binding.field-quantity"
    two = context({"water.flow": (V,), "water.slip": (V,)})
    ambiguous = two.field_for("medium", V)
    assert isinstance(ambiguous, Problem) and ambiguous.code == "binding.field-ambiguous"
    assert "water.flow" in ambiguous.message and "water.slip" in ambiguous.message
    assert two.field_for("medium", V, named="water.slip") == "water.slip"
    wrong = two.field_for("medium", V, named="water.drug")
    assert isinstance(wrong, Problem) and wrong.code == "binding.field-quantity"


def test_the_flow_kinds_provide_the_velocity_as_their_solved_field_declares():
    assert FVM.provides == (V,) and Womersley.provides == (V,)
    assert fvm_field("imex_euler").provides == FVM.provides and Womersley.field.provides == Womersley.provides


@pytest.mark.parametrize(("provides", "applies", "problem"), [
    ((FLOW_BODY_FORCE.name,), ThingKind.MEDIUM, "not a field interface"),
    (("fieldrule.nothing",), ThingKind.MEDIUM, "not a field interface"),
    ((V,), ThingKind.BODY, "only a field kind a medium carries"),
])
def test_a_dynamics_providing_what_is_not_a_medium_fields_quantity_is_refused_at_registration(provides, applies,
                                                                                              problem):
    """The control is the flow kinds themselves, which register."""
    body = {"name": f"fieldrule.provider_{provides[0].replace('.', '_')}_{applies.value}", "version": "0.0.1",
            "differentiability": GradientStatus.EXACT, "applies_to": frozenset({applies}), "provides": provides}
    with pytest.raises(TypeError, match=problem):
        register_dynamics(dataclass(frozen=True, kw_only=True)(type("Provider", (Dynamics,), body)))
