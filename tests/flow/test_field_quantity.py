"""SF-2's quantity rule: a dynamics a medium carries is a field kind, and its solved-field declarations
(``Dynamics.field_declarations``, each a ``mime.fields.SolvedField``) are the one statement of the quantities it
provides, checked at registration: a medium's kind that declares nothing, or provides nothing, is refused there. A law
finds the field by quantity (``PlanContext.fields_providing``, ``field_for``): the one field providing it, refused when
none does and when several do unless the law names one with ``field=``."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

from mime.dynamics import FVM, Womersley
from mime.dynamics.flow._fields import fvm_field
from mime.gradients import GradientStatus
from mime.nodes.flow.fvm import SCHEMES
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


def test_the_flow_kinds_provide_the_velocity_by_their_solved_field_declarations():
    assert FVM.field_declarations() == tuple(fvm_field(name) for name in SCHEMES)
    assert Womersley.field_declarations() == (Womersley.field,)
    assert {q for d in (*FVM.field_declarations(), *Womersley.field_declarations()) for q in d.provides} == {V}


def _kind(label: str, applies: ThingKind, declared: tuple[object, ...]) -> type[Dynamics]:
    body = {"name": f"fieldrule.{label}_{applies.value}", "version": "0.0.1",
            "differentiability": GradientStatus.EXACT, "applies_to": frozenset({applies}),
            "field_declarations": classmethod(lambda cls: declared)}
    return dataclass(frozen=True, kw_only=True)(type("Kind", (Dynamics,), body))


@pytest.mark.parametrize(("label", "declared", "applies", "problem"), [
    ("nothing", (), ThingKind.MEDIUM, "a dynamics a medium carries is a field kind"),
    ("provides_nothing", (dataclasses.replace(Womersley.field, provides=()),), ThingKind.MEDIUM,
     "provides at least one quantity"),
    ("a_load", (dataclasses.replace(Womersley.field, provides=(FLOW_BODY_FORCE.name,)),), ThingKind.MEDIUM,
     "not an interface of the vocabulary that a reading gives"),
    ("unknown", (dataclasses.replace(Womersley.field, provides=("fieldrule.nothing",)),), ThingKind.MEDIUM,
     "not an interface of the vocabulary that a reading gives"),
    ("on_a_body", (Womersley.field,), ThingKind.BODY, "only a field kind a medium carries"),
])
def test_a_field_kind_that_declares_nothing_it_provides_is_refused_at_registration(label, declared, applies,
                                                                                   problem):
    """The review's plant among them: Womersley's declaration with ``provides=()``. Each case's control is the same
    kind with Womersley's own declaration (a body's dynamics: with none), which registers."""
    with pytest.raises(TypeError, match=problem):
        register_dynamics(_kind(label, applies, declared))
    register_dynamics(_kind(f"{label}_control", applies, (Womersley.field,) if applies is ThingKind.MEDIUM else ()))
