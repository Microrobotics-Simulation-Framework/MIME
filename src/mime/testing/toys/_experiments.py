"""Toy experiments as lists of declarations, binding by name so that any order of the calls is a valid script."""

from __future__ import annotations

from collections.abc import Sequence

from mime.experiment import Experiment
from mime.testing._graph import Declare, declare_in_order
from mime.testing.toys._entries import (
    ToyBody,
    ToyCouple,
    ToyDrag,
    ToyField,
    ToyHold,
    ToyLift,
    ToyRecorder,
    ToyRelaxation,
    ToySpring,
)
from mime.things import Apparatus, Body, Medium, State
from mime.units import m, m_per_s, s, standard_gravity

DT = s(0.02)
PROBE_START = State(position=m((0.3, -0.2, 0.1)), velocity=m_per_s((0.5, 0.1, -0.3)))


def toy_experiment(name: str = "toy") -> Experiment:
    return Experiment(name, up=(0.0, 0.0, 1.0), gravity=standard_gravity(), dt=DT)


ADD_BATH = Declare("add_medium(bath)", lambda e: e.add_medium(Medium("bath", flow=ToyRelaxation())))
ADD_PROBE = Declare("add_body(probe)",
                    lambda e: e.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START)))
ATTACH_DRAG = Declare("attach(drag)", lambda e: e.attach(ToyDrag.Quadratic(), body="probe"))
ATTACH_SPRING = Declare("attach(spring)", lambda e: e.attach(ToySpring(), body="probe"))


def coupled_declarations() -> tuple[Declare, ...]:
    """A medium and a body coupled both ways, and three loads on the body (a spring, a drag and a lift from the
    medium), with no coupling group: every loop is staggered, so the order nodes and edges are added in is part of
    the model."""
    return (
        ADD_BATH,
        ADD_PROBE,
        Declare("attach(couple)", lambda e: e.attach(ToyCouple(), body="probe", medium="bath")),
        ATTACH_DRAG,
        Declare("attach(lift)", lambda e: e.attach(ToyLift(), body="probe", medium="bath")),
        ATTACH_SPRING,
    )


def full_declarations() -> tuple[Declare, ...]:
    """Every part of the build at once: the coupled toy, plus a field provider with a reader slot, an exclusive
    target, two overlapping solve-together requests merged into one coupling group, and an observation at half the
    base rate."""
    return (
        Declare("add_apparatus(source)",
                lambda e: e.add_apparatus(Apparatus("source", fields={"push": ToyField()}))),
        ADD_BATH,
        ADD_PROBE,
        Declare("attach(couple, together)",
                lambda e: e.attach(ToyCouple(solve_together=True), body="probe", medium="bath")),
        ATTACH_DRAG,
        Declare("attach(lift, together)",
                lambda e: e.attach(ToyLift(solve_together=True), body="probe", medium="bath")),
        ATTACH_SPRING,
        Declare("attach(hold)", lambda e: e.attach(ToyHold(), body="probe", field="source.push")),
        Declare("observe(recorder)", lambda e: e.observe(ToyRecorder(body="probe", rate=s(0.04)))),
    )


def coupled_toy(order: Sequence[int] | None = None) -> Experiment:
    return declare_in_order(lambda: toy_experiment("coupled_toy"), coupled_declarations(), order)


def full_toy(order: Sequence[int] | None = None) -> Experiment:
    return declare_in_order(lambda: toy_experiment("full_toy"), full_declarations(), order)
