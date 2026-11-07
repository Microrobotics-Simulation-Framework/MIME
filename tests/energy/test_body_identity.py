"""The rigid body's step identity with the energy ledger attached (F0-1 M6.2): ``rigid_inertial`` 0.4.0's node, metered
for a load law and a resistance law, one step from each drawn state, books ``H' - H = sum W - N`` within 16 eps of the
step's scale, the magnitude of the arithmetic its terms are formed by (``mime.core.ledger.IDENTITY_ULPS``). Three
arms: the free body, its resistance FL-9's or up to 10^6 times it (a stiff solve); its centre held on script A's tube
wall, moving outward across it (the hold in the solve, the projection and the velocity rule after it, the contact's
terms under its name); and a rod held by its hull's points at the wall, its projection's turn joining the
configuration's change. Both dtypes, drawn in the dtype the node computes in.

**The control**, on every draw: a step that keeps 0.999 of its twist without booking it, a loss the books do not see,
fails the identity in both dtypes (``tests/energy/predictions.json``: by more than 20 times the floor in float32). It
is drawn on the free body at FL-9's own resistance, where what the step keeps is most of what it had: where a stiff
solve, or a velocity rule at arrival, has taken most of the step's energy, a share of what is left falls below the
step's rounding, which is the floor's purpose (on the centre-held arm Hypothesis found one 2.2 times the floor in
float32, its outward speed removed by the rule).

**The run's books**: over a 2000-step driven run of the free body, the balance (the residuals' sum) within 16 eps of
the run's summed step scale."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from ledger_cases import (
    centre_held_states,
    in_dtype,
    free_states,
    hull_held_states,
    identity_over_floor,
    law_inputs,
    metered_body,
    one_step,
)

from mime.core import ledger as books
from mime.nodes.body import RigidInertialTensorNode

#: The control's share of the twist the step keeps, unbooked.
KEPT = 0.999


class UnbookedLoss(RigidInertialTensorNode):
    """A plant: the step keeps 0.999 of its twist, and its books do not know it."""

    def stepped_with_parts(self, p, state, boundary_inputs, dt):
        ended, parts = super().stepped_with_parts(p, state, boundary_inputs, dt)
        return {**ended, "twist": KEPT * ended["twist"]}, parts


DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])


NODES: dict[tuple[type, str], RigidInertialTensorNode] = {}


def node(held: str = "", node_class: type[RigidInertialTensorNode] = RigidInertialTensorNode):
    if (node_class, held) not in NODES:
        NODES[node_class, held] = metered_body(node_class, held=held)
    return NODES[node_class, held]


@DTYPES
@settings(deadline=None)
@given(drawn=free_states(stiff=True))
def test_a_free_body_s_step_books_close_within_rounding(name, drawn):
    with in_dtype(name) as dtype:
        found = one_step(node(), drawn, dtype)
    assert identity_over_floor(found) <= 1.0, identity_over_floor(found)
    assert float(found["ledger_unattributed_worst"]) == 0.0


@DTYPES
@settings(deadline=None)
@given(drawn=centre_held_states())
def test_a_centre_held_body_s_step_books_close_with_the_contact_s_terms_under_its_name(name, drawn):
    with in_dtype(name) as dtype:
        found = one_step(node("centre"), drawn, dtype)
    assert identity_over_floor(found) <= 1.0, identity_over_floor(found)
    contact = books.laws(books.deliveries(node("centre").constants())).index("contact:case")
    assert float(found["ledger_terms"][2]) == 0.0, "the configuration's term goes to the hosted contact"
    rounding = books.IDENTITY_ULPS * float(jnp.finfo(name).eps) * float(found["ledger_scale"])
    assert float(found["ledger_hosted"][contact, 0]) >= -rounding, "the velocity rule only removes energy"


@DTYPES
@settings(deadline=None)
@given(drawn=hull_held_states())
def test_a_hull_held_body_s_step_books_close_with_the_projection_s_turn(name, drawn):
    with in_dtype(name) as dtype:
        found = one_step(node("hull"), drawn, dtype)
    assert identity_over_floor(found) <= 1.0, identity_over_floor(found)


@DTYPES
@settings(deadline=None)
@given(drawn=free_states())
def test_an_unbooked_loss_fails_the_identity_on_every_draw(name, drawn):
    with in_dtype(name) as dtype:
        found = one_step(node("", UnbookedLoss), drawn, dtype)
    margin = 20.0 if name == "float32" else 1e8
    assert identity_over_floor(found) > margin, identity_over_floor(found)


def driven_run(dtype: Any, steps: int = 2000) -> dict[str, jax.Array]:
    """The free body driven 2000 steps by a turning torque against FL-9's resistance, its records at the end."""
    body = node()
    dt = 5e-4
    p = {k: jnp.asarray(v, dtype=dtype) for k, v in body._step_constants(None).items()}
    resistance = jnp.asarray(np.diag([4e-5, 8e-5, 8e-5, 8e-10, 2.4e-9, 2.4e-9]), dtype=dtype)

    def one(state, k):
        phase = 2.0 * np.pi * 10.0 * k * dt
        zero = jnp.zeros_like(phase)
        load = jnp.stack([zero, 1e-7 * jnp.cos(phase), zero, 2e-9 * jnp.cos(phase), 2e-9 * jnp.sin(phase), zero])
        return body.advance(p, state, law_inputs(body, load, resistance, dtype), dt), None

    start = {k: jnp.asarray(v, dtype=dtype) for k, v in body.initial_state().items()}
    ended, _ = jax.jit(lambda s: jax.lax.scan(one, s, jnp.arange(steps, dtype=dtype)))(start)
    return ended


@DTYPES
def test_a_driven_run_s_books_balance_within_the_run_s_rounding(name):
    with in_dtype(name) as dtype:
        ended = driven_run(dtype)
        balance, floor = float(books.balance(ended)), float(books.balance_floor(ended))
        worked = float(jnp.sum(jnp.abs(ended["ledger_work"])))
    assert floor > 0.0
    assert abs(balance) <= 0.1 * floor, (balance, floor)
    assert float(ended["ledger_identity_worst"]) <= books.IDENTITY_ULPS * float(jnp.finfo(name).eps)
    assert worked > 0.0, "the run did work: the books had something to close"
