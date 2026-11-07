"""The ledger's controls, sized in the floor's own units (the fix round of the first review, M1 and M2): every planted
defect is a number of the step's floors, 16 eps of the step's scale taken from the unplanted step, so that a plant
clears its floor on every draw by construction, in float32 as in float64, on every arm: the free body, its centre held
on script A's tube wall, and the rod held by its hull (the contact 0.5.0's loops, the build's default; the plants act
on the books, which the contact 0.4.0's unrolled program shares, and compiling that program as well costs more than the
test's budget: ``test_body_identity`` runs both programs' identity, and the counts taken for this round on the unrolled
program, 200 draws a cell, are in ``predictions.json``).

**What each plant is.** A gift of 64 floors (the end of the step's storage raised by 64 floors by scaling the ended
angular twist, unbooked) must read as 64 floors on the identity, on every draw. The same for a term of the books
misbooked by 64 floors, either sign, on the contact's loss and its projection's shift (separately) and on each of the
scheme's three terms. And **a contact that gives energy**: its net work over the step made 64 floors, positive, the
identity closed (the gift booked under the contact's name, as an honest account of a dishonest law), caught by the
law's class alone: the dissipative law's sign, ``ledger_gift_steps``. Its physical form, the hold reversed (the
correction taken the wrong way, 100 times over), is caught by the same sign. An unplanted step counts no gift
(``test_body_identity``, every arm).

**What the books cannot see, stated.** A gift in the end's *translational* twist of a held body is absorbed by the
velocity rule's loss, which is the difference between the solve's speed and the step's end speed, so it reads as a
smaller loss (the identity does not move; the loss changes by the gift). The projection's shift is likewise defined by
the step's end position. Those are the contact family's oracles' to check; here the plant is the free body's.

**The formulae by value, in float64.** The scheme's terms halved: a plant of half a term is a number of floors that
depends on the draw, and in float32 the floor sits above half a term on a share of draws, so by value the formulae are
checked in float64 (where the least margin is far above the floor); by weight (a term entering the identity with unit
weight, to 64 floors) they are checked in float32 and float64 above."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from ledger_cases import (
    ANGULAR_GIFT,
    CONFIGURATION,
    CONTACT_GIFT,
    EULER,
    GYROSCOPIC,
    HALVED_TERMS,
    HOSTED_LOSS,
    HOSTED_SHIFT,
    LINEAR_GIFT,
    PLANTS,
    PlantedBody,
    REVERSED_HOLD,
    centre_held_states,
    free_states,
    hull_held_states,
    identity_over_floor,
    in_dtype,
    metered_body,
    one_step,
)

from mime.core import ledger as books
from mime.nodes.body import RigidInertialTensorNode

#: The size of every plant, in floors.
SIZE = 64.0
#: The unplanted identity is within one floor (``test_body_identity``); the plant reads as its size within that.
SLACK = 1.0
#: How many times over the reversed hold takes the correction the wrong way.
REVERSAL = 100.0

ARMS = {
    "free": ("", False, free_states(stiff=True)),
    "centre": ("centre", False, centre_held_states()),
    "hull_looped": ("hull", True, hull_held_states()),
}
HELD = ["centre", "hull_looped"]
EVERY_ARM = pytest.mark.parametrize("arm", list(ARMS))
HELD_ARM = pytest.mark.parametrize("arm", HELD)
DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])


BODIES: dict[tuple[type, str, bool], RigidInertialTensorNode] = {}


def body(arm: str, node_class: type[RigidInertialTensorNode] = PlantedBody) -> RigidInertialTensorNode:
    held, loops, _ = ARMS[arm]
    if (node_class, held, loops) not in BODIES:
        BODIES[node_class, held, loops] = metered_body(node_class, held=held, loops=loops)
    return BODIES[node_class, held, loops]


def contact_row(node: RigidInertialTensorNode) -> int:
    return books.laws(books.deliveries(node.constants())).index("contact:case")


def floors(found: dict[str, jax.Array], quantity: jax.Array) -> float:
    """``quantity`` in units of the step's floor."""
    return float(quantity) / (books.IDENTITY_ULPS * float(jnp.finfo(found["ledger_scale"].dtype).eps)
                              * float(found["ledger_scale"]))


@DTYPES
@EVERY_ARM
def test_a_plant_of_zero_is_the_unplanted_step_bit_for_bit(arm, name):
    """The instrument is inert at zero: the planted node with no plant, against the plain node, every record."""
    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        with in_dtype(name) as dtype:
            planted = one_step(body(arm), drawn, dtype)
            plain = one_step(body(arm, RigidInertialTensorNode), drawn, dtype)
        for key in plain:
            assert bool(jnp.array_equal(plain[key], planted[key], equal_nan=True)), key

    check()


@DTYPES
@EVERY_ARM
def test_a_gift_and_each_misbooked_term_of_64_floors_read_as_64_floors_on_every_draw(arm, name):
    kinds = [ANGULAR_GIFT, EULER, GYROSCOPIC, CONFIGURATION]
    if arm == "free":
        kinds.append(LINEAR_GIFT)
    else:
        kinds += [HOSTED_LOSS, HOSTED_SHIFT]

    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        with in_dtype(name) as dtype:
            for kind in kinds:
                for size in (SIZE,) if kind in (ANGULAR_GIFT, LINEAR_GIFT) else (SIZE, -SIZE):
                    margin = identity_over_floor(one_step(body(arm), drawn, dtype, plant={kind: size}))
                    assert abs(margin - SIZE) <= SLACK, (PLANTS[kind], size, margin)

    check()


@DTYPES
@HELD_ARM
def test_a_contact_that_gives_energy_is_caught_by_its_class_alone_on_every_draw(arm, name):
    """Net work of +64 floors under the contact's name, the identity closed (below one floor), the dissipative law's
    sign read on its own row; and the unplanted step, whatever the contact dissipated in it, counts none."""
    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        node = body(arm)
        row = contact_row(node)
        with in_dtype(name) as dtype:
            clean = one_step(node, drawn, dtype)
            gift = one_step(node, drawn, dtype, plant={CONTACT_GIFT: SIZE})
        assert float(clean["ledger_gift_steps"][row]) == 0.0, floors(clean, clean["ledger_step_work"][row])
        assert identity_over_floor(gift) <= 1.0, identity_over_floor(gift)
        # In the unplanted step's floors: a gift of thousands of floors (a contact that had dissipated that much)
        # raises the planted step's scale, and with it the floor, by a share of a percent.
        assert abs(floors(clean, gift["ledger_step_work"][row]) - SIZE) <= SLACK
        assert float(gift["ledger_gift_steps"][row]) == 1.0, floors(gift, gift["ledger_step_work"][row])

    check()


@DTYPES
@HELD_ARM
def test_a_hold_reversed_is_caught_by_the_contact_s_sign_wherever_the_hold_acts(arm, name):
    """The physical form: the correction taken the wrong way (``REVERSAL`` times over), the identity closed (the work
    booked under the contact's name), the contact's work positive beyond the floor, on every draw on which the hold
    acts. Where it does not (the rod's nearest point, carried by its turning, is not moving into the wall: the hull
    contact has nothing to hold) the reversed step is the unplanted step bit for bit and there is nothing planted; the
    share of draws on which the hold acts is counted and must be at least half."""
    acting = []

    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        node = body(arm)
        row = contact_row(node)
        with in_dtype(name) as dtype:
            found = one_step(node, drawn, dtype, plant={REVERSED_HOLD: REVERSAL})
            clean = one_step(node, drawn, dtype)
        holds = not bool(jnp.array_equal(found["twist"], clean["twist"]))
        acting.append(holds)
        if holds:
            assert identity_over_floor(found) <= 1.0, identity_over_floor(found)
            assert float(found["ledger_gift_steps"][row]) == 1.0, floors(found, found["ledger_step_work"][row])
        else:
            assert float(found["ledger_gift_steps"][row]) == 0.0

    check()
    assert sum(acting) >= 0.5 * len(acting), (sum(acting), len(acting))


@HELD_ARM
@pytest.mark.parametrize("name", ["float64"])
def test_a_gift_in_a_held_body_s_translational_twist_is_booked_as_a_smaller_loss(arm, name):
    """What the books cannot see, as a test: the identity does not move for a translational gift of 64 floors, and
    the contact's loss reads that much less (or more negative): the end speed defines it."""
    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        node = body(arm)
        row = contact_row(node)
        with in_dtype(name) as dtype:
            clean = one_step(node, drawn, dtype)
            gift = one_step(node, drawn, dtype, plant={LINEAR_GIFT: SIZE})
        assert identity_over_floor(gift) <= 1.0, identity_over_floor(gift)
        change = floors(gift, gift["ledger_hosted"][row, 0] - clean["ledger_hosted"][row, 0])
        assert abs(change + SIZE) <= SLACK, change

    check()


@EVERY_ARM
@pytest.mark.parametrize("name", ["float64"])
def test_the_scheme_s_terms_halved_are_caught_by_value_in_float64(arm, name):
    @settings(deadline=None)
    @given(drawn=ARMS[arm][2])
    def check(drawn):
        with in_dtype(name) as dtype:
            found = one_step(body(arm), drawn, dtype, plant={HALVED_TERMS: 1.0})
        assert identity_over_floor(found) > 1e3, identity_over_floor(found)

    check()
