"""The energy ledger's parts, each on its own: the vocabulary's port kinds (every one registered, each delivery's units
paired with its flow's to watts, a kind whose units do not pair refused), the core's delivery-kind and class strings
agreeing with the vocabulary's (``mime.core`` imports nothing above it, so it reads them as strings), the one pairing
rule per delivery kind, an encoded delivery's round trip, and what the ledger refuses: a node that writes no step
identity, a body held by a confinement whose ledger hosts no law for it. And the default: a body built without the
ledger takes the inputs and keeps the state and the parameters it always did."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ledger_cases import CONTACT_LAW, DRAG_LAW, LOAD_LAW, in_dtype, metered_body

from mime.core import MimeNode
from mime.core import ledger as books
from mime.nodes.body import RigidInertialTensorNode
from mime.vocabulary import (
    APPLIED_WRENCH,
    LINK_INERTIA,
    RESISTANCE,
    TWIST,
    Definiteness,
    DeliveryKind,
    EnergyClass,
    PortKind,
    port_kind,
    port_kinds,
    register_port_kind,
)


def test_the_built_in_port_kinds_are_registered_and_their_units_pair_to_watts():
    assert port_kinds() == ["delivered_inertia", "flow_in", "joint", "rigid_body"]
    assert all(port_kind(name) is not None and port_kind(name).problems() == [] for name in port_kinds())


def test_a_kind_whose_units_do_not_pair_is_refused():
    with pytest.raises(ValueError, match="do not pair"):
        PortKind("inertia_on_a_twist", TWIST.name, {LINK_INERTIA.name: DeliveryKind.LOAD})
    with pytest.raises(ValueError, match="not a state"):
        PortKind("a_load_for_a_flow", APPLIED_WRENCH.name, {})
    with pytest.raises(ValueError, match="does not hold"):
        PortKind("unknown", "body.nothing", {})
    held = port_kind("rigid_body")
    with pytest.raises(ValueError, match="already registered"):
        register_port_kind(PortKind("rigid_body", TWIST.name, {RESISTANCE.name: DeliveryKind.SOLVED}))
    assert register_port_kind(held) is held


def test_the_core_s_strings_are_the_vocabulary_s_values():
    assert books.DELIVERY_KINDS == tuple(kind.value for kind in DeliveryKind)
    assert books.ENERGY_CLASSES == tuple(cls.value for cls in EnergyClass)
    assert books.CHECK_DEFINITE == Definiteness.STATE.value


def test_the_one_pairing_rule_per_delivery_kind():
    load, change = jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([0.25, 0.0, -0.25])
    assert float(books.paired_work(books.LOAD, load, change=change)) == -0.5
    velocity = jnp.asarray([1.0, -1.0, 0.5])
    assert float(books.paired_work(books.SOLVED, -2.0 * velocity, solution=velocity, dt=0.5)) == -2.25
    assert float(books.paired_work(books.HOSTED, -2.0 * velocity, solution=velocity, dt=0.5)) == -2.25
    assert float(books.paired_work(books.FLOW_IN, load, returned=velocity, velocity=load, dt=2.0)) == 1.0
    assert float(books.paired_work(books.INERTIA, jnp.diag(jnp.asarray([1.0, 0.0, 0.0])), velocity=velocity)) == -0.5
    with pytest.raises(ValueError, match="no pairing"):
        books.paired_work(books.LOAD, load)


@given(law=st.text(st.characters(blacklist_characters="|", blacklist_categories=("Cs",)), min_size=1),
       kind=st.sampled_from([*books.DELIVERY_KINDS, books.HOSTED]),
       energy_class=st.sampled_from([*books.ENERGY_CLASSES, books.UNMETERED]),
       check=st.sampled_from(["", books.CHECK_DEFINITE]))
def test_an_encoded_delivery_reads_back_as_it_was(law, kind, energy_class, check):
    held = books.LedgerDelivery("applied_wrench", law, energy_class, kind, check, "field")
    assert books.LedgerDelivery.decoded(held.encoded()) == held


def test_a_delivery_with_a_kind_or_class_the_ledger_does_not_know_is_refused():
    for bad in ({"kind": "pushed"}, {"energy_class": "source"}, {"check": "always"}, {"law": "a|b"}):
        fields = {"field": "applied_wrench", "law": "gravity:umr", "energy_class": "potential", "kind": "load", **bad}
        with pytest.raises(ValueError):
            books.LedgerDelivery(**fields)


class Unmetered(MimeNode):
    """A node that takes any argument and writes no step identity."""

    def initial_state(self):
        return {}

    def boundary_input_spec(self):
        return {}


def test_a_node_that_writes_no_step_identity_is_refused_the_ledger():
    with pytest.raises(TypeError, match="no step identity"):
        Unmetered("toy", 5e-4, value=1.0, ledger=(LOAD_LAW.encoded(),))
    assert Unmetered("toy", 5e-4, value=1.0).constants() == {"value": 1.0}


def test_a_held_body_s_ledger_hosts_one_law_for_its_confinement():
    with pytest.raises(ValueError, match="hosted delivery"):
        metered_body(laws=(LOAD_LAW, DRAG_LAW, CONTACT_LAW))
    with pytest.raises(ValueError, match="hosted delivery"):
        RigidInertialTensorNode("umr", 5e-4, mass=1.0, inertia=tuple(np.eye(3).reshape(9)), initial_pose=(0.0,) * 7,
                                initial_twist=(0.0,) * 6, surface_shape="cylinder", extent=1e-3, margin=1e-4,
                                velocity_rule="remove_outward", surface_centre=(0.0, 0.0, 0.0),
                                surface_axis=(1.0, 0.0, 0.0), surface_radius=3e-3, ledger=(LOAD_LAW.encoded(),))


def test_a_body_checks_the_definiteness_of_its_resistance_alone():
    checked_load = books.LedgerDelivery("applied_wrench", "load:case", books.UNMETERED, books.LOAD,
                                        books.CHECK_DEFINITE)
    with pytest.raises(ValueError, match="definiteness"):
        metered_body(laws=(checked_load, DRAG_LAW))


def test_a_body_built_without_the_ledger_is_the_body_it_was():
    plain = RigidInertialTensorNode("umr", 5e-4, mass=1.0, inertia=tuple(np.eye(3).reshape(9)),
                                    initial_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), initial_twist=(0.0,) * 6)
    assert not plain.metered
    assert books.LEDGER not in plain.constants()
    assert sorted(plain.boundary_input_spec()) == ["applied_wrench", "background_load", "resistance"]
    assert sorted(plain.initial_state()) == ["pose", "twist"]
    metered = metered_body()
    assert metered.metered
    assert sorted(set(metered.boundary_input_spec()) - set(plain.boundary_input_spec())) == [
        books.law_input("applied_wrench", 0), books.law_input("resistance", 1)]
    assert all(key.startswith(books.RECORDS) for key in set(metered.initial_state()) - {"pose", "twist"})


@pytest.mark.parametrize("name", ["float32", "float64"])
@pytest.mark.parametrize("coupling", [1e-9, 1e-8, 1e-7, 1e-12])
def test_a_zero_diagonal_with_a_coupled_row_is_indefinite_at_any_size_of_the_coupling(name, coupling):
    """The review's matrix: FL-9's resistance with one turning entry exactly zero and a coupling to the force row, its
    least eigenvalue negative (``-coupling^2 / 4e-5``, from -2.5e-14 to -2.5e-10). A zero diagonal entry was scaled by
    1 where the others were scaled to unit diagonal, so in SI units its coupling fell below the shift (16 eps, 1.9e-6
    in float32) and the matrix passed in float32: it is now indefinite by the exact identity (a positive semidefinite
    matrix's zero diagonal entry has its whole row zero), not by a scale."""
    resistance = np.diag([4e-5] * 3 + [8e-10] * 3)
    resistance[3, 3] = 0.0
    resistance[0, 3] = resistance[3, 0] = coupling
    assert np.linalg.eigvalsh(resistance)[0] < 0.0
    with in_dtype(name) as dtype:
        assert not bool(books.definite_at_floor(jnp.asarray(resistance, dtype=dtype)))
        resistance[0, 3] = resistance[3, 0] = 0.0
        assert bool(books.definite_at_floor(jnp.asarray(resistance, dtype=dtype))), "its row zero, it is semidefinite"


@pytest.mark.parametrize("name", ["float32", "float64"])
def test_two_zero_diagonal_entries_coupled_to_each_other_alone_are_indefinite(name):
    resistance = np.diag([4e-5, 4e-5, 4e-5, 0.0, 0.0, 8e-10])
    resistance[3, 4] = resistance[4, 3] = 1e-12
    with in_dtype(name) as dtype:
        assert not bool(books.definite_at_floor(jnp.asarray(resistance, dtype=dtype)))
