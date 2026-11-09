"""The registry walk's energy half (F0-1 M3, M6): every dynamics and every interaction of MIME's own (its toys
included) declares what the energy ledger meters it by, a port kind (``Dynamics.port``) or a class
(``Interaction.energy_class``), or states why it is unmetered (``unmetered``); and every dynamics that names a port
kind passes its step identity on drawn states (``mime.testing.step_identity``), every version kept. An entry
registered with neither is caught by name. An entry outside MIME may declare nothing: the ledger then logs it under
exploratory and refuses it under research (``ledger.unmetered``), here with the lab fixture's nudge and a planted law.

**The step identity's control**: a receiver whose step raises its end storage by 64 floors of its own (the step's
ended angular twist scaled, unbooked; ``ledger_cases.PlantedBody``) fails the case: P1a-1's planted counts read such a
gift as 64 floors within one on every draw of every arm, both dtypes. **The port kind is consumed**: the body pairs
each of its inputs as the rigid body's port kind says (a load with the configuration's change, the resistance with
the step's solution), and refuses a delivery paired otherwise."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import jax.numpy as jnp
import pytest
from ledger_cases import ANGULAR_GIFT, PLANTS, PlantedBody

from mime.build import BuildRefused
from mime.core import ledger
from mime.docs import load_namespaces
from mime.dynamics import RigidInertial
from mime.interactions import Gravity
from mime.nodes.body import RigidInertialNode
from mime.nodes.body._rigid_inertial import _PAIRED_AS
from mime.observations import EnergyLedger
from mime.registry import Dynamics, EntryKind, Interaction, lookup, register_interaction
from mime.testing import PORT_CASES, PORT_DRAWS, installed_entries, step_identity
from mime.testing._energy import dynamics_rig
from mime.vocabulary import RIGID_BODY, register_namespace

register_namespace("unstated", "0.0.1")


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class UnstatedLaw(Gravity):
    """Gravity that declares no class and states no reason, as an outside entry may."""

    name = "unstated.gravity"
    energy_class = None


def classes(key: tuple[str, str, str]) -> type:
    load_namespaces()
    record = lookup(EntryKind(key[0]), key[1], key[2])
    assert record is not None, key
    return record.cls


def undeclared(keys: list[tuple[str, str, str]]) -> list[str]:
    """Each of MIME's own dynamics and interactions among ``keys`` that declares neither what the ledger meters it by
    nor why it is unmetered."""
    found: list[str] = []
    for key in keys:
        if key[0] not in ("dynamics", "interaction"):
            continue
        cls = classes(key)
        if not cls.__module__.startswith("mime."):
            continue
        declared = cls.port if issubclass(cls, Dynamics) else cls.energy_class
        if not declared and not cls.unmetered:
            found.append(" ".join(key))
    return found


@pytest.fixture(scope="module")
def registered():
    found = installed_entries()
    assert ("dynamics", "rigid_inertial", "0.4.0") in found, "the walk would pass over an empty registry"
    return found


def test_every_dynamics_and_interaction_of_mime_says_how_it_is_metered_or_why_not(registered):
    assert not undeclared(registered)


def test_an_entry_of_mime_s_that_says_neither_is_caught_by_name(registered, monkeypatch):
    cls = classes(("interaction", "toy.spring", "0.1.0"))
    monkeypatch.setattr(cls, "unmetered", "")
    assert undeclared(registered) == ["interaction toy.spring 0.1.0"]


METERED = sorted(key for key in installed_entries() if key[0] == "dynamics" and classes(key).port)


@pytest.mark.parametrize("key", METERED, ids=" ".join)
def test_every_metered_dynamics_passes_its_step_identity(key):
    cls = classes(key)
    assert cls.port in PORT_DRAWS, "a new port kind brings its draws with it (mime.testing.PORT_DRAWS)"
    # A chain segment's and a flow's cases step their port kind's own rig, so they need no settings of the entry's.
    result = step_identity(cls.__new__(cls) if cls.port in PORT_CASES else cls())
    print(f"\n{result}")
    assert result.passed, result


class GiftedBody(PlantedBody):
    """A step that raises its end storage by 64 of its own floors, unbooked."""

    def advance(self, p: Any, state: Any, boundary_inputs: Any, dt: float) -> Any:
        if not self.metered:
            return super(PlantedBody, self).advance(p, state, boundary_inputs, dt)
        plant = jnp.zeros(len(PLANTS), dtype=jnp.asarray(p["mass"]).dtype).at[ANGULAR_GIFT].set(64.0)
        return super().advance({**p, "plant": plant}, state, boundary_inputs, dt)


def test_a_step_that_gives_64_floors_unbooked_fails_its_step_identity():
    node = dynamics_rig(RigidInertial())
    assert not isinstance(node, str), node
    held: Any = node
    gifted = GiftedBody(held.name, float(held.delta_t), **dict(held.params))
    result = step_identity(RigidInertial(), gifted)
    print(f"\n{result}")
    assert result.failed and "of its floor" in result.detail


def misbooking(base: type) -> type:
    """``base`` (a body's node) with a step that books the work of its unattributed input under its first law, as an
    honest-looking account of a hidden delivery: the identity closes and nothing is unattributed."""

    class MisbookedBody(base):  # type: ignore[misc, valid-type]
        def identity_terms(self, p: Any, held: Any, state: Any, boundary_inputs: Any, ended: Any, parts: Any,
                           dt: float) -> Any:
            account = super().identity_terms(p, held, state, boundary_inputs, ended, parts, dt)
            return dataclasses.replace(account, work=account.work.at[0].add(account.unattributed),
                                       unattributed=jnp.zeros_like(account.unattributed))

    return MisbookedBody


def test_work_booked_under_a_law_that_was_fed_to_the_unattributed_input_fails_its_step_identity_on_every_draw():
    """The assertion that unattributed work is booked as such can fail: the plant closes the identity (the work is in
    the books, under a law) and leaves nothing unattributed, so only that assertion sees it. It reads zero on every
    draw (Hypothesis stops at the first and replays it), and every draw feeds a load whose paired work is far above
    zero, so it is caught on every run."""
    node = dynamics_rig(RigidInertial())
    assert not isinstance(node, str), node
    held: Any = node
    misbooked = misbooking(type(held))(held.name, float(held.delta_t), **dict(held.params))
    for _ in range(3):
        result = step_identity(RigidInertial(), misbooked, max_examples=20)
        print(f"\n{result}")
        assert result.failed and "was not booked as unattributed work" in result.detail
        assert result.n_examples <= 2, "Hypothesis stops at the first draw, which fails it, and replays it once"
    assert step_identity(RigidInertial(), node, max_examples=20).passed


def test_a_dynamics_with_no_port_kind_passes_by_its_reason_and_fails_without_one(monkeypatch):
    toy = classes(("dynamics", "toy.body", "0.1.0"))
    assert not toy.port and toy.unmetered
    assert step_identity(toy.__new__(toy)).passed
    monkeypatch.setattr(toy, "unmetered", "")
    assert step_identity(toy.__new__(toy)).failed


def test_the_body_pairs_each_input_as_the_rigid_body_s_port_kind_says_and_refuses_another_pairing():
    accepts = RigidInertial.accepts
    assert {accepts[name].field: kind.value for name, kind in RIGID_BODY.deliveries.items()
            if name in accepts and accepts[name].field in _PAIRED_AS} == _PAIRED_AS
    node = dynamics_rig(RigidInertial())
    assert not isinstance(node, str), node
    held: Any = node
    wrong = ledger.LedgerDelivery("resistance", "drag:case", ledger.DISSIPATIVE, ledger.LOAD)
    with pytest.raises(ValueError, match="paired otherwise"):
        RigidInertialNode(held.name, float(held.delta_t), **{**dict(held.params), "ledger": (wrong.encoded(),)})


def unmetered_law(profile: str):
    """A body in water in a tube under ``profile``, with nothing on it but the planted law that says nothing."""
    from mime import Body, Experiment, Medium, State, Structure
    from mime.scene import Scene, Tube, X, Z
    from mime.units import kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, ms, rad_per_s

    exp = Experiment("unmetered", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1.0),
                     profile=profile)
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                             length=mm(100.0))))
    exp.add_medium(Medium("water", fills=tube.lumen))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("umr.mass", kg(8.5e-5)), ("umr.inertia", kg_m2((1e-10,) * 3)),
                        ("umr.buoyant_volume", m3(6e-8)), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
                        ("water.density", kg_per_m3(1000.0))):
        exp.declare(path, value, reason="the test's body and water")
    exp.attach(UnstatedLaw(), body="umr", medium="water")
    exp.observe(EnergyLedger())
    return exp


#: The line a law gets that declares no class, states no reason, and whose own node reads nothing of the components:
#: a source by the graph, outside and supplying its own work, its missing declaration logged all the same.
SAYS_NOTHING = ("{}: declares no energy class and states no reason; its own nodes read nothing of the components, so it "
                "is outside and supplies its own work")


def test_an_outside_law_that_says_nothing_is_logged_under_exploratory_and_refused_under_research():
    built = unmetered_law("exploratory").build()
    assert SAYS_NOTHING.format("unstated.gravity:umr") in built.report.energy
    with pytest.raises(BuildRefused) as refused:
        unmetered_law("research").build()
    assert refused.value.codes() == ["ledger.unmetered"], refused.value


def test_the_lab_fixture_s_nudge_on_a_toy_is_logged_with_the_toy_s_reason():
    lab = importlib.import_module("mime_lab_fixture")
    from mime.testing.toys import coupled_toy

    experiment = coupled_toy()
    experiment.attach(lab.Nudge(), body="probe")
    experiment.observe(EnergyLedger())
    lines = experiment.build().report.energy
    assert any(line.startswith("probe: receives deliveries") and "a toy with no physics" in line for line in lines)
    assert SAYS_NOTHING.format("labx.nudge:probe") in lines
