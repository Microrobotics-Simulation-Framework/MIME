"""The energy ledger beside a medium's near field (the coordinator's ruling on W3's proposal; the refuting review's M2):
each body's own coupled resistance is booked under its own drag law (dissipative, paired at its step's solution, its
sign judged on its row as any drag's), and the lagged cross loads under the near field's own law,
``near_field:<medium>``, booked by the one pairing rule on each body and never judged on one body's row
(``CHECK_CLUSTER``): one body drags another, so the cross load's paired work on the dragged body is positive with
nothing wrong, and the control below shows the row's own sign check firing on it.

**The cluster's judgement is on the books** (``JointDragNode.judged_books``): once the bodies have taken a step, the next
judges the near field's own work on them over it, from what it delivered and the twists they reached, paired as the
ledger pairs it; the fluid may not give the cluster energy past the ledger's floor, as the ledger's route 1 judges a
step, the gift steps counted (``cluster_gift_steps``) and read into the law's lines by the run's account. Held here:
the node's books equal the ledger's, step by step, both dtypes; an unplanted run has no gift step.

**Its reach** (measured, the dragged pair at h/a 0.1, 0.3, 1 and 2, along and across the line of centres, both
dtypes): cross loads delivered three times over make the fluid a source to a radius's gap along the line of centres and
a third of one across it, and are caught; twice over, to a third of a radius along it; those at wider gaps, at half
their value or with their sign flipped, or a grand resistance whose cross blocks are scaled and stay definite, leave
the fluid dissipating, and no energy check can see them (an error that makes the grand resistance indefinite stops
the run, gate 7). Those are the force oracles' to catch: the delivery's own check against the grand resistance (``test_build``), the
dense assembly (gate 2) and the two-sphere series and Rotne and Prager's mobility (gate 3). The tests below hold both
sides: the plants it catches, caught on every configuration, and the blind ones, unseen, so that a change to either is
seen."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs
from rigid_oracles import ConstantWrench

from mime.build._maddening import parameter_leaves
from mime.core import ledger
import mime.nodes.hydro.near_field._node as node_module
from mime.observations import EnergyLedger
from mime.results import energy_account
from mime.units import ms

SPACING = 0.6e-3
LAW = "near_field:water"


def dragged(metered: bool = True):
    """s0 pulled along x by a constant force, s1 free beside it: s0 drags s1 along through the fluid."""
    experiment = bodies("dragged", spheres(0.3e-3, 2), mrs(SPACING), mass=1.9e-8, inertia=7.5e-15)
    experiment.attach(ConstantWrench(value=(2e-7, 0.0, 0.0, 0.0, 0.0, 0.0)), body="s0")
    if metered:
        experiment.observe(EnergyLedger())
    return experiment


def rows_of(built, body: str) -> dict[str, ledger.LedgerDelivery]:
    held = ledger.deliveries(dict(built.graph.get_node(body).params))
    return {d.law: d for d in held}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_near_fields_cross_loads_are_metered_under_its_own_law_and_judged_in_the_clusters_sum(dtype):
    with precision(dtype):
        built = dragged().build()
        result = built.run(ms(8.0), history=True)
    for body in ("s0", "s1"):
        rows = rows_of(built, body)
        assert rows[f"drag:{body}"].field == "resistance" and rows[f"drag:{body}"].check == ledger.CHECK_DEFINITE
        assert rows[LAW].field == "background_load" and rows[LAW].check == ledger.CHECK_CLUSTER
        assert rows[LAW].energy_class == ledger.DISSIPATIVE
    laws = ledger.laws(ledger.deliveries(dict(built.graph.get_node("s1").params)))
    work = np.asarray(result.history["s1"][f"{ledger.RECORDS}step_work"])[:, laws.index(LAW)]
    final = result.final["s1"]
    print(f"\n{dtype}: the cross load's paired work on the dragged body, per step {np.round(work * 1e12, 4)} pJ; its "
          f"row's gift steps {float(np.asarray(final[f'{ledger.RECORDS}gift_steps'])[laws.index(LAW)])}, the cluster's "
          f"{float(np.asarray(result.final[NODE]['cluster_gift_steps']))}; unattributed "
          f"{float(np.asarray(final[f'{ledger.RECORDS}unattributed']))}")
    assert np.any(work > 0.0), "the dragged body takes energy from the cross load: no sign on its row alone"
    assert float(np.asarray(final[f"{ledger.RECORDS}gift_steps"])[laws.index(LAW)]) == 0.0
    assert float(np.asarray(result.final[NODE]["cluster_gift_steps"])) == 0.0
    for body in ("s0", "s1"):
        assert float(np.asarray(result.final[body][f"{ledger.RECORDS}unattributed"])) == 0.0
    account = result.energy
    assert account is not None
    lines = [line for line in account.laws if line.law == LAW]
    assert len(lines) == 2 and all(line.gift_steps == 0 for line in lines)


def books_against_the_ledger(built, result):
    """The node's books of each step (judged at the step after, ``-cluster_dissipation``), the ledger's (each body's own
    drag row and its near-field row, summed over both bodies) and their scale (the rows' magnitudes summed); the
    rounding by which the two pairings of a load differ (the ledger pairs it with the configuration's change, the
    positions' difference, rounded to the positions' scale, the node with ``dt V'``): ``|F| . |x'|`` over the bodies;
    and the control, the books paired at each step's starting twists in place of the twists it reached."""
    history = result.history
    ledger_books, scale, positions = 0.0, 0.0, 0.0
    for i, body in enumerate(("s0", "s1")):
        laws = ledger.laws(ledger.deliveries(dict(built.graph.get_node(body).params)))
        work = np.asarray(history[body][f"{ledger.RECORDS}step_work"], dtype=np.float64)
        rows = work[:, [laws.index(f"drag:{body}"), laws.index(LAW)]]
        ledger_books, scale = ledger_books + rows.sum(axis=1), scale + np.abs(rows).sum(axis=1)
        reached = np.abs(np.asarray(history[body]["pose"], dtype=np.float64)[:, :3])
        loads = np.abs(np.asarray(history[NODE][f"background_load_{i}"], dtype=np.float64)[:, :3])
        positions = positions + np.sum(loads * reached, axis=1)
    node_books = -np.asarray(history[NODE]["cluster_dissipation"], dtype=np.float64)[1:]
    dt = float(joint_node(built).delta_t)
    start = np.asarray(history[NODE]["start_twists"], dtype=np.float64)
    at_start = sum(dt * (np.einsum("ti,ti->t", np.asarray(history[NODE][f"background_load_{i}"]), start[:, i])
                         - np.einsum("ti,tij,tj->t", start[:, i], np.asarray(history[NODE][f"resistance_{i}"]),
                                     start[:, i])) for i in range(2))
    return node_books, ledger_books[:-1], scale[:-1], positions[:-1], at_start[:-1]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_nodes_books_are_the_ledgers_step_by_step_and_paired_at_the_starting_twists_they_are_not(dtype):
    """What the cluster's judgement reads is what the ledger booked: the node's books of each step, from what it
    delivered and the twists the bodies reached, equal the sum of both bodies' drag and near-field rows to 64 units of
    eps on the rows' scale and on the positions' rounding of the configuration's change (the two pairings of a load).
    The control: the same books paired at each step's starting twists (the operator's velocity, the judgement this
    replaced) differ from the ledger's by more than a hundred times that line at some step."""
    with precision(dtype):
        built = dragged().build()
        result = built.run(ms(8.0), history=True)
    node_books, ledger_books, scale, positions, at_start = books_against_the_ledger(built, result)
    line = 64.0 * np.finfo(np.dtype(dtype)).eps * (scale + positions)
    found, control = np.abs(node_books - ledger_books) / line, np.abs(at_start - ledger_books) / line
    print(f"\n{dtype}: the node's books against the ledger's, over the line, per step {np.round(found, 3)}; paired at "
          f"the starting twists {np.round(control, 1)}")
    assert np.all(found <= 1.0)
    assert np.max(control) > 100.0


def test_control_judged_on_its_row_the_dragged_bodys_cross_load_would_count_false_gifts():
    """The row's own sign check applied to the cross loads, from the run's history (each step's paired work against
    the ledger's floor, 16 eps of the step's scale, as ``recorded`` judges a dissipative row): the dragged body's row
    would count gift steps, though nothing is wrong. What the cluster's judgement is for."""
    with precision("float64"):
        built = dragged().build()
        result = built.run(ms(8.0), history=True)
    laws = ledger.laws(ledger.deliveries(dict(built.graph.get_node("s1").params)))
    history = result.history["s1"]
    work = np.asarray(history[f"{ledger.RECORDS}step_work"])[:, laws.index(LAW)]
    scale = np.diff(np.concatenate([[0.0], np.asarray(history[f"{ledger.RECORDS}scale"])]))
    gifts = int(np.sum(work > ledger.IDENTITY_ULPS * np.finfo(np.float64).eps * scale))
    print(f"\nthe cross loads judged on the dragged body's row: {gifts} gift steps of {len(work)}")
    assert gifts > 0


def run_planted(gap: float, along: bool, factor: float, dtype: str):
    """The dragged pair ``gap`` radii apart, the force along (or across) the line of centres, each cross load the node
    delivers scaled by ``factor`` (the grand resistance untouched), 8 steps metered; with how often the plant was
    traced (a plant that is never traced would test nothing: the node's step is compiled once per instance)."""
    calls = {"n": 0}
    real = node_module.cross_loads

    def planted(off, twists):
        calls["n"] += 1
        return factor * real(off, twists)

    jax.clear_caches()
    node_module.cross_loads = planted
    try:
        with precision(dtype):
            force = (2e-7, 0.0, 0.0) if along else (0.0, 2e-7, 0.0)
            experiment = bodies("planted", spheres(gap * 1e-3, 2), mrs(SPACING), mass=1.9e-8, inertia=7.5e-15)
            experiment.attach(ConstantWrench(value=(*force, 0.0, 0.0, 0.0)), body="s0")
            experiment.observe(EnergyLedger())
            result = experiment.build().run(ms(8.0))
    finally:
        node_module.cross_loads = real
    return result.final[NODE], calls["n"]


#: (h/a, along the line of centres, the factor on the delivered cross loads): measured to make the fluid a source in
#: both dtypes (the review's own plant is the second, its booked total +8.8e-10 J against 8.9e-11 J of work).
CAUGHT = [(0.1, True, 3.0), (0.3, True, 3.0), (1.0, True, 3.0), (0.1, False, 3.0)]
#: Measured to leave the fluid dissipating: half the cross loads, their sign flipped, twice them at a radius's gap or
#: across the line of centres.
BLIND = [(0.1, True, 0.5), (0.1, True, -1.0), (1.0, True, 2.0), (0.1, False, 2.0)]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(("gap", "along", "factor"), CAUGHT)
def test_a_cross_load_that_makes_the_fluid_a_source_is_caught_by_the_clusters_books(gap, along, factor, dtype):
    final, traced = run_planted(gap, along, factor, dtype)
    gifts = float(np.asarray(final["cluster_gift_steps"]))
    print(f"\n{dtype}, h/a {gap}, the cross loads x{factor:g}: {gifts:g} gift steps of 7 judged (traced {traced})")
    assert traced >= 1 and gifts >= 1.0 and float(np.asarray(final["stopped_at"])) == 0.0


@pytest.mark.parametrize(("gap", "along", "factor"), BLIND)
def test_the_reach_a_cross_load_that_leaves_the_fluid_dissipating_is_not_seen(gap, along, factor):
    """The check's documented blind side, held so that a change to it is seen: these plants are the force oracles'."""
    final, traced = run_planted(gap, along, factor, "float64")
    gifts = float(np.asarray(final["cluster_gift_steps"]))
    print(f"\nh/a {gap}, the cross loads x{factor:g}: {gifts:g} gift steps (traced {traced})")
    assert traced >= 1 and gifts == 0.0


def test_the_account_reads_the_clusters_judgement_into_the_near_fields_lines():
    with precision("float64"):
        built = dragged().build()
        result = built.run(ms(2.0))
        final = {name: dict(fields) for name, fields in result.final.items()}
        final[NODE]["cluster_gift_steps"] = jnp.asarray(3.0)
        account = energy_account(built.graph, final, parameter_leaves(built.graph))
    assert account is not None
    assert [line.gift_steps for line in account.laws if line.law == LAW] == [3, 3]
    assert all(line.gift_steps == 0 for line in account.laws if line.law != LAW)
