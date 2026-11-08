"""The energy ledger beside a medium's near field (the coordinator's ruling on W3's proposal): each body's own coupled
resistance is booked under its own drag law (dissipative, paired at its step's solution, its sign judged on its row as
any drag's), and the lagged cross loads under the near field's own law, ``near_field:<medium>``, booked by the one
pairing rule on each body and judged only in the cluster's sum (``CHECK_CLUSTER``): the joint node judges the grand
resistance's quadratic form at the step's starting twists, at or above minus its rounding floor, and counts the steps
it is not (``cluster_gift_steps``), which the run's account reads into that law's lines.

Why not on the row: one body drags another, so the cross load's paired work on the dragged body is positive with
nothing wrong; the control below shows the row's own sign check firing on it, which is the false alarm the cluster's
judgement avoids. Why at one velocity: the lagged booked sum differs from the quadratic form by a first-order residual
(the cross loads paired at the others' starting twists), so a check on it would fire falsely as well. The plant: every
cross block doubled where the pair's gain is above one half, which leaves each body's own block (and its row's sign)
untouched and makes the quadratic form negative along the bodies' co-moving twist: the cluster's judgement catches it
on every configuration, in both dtypes."""

import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, spheres, start_inputs
from rigid_oracles import ConstantWrench

from mime.build._maddening import parameter_leaves
from mime.core import ledger
from mime.nodes.hydro.near_field import JointDragNode
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


class CrossDoubled(JointDragNode):
    """The plant: every cross block of the grand resistance doubled, each body's own block as it was."""

    def solved(self, p, poses):
        found, *rest = super().solved(p, poses)
        count = len(self.members)
        mask = jnp.kron(jnp.eye(count, dtype=found.dtype), jnp.ones((6, 6), dtype=found.dtype))
        return (found * mask + 2.0 * found * (1.0 - mask), *rest)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("gap", [0.05e-3, 0.1e-3, 0.2e-3, 0.3e-3])
def test_a_planted_cross_block_is_caught_by_the_clusters_judgement_on_every_configuration(gap, dtype):
    with precision(dtype):
        built = bodies("planted", spheres(gap, 2), mrs(SPACING), velocity=(1e-3, 0.0, 0.0)).build()
        node = joint_node(built)
        planted = CrossDoubled(node.name, float(node.delta_t), **node.constants())
        params = parameter_leaves(built.graph)[NODE]
        inputs = start_inputs(built)
        clean = node.update(node.initial_state(), inputs, node.delta_t, params=params)
        caught = planted.update(planted.initial_state(), inputs, planted.delta_t, params=params)
    print(f"\n{dtype}, gap {gap * 1e3:g} mm: the cluster's dissipation rate {float(clean['dissipation']):.4g} W clean, "
          f"{float(caught['dissipation']):.4g} W planted; gift {float(clean['cluster_gift'])} and "
          f"{float(caught['cluster_gift'])}; the bodies' own margins planted {np.round(np.asarray(caught['least_eigenvalue']), 3)}")
    assert float(clean["cluster_gift"]) == 0.0 and float(clean["cluster_gift_steps"]) == 0.0
    assert float(caught["cluster_gift"]) == 1.0 and float(caught["cluster_gift_steps"]) == 1.0
    assert np.all(np.asarray(caught["least_eigenvalue"]) > 0.0), "each body's own check cannot see it"


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
