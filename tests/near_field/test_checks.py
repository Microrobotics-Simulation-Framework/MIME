"""Gate 7, the coupled resistances' checks, per the maintainer's interim (no floor before sitting 4): each body's coupled
6 x 6 and the cluster's grand resistance are judged by their least eigenvalue, each row and column scaled by its
diagonal, and where one is not above zero every delivery is NaN and the run stops (dev1's ``stops_where``), the margin
reported each step. Two plants, each measured to stop every configuration it is planted in:

- a body's own 6 x 6 made indefinite (its block shifted down by twice its largest eigenvalue): the member's check;
- every cross block doubled where the pair's lagged gain is above one half, so that the cluster's grand resistance is
  indefinite while every body's own 6 x 6 is untouched and definite: the cluster's check, which the members' cannot see
  (the coordinator's planted cross-block error, judged in the cluster's total).

The unplanted configurations stop none. Pairs at gaps of 0.05 to 0.3 radii, each turned at random, coarse spheres."""

import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, joint_node, mrs, precision, rebuilt, spheres, start_inputs

from mime.build._maddening import parameter_leaves
from mime.nodes.hydro.near_field import JointDragNode
from mime.nodes.hydro.near_field._node import WHY_CLUSTER, WHY_MEMBER

GAPS = (0.05, 0.1, 0.2, 0.3)
SPACING = 0.4e-3


class OwnIndefinite(JointDragNode):
    """The plant: the first body's own block shifted down by twice its largest eigenvalue."""

    def solved(self, p, poses, steps=None):
        found, *rest = super().solved(p, poses, steps)
        block = found[:6, :6]
        shift = 2.0 * jnp.max(jnp.abs(jnp.linalg.eigvalsh(0.5 * (block + block.T))))
        return (found.at[:6, :6].add(-shift * jnp.eye(6, dtype=found.dtype)), *rest)


class CrossDoubled(JointDragNode):
    """The plant: every cross block doubled, every body's own block as it was."""

    def solved(self, p, poses, steps=None):
        found, *rest = super().solved(p, poses, steps)
        count = len(self.members)
        mask = jnp.kron(jnp.eye(count, dtype=found.dtype), jnp.ones((6, 6), dtype=found.dtype))
        return (found * mask + 2.0 * found * (1.0 - mask), *rest)


def configuration(gap: float, seed: int, **settings):
    rng = np.random.default_rng(seed)
    turns = {f"s{i}": tuple(q / np.linalg.norm(q)) for i, q in enumerate(rng.normal(size=(2, 4)))}
    return bodies("checked", spheres(gap * 1e-3, 2), mrs(SPACING, **settings), orientations=turns).build()


def step_with(built, cls):
    node = joint_node(built)
    planted = cls(node.name, float(node.delta_t), **node.constants()) if cls is not None else node
    out = planted.update(planted.initial_state(), start_inputs(built), planted.delta_t,
                         params=parameter_leaves(built.graph)[NODE])
    return planted, {key: np.asarray(value) for key, value in out.items()}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_each_plant_stops_every_configuration_and_none_stops_unplanted(dtype):
    rows = []
    with precision(dtype):
        for gap in GAPS:
            for seed in range(2):
                built = configuration(gap, seed)
                _, clean = step_with(built, None)
                own_node, own = step_with(built, OwnIndefinite)
                cross_node, cross = step_with(built, CrossDoubled)
                rows.append((gap, seed, clean, own, cross, own_node, cross_node))
    for gap, seed, clean, own, cross, own_node, cross_node in rows:
        print(f"\n{dtype} gap {gap} a, turn {seed}: margins {np.round(clean['least_eigenvalue'], 3)} cluster "
              f"{float(clean['cluster_least']):.3g}; planted own {float(own['stopped_why'])}, planted cross "
              f"{float(cross['stopped_why'])} (its bodies' own margins {np.round(cross['least_eigenvalue'], 3)})")
        assert float(clean["stopped_at"]) == 0.0 and float(clean["indefinite"]) == 0.0
        assert float(own["stopped_why"]) == WHY_MEMBER and np.all(np.isnan(own["resistance_0"]))
        assert own_node.stop_problem(own).code == "drag.near-field-indefinite"
        assert float(cross["stopped_why"]) == WHY_CLUSTER and np.all(np.isnan(cross["background_load_1"]))
        assert np.all(cross["least_eigenvalue"] > 0.0), "the members' own checks cannot see the cross plant"
        assert "cluster's grand resistance" in cross_node.stop_problem(cross).message


def test_the_margin_is_reported_each_step_and_the_stop_recorded_once():
    with precision("float64"):
        built = configuration(0.1, 0)
        node = joint_node(built)
        planted = OwnIndefinite(node.name, float(node.delta_t), **node.constants())
        inputs = start_inputs(built)
        state = planted.initial_state()
        params = parameter_leaves(built.graph)[NODE]
        for _ in range(3):
            state = planted.update(state, inputs, planted.delta_t, params=params)
    assert float(state["steps"]) == 3.0 and float(state["stopped_at"]) == 1.0
    assert np.asarray(state["least_eigenvalue"]).shape == (2,)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_on_the_iterative_route_each_plant_stops_and_the_unplanted_steps_are_the_dense_routes(dtype):
    """Gate 7 past the takeover line (gate 11's dtype arm): the same plants stop every configuration, none stops
    unplanted, and the unplanted step's margins are the dense route's to the solve's tolerance."""
    rows = []
    with precision(dtype):
        for gap in (0.05, 0.3):
            dense = step_with(configuration(gap, 0), None)[1]
            built = configuration(gap, 0, near_field_dense_unknowns=0)
            rows.append((gap, dense, step_with(built, None)[1], step_with(built, OwnIndefinite),
                         step_with(built, CrossDoubled)))
    for gap, dense, clean, (own_node, own), (cross_node, cross) in rows:
        moved = float(np.max(np.abs(clean["least_eigenvalue"] - dense["least_eigenvalue"])))
        print(f"\n{dtype} gap {gap} a, GMRES: margins {np.round(clean['least_eigenvalue'], 4)}, the dense route's moved "
              f"by {moved:.2g}; planted own {float(own['stopped_why'])}, planted cross {float(cross['stopped_why'])}")
        assert float(clean["stopped_at"]) == 0.0 and float(clean["solve_flagged"]) == 0.0
        assert moved <= (1e-8 if dtype == "float64" else 1e-4)
        assert float(own["stopped_why"]) == WHY_MEMBER and own_node.stop_problem(own).code == "drag.near-field-indefinite"
        assert float(cross["stopped_why"]) == WHY_CLUSTER and np.all(cross["least_eigenvalue"] > 0.0)
        assert "cluster's grand resistance" in cross_node.stop_problem(cross).message
