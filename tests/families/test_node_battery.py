"""MADDENING's node verification as MIME runs it on every node: the ``verify_node`` battery, and
``verify_node_order`` on each axis a node declares an order on. A declared order is measured against a refinement
ladder its case gives; one with no ladder fails, and a wrong claim fails the measurement. Controls: the same explicit
Euler decay declared first order passes, declared second order fails, and declared with no ladder fails."""

import math

import jax.numpy as jnp
import pytest
from maddening.core.compliance.metadata import DiscretizationOrder
from maddening.core.node import SimulationNode
from maddening.testing.mms import RefinementAxis

from mime.testing import OrderLadder, node_results

HORIZON = 1.0


class EulerDecay(SimulationNode):
    """``dx/dt = -x`` by explicit Euler, first order in time; it declares the order it is built with."""

    def __init__(self, name, timestep, declared=1.0):
        super().__init__(name, timestep, declared=declared)

    def discretization_order(self):
        return DiscretizationOrder(temporal=float(self.params["declared"]))

    def initial_state(self):
        return {"x": jnp.ones(())}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] - dt * state["x"]}


def error_at(steps: int) -> float:
    node, dt = EulerDecay("decay", HORIZON / steps), HORIZON / steps
    state = node.initial_state()
    for _ in range(steps):
        state = node.update(state, {}, dt)
    return abs(float(state["x"]) - math.exp(-HORIZON)) / math.exp(-HORIZON)


LADDER = OrderLadder(RefinementAxis.TIME, error_at, (16, 32, 64, 128))


@pytest.mark.parametrize(("declared", "ladders", "status"), [
    (1.0, (LADDER,), "PASS"),
    (2.0, (LADDER,), "FAIL"),
    (1.0, (), "FAIL"),
])
def test_a_declared_order_is_measured_and_a_claim_with_no_ladder_fails(declared, ladders, status):
    results = node_results(EulerDecay("decay", 0.01, declared=declared), ladders=ladders)
    print(f"\n{results['temporal_order']}")
    assert results["temporal_order"].status == status
    assert "spatial_order" not in results
    assert all(r.passed for name, r in results.items() if name != "temporal_order")
    if not ladders:
        assert "gives no refinement ladder" in results["temporal_order"].detail
