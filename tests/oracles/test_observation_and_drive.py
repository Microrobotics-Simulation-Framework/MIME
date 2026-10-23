"""The swim speed and the constant drive against their closed forms. A body moving at a constant velocity ``v`` swims
at ``v . along``; its mean over the steps ending after ``after`` is that speed; before ``after`` nothing is counted.
The constant drive's node holds its value every step, and a value written through ``params`` moves it without a
rebuild. The control: a speed read along another direction is another number."""

import jax.numpy as jnp
import numpy as np

from mime.nodes.actuation import ConstantNode
from mime.nodes.body import SwimSpeedNode

DT = 1e-3
ALONG = np.array([1.0, 0.0, 0.0])
TWIST = np.array([2e-3, -1e-3, 4e-4, 30.0, 0.0, 0.0])


def run(node: SwimSpeedNode, steps: int, twist=TWIST) -> dict:
    state = node.initial_state()
    for _ in range(steps):
        state = node.update(state, {"twist": jnp.asarray(twist)}, DT)
    return state


def test_the_speed_and_its_mean_are_the_velocity_along_the_direction():
    state = run(SwimSpeedNode("swim", DT, along=ALONG), 10)
    assert float(state["speed"]) == float(TWIST[0])
    assert abs(float(state["mean_speed"]) - float(TWIST[0])) <= 1e-15
    assert abs(float(state["distance"]) - 10 * DT * float(TWIST[0])) <= 1e-18
    assert int(state["steps"]) == 10


def test_steps_ending_before_after_are_not_counted():
    node = SwimSpeedNode("swim", DT, along=ALONG, after=4.5e-3)
    state = run(node, 4)
    assert float(state["observed"]) == 0.0 and float(state["mean_speed"]) == 0.0
    state = run(node, 10)
    assert abs(float(state["observed"]) - 6 * DT) <= 1e-18


def test_a_speed_along_another_direction_is_another_number():
    """The control: the oracle can tell the direction read along."""
    state = run(SwimSpeedNode("swim", DT, along=(0.0, 1.0, 0.0)), 3)
    assert float(state["speed"]) == float(TWIST[1]) != float(TWIST[0])


def test_the_constant_drive_holds_its_value_and_a_written_value_moves_it():
    node = ConstantNode("drive", DT, value=10.0)
    state = node.initial_state()
    assert float(state["value"]) == 10.0
    assert float(node.update(state, {}, DT)["value"]) == 10.0
    assert float(node.update(state, {}, DT, params={"value": jnp.asarray(11.0)})["value"]) == 11.0
