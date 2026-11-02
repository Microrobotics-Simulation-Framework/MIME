"""The dynamic arm's order in time and MADDENING's battery, on A0's AR4 as the build makes it (float64, per module):
first order in time, measured on the AR4 under its weight, and the node passing ``verify_node``."""

import warnings

import _cache_ceiling
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import dynamic_rpm, holding_torques
from maddening.testing.mms import RefinementAxis
from maddening.testing.verification import DEFAULT_CHECKS

from mime.testing import OrderLadder, node_results


# -- its order, and MADDENING's battery ------------------------------------------------------------------------------

ORDER_HORIZON, REFERENCE_STEPS = 0.05, 4096
#: How far above where its worker stands a test of this file lets resident memory grow before it gives its compiled
#: programs back (tests/_cache_ceiling.py): the ``params_effective`` check compiles new programs for each of its
#: examples and held 7.7 GB of them at its end.
HEADROOM_MIB = 1024


@pytest.fixture(autouse=True)
def programs_given_back():
    with _cache_ceiling.under_ceiling(HEADROOM_MIB):
        yield


@pytest.fixture(scope="module")
def ar4_node():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dynamic_rpm("ordered_arm", motor=False, laws=()).build().graph.get_node("rpm.arm")


def angles_after(node, steps: int) -> np.ndarray:
    """The AR4's joints after ``ORDER_HORIZON`` in ``steps`` steps from its home pose, at rest, under its weight and a
    tenth of the torques that would hold it."""
    p = node._step_constants(None)
    dt = ORDER_HORIZON / steps
    torque = 0.1 * jnp.asarray(holding_torques())

    def step(held, _):
        return node.advance(p, held, {"joint_torques": torque}, dt), None

    final, _ = jax.jit(lambda s: jax.lax.scan(step, s, None, length=steps))(node.initial_state())
    return np.asarray(final["joint_positions"])


def battery_options() -> dict:
    return {"bounds": {"joint_positions": (-1.5, 1.5), "joint_velocities": (-1.0, 1.0), "frame_poses": (-1.0, 1.0)},
            "boundary_bounds": {"joint_torques": (-0.5, 0.5), "link_wrench": (-1e-2, 1e-2),
                                "link_momentum": (-1e-3, 1e-3), "link_inertia": (-1e-7, 1e-7),
                                "mount_poses": (-1.0, 1.0)},
            "constrain_state": lambda state: {**state, "frame_poses": unit_rows(state["frame_poses"])},
            "constrain_boundary": lambda given: {**given, "mount_poses": unit_rows(given["mount_poses"])}}


def test_the_dynamic_arm_is_first_order_in_time(ar4_node):
    reference = angles_after(ar4_node, REFERENCE_STEPS)

    def error_at(steps: int) -> float:
        return float(np.max(np.abs(angles_after(ar4_node, steps) - reference)))

    ladder = OrderLadder(RefinementAxis.TIME, error_at, (16, 32, 64, 128))
    results = node_results(ar4_node, {**battery_options(), "checks": []}, (ladder,))
    print("\n" + "\n".join(str(r) for r in results.values()))
    assert results and all(r.passed for r in results.values())


#: Longest first (the reverse of MADDENING's order, which runs the cheap checks first): the file holds a shard of its own
#: (tests/shard_weights.json), pytest-xdist hands each worker a run of consecutive tests, and a long check that comes
#: last in a worker's run leaves the others idle.
LONGEST_FIRST = tuple(reversed(DEFAULT_CHECKS))


@pytest.mark.parametrize("check", LONGEST_FIRST)
def test_the_dynamic_arm_passes_maddenings_check(ar4_node, check):
    """MADDENING's battery, one check to a test: the checks do not depend on each other, and the battery whole as one
    test was 541 s of one worker in a shard run and held 8.1 GB at its end, which was its shard's whole time and most
    of its memory. ``params_effective`` is nearly all of both."""
    results = node_results(ar4_node, {**battery_options(), "checks": [check]})
    print("\n" + "\n".join(str(r) for r in results.values()))
    assert check in results and results[check].passed, results[check]


def unit_rows(poses):
    held = np.asarray(poses)
    rows = held.reshape(-1, 7).astype(np.float64)
    quaternions = rows[:, 3:]
    lengths = np.linalg.norm(quaternions, axis=1, keepdims=True)
    rows[:, 3:] = np.where(lengths > 0, quaternions / np.where(lengths > 0, lengths, 1.0), [1.0, 0.0, 0.0, 0.0])
    return rows.reshape(held.shape).astype(held.dtype)
