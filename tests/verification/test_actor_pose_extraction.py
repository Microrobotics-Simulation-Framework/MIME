"""Declarative actor pose extraction in the experiment runner.

Pins the actor → ``ResultFrame`` mapping done by
``_state_to_result_frame``: ``pose_from: { node, field, index? }`` reads a
7-vector pose from the named graph node's state (optionally an indexed row
of an ``(N, 7)`` array); ``state_fields`` is the generic fallback when the
actor name matches a graph node directly. The hardcoded ``arm_link_<N>`` /
``motor_rotor`` / ``magnet`` recipes that used to live in the runner are
gone — every composite actor now declares its source in YAML.
"""

import jax.numpy as jnp
import pytest

from mime.runner.server import _state_to_result_frame


def test_pose_from_simple_state_field():
    """``pose_from`` reads a 7-vector pose from ``state[node][field]``;
    two actors pointing at the same source mirror each other (the
    motor_rotor / magnet pattern)."""
    state = {
        "motor": {
            "rotor_pose_world": jnp.array(
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32,
            ),
        },
    }
    actors = {
        "motor_rotor": {
            "pose_from": {"node": "motor", "field": "rotor_pose_world"},
        },
        "magnet": {
            "pose_from": {"node": "motor", "field": "rotor_pose_world"},
        },
    }
    frame = _state_to_result_frame(0.0, state, actors)
    assert frame["positions"]["motor_rotor"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert frame["orientations"]["motor_rotor"] == {
        "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0,
    }
    assert frame["positions"]["magnet"] == frame["positions"]["motor_rotor"]


def test_pose_from_indexed_row():
    """``pose_from`` with ``index`` reads a row of an ``(N, 7)`` array
    (the arm-link pattern)."""
    link_poses = jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    state = {"arm": {"link_poses_world": link_poses}}
    actors = {
        "arm_link_1": {
            "pose_from": {
                "node": "arm", "field": "link_poses_world", "index": 1,
            },
        },
    }
    frame = _state_to_result_frame(0.0, state, actors)
    assert frame["positions"]["arm_link_1"] == {
        "x": 1.0, "y": 0.0, "z": 0.0,
    }
    assert frame["orientations"]["arm_link_1"] == {
        "w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0,
    }


def test_state_fields_fallback_position_orientation():
    """No ``pose_from`` → the actor name must match a graph node and
    ``state_fields`` drives the extraction (the generic body actor)."""
    state = {
        "body": {
            "position": jnp.array([1.0, 2.0, 3.0]),
            "orientation": jnp.array([1.0, 0.0, 0.0, 0.0]),
        },
    }
    actors = {"body": {"state_fields": ["position", "orientation"]}}
    frame = _state_to_result_frame(0.0, state, actors)
    assert frame["positions"]["body"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert frame["orientations"]["body"] == {
        "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0,
    }


def test_pose_from_missing_source_silently_skips():
    """A ``pose_from`` referencing a node not yet in the state pytree —
    or a state pytree where the field hasn't been populated — silently
    skips the actor for that frame (matches the pre-refactor behaviour
    that swallowed transient first-frame absences)."""
    actors = {
        "motor_rotor": {
            "pose_from": {"node": "motor", "field": "rotor_pose_world"},
        },
    }
    # Missing node entirely.
    frame = _state_to_result_frame(0.0, {}, actors)
    assert "motor_rotor" not in frame["positions"]
    # Node present, field absent.
    frame = _state_to_result_frame(0.0, {"motor": {}}, actors)
    assert "motor_rotor" not in frame["positions"]


@pytest.mark.parametrize("idx,expected_x", [(0, 0.0), (1, 1.0), (2, 2.0)])
def test_pose_from_index_picks_the_right_row(idx, expected_x):
    """``index`` indexes axis 0 of an ``(N, 7)`` field."""
    state = {
        "arm": {
            "link_poses_world": jnp.stack([
                jnp.array([float(i), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                for i in range(3)
            ]),
        },
    }
    actors = {
        "arm_link": {
            "pose_from": {
                "node": "arm", "field": "link_poses_world", "index": idx,
            },
        },
    }
    frame = _state_to_result_frame(0.0, state, actors)
    assert frame["positions"]["arm_link"]["x"] == expected_x
