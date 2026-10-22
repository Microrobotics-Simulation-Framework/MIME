"""SourceInputProvider — constant / node-field / external source drives (E2).

Pins the three ways a SourcedEffectModel's per-source input is provided, each
resolving uniformly into a GraphManager: a ConstantInput materialises a
constant node + edge; a NodeFieldRef materialises an edge from an existing
node; an ExternalInputRef materialises a graph-external input. MagneticSource
resolves a bundle of them.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryFluxSpec, SimulationNode

from mime.effects import (
    ConstantInput,
    ExternalInputRef,
    MagneticSource,
    NodeFieldRef,
)
from mime.nodes.actuation.constant_input import ConstantInputNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode

_DT = 1e-3
_IDENTITY_POSE = jnp.array([0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0])


def _magnet(name="magnet"):
    return PermanentMagnetNode(
        name, _DT, dipole_moment_a_m2=1.0,
        magnet_radius_m=2e-3, magnet_length_m=4e-3,
    )


class _PoseStub(SimulationNode):
    def initial_state(self) -> dict:
        return {"rotor_pose_world": _IDENTITY_POSE}

    def update(self, state, boundary_inputs, dt) -> dict:
        return state

    def boundary_flux_spec(self) -> dict:
        return {"rotor_pose_world": BoundaryFluxSpec(
            shape=(7,), description="pose", output_units="SI")}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt) -> dict:
        return {"rotor_pose_world": state["rotor_pose_world"]}


# ── ConstantInputNode ─────────────────────────────────────────────────────

def test_constant_input_node_emits_value():
    node = ConstantInputNode("c", _DT, value=_IDENTITY_POSE)
    out = node.update(node.initial_state(), {}, _DT)
    assert jnp.allclose(out["value"], _IDENTITY_POSE)
    assert node.boundary_flux_spec()["value"].shape == (7,)


# ── providers ──────────────────────────────────────────────────────────────

def test_constant_input_resolves_and_compiles():
    gm = GraphManager()
    gm.add_node(_magnet())
    ConstantInput(_IDENTITY_POSE).resolve_into(
        gm, "magnet", "magnet_pose_world", timestep=_DT)
    assert "magnet__magnet_pose_world__const" in gm.node_names
    gm.compile()


def test_node_field_ref_resolves_as_edge_and_compiles():
    gm = GraphManager()
    gm.add_node(_magnet())
    gm.add_node(_PoseStub("motor", _DT))
    NodeFieldRef("motor", "rotor_pose_world").resolve_into(
        gm, "magnet", "magnet_pose_world")
    gm.compile()
    # an edge motor.rotor_pose_world -> magnet.magnet_pose_world now exists
    assert any(
        e for e in gm._edges
        if getattr(e, "source_node", None) == "motor"
        and getattr(e, "target_node", None) == "magnet"
    )


def test_external_input_ref_resolves_and_compiles():
    gm = GraphManager()
    gm.add_node(_magnet())
    ExternalInputRef(shape=(7,), dtype=jnp.float32).resolve_into(
        gm, "magnet", "magnet_pose_world")
    gm.compile()
    assert gm._external_inputs  # at least one external input registered


def test_constant_input_custom_name():
    gm = GraphManager()
    gm.add_node(_magnet())
    ConstantInput(jnp.zeros(3), name="myconst").resolve_into(
        gm, "magnet", "target_position_world", timestep=_DT)
    assert "myconst" in gm.node_names


# ── MagneticSource bundle ──────────────────────────────────────────────────

def test_magnetic_source_resolves_all_inputs():
    gm = GraphManager()
    gm.add_node(_magnet())
    gm.add_node(_PoseStub("motor", _DT))
    src = MagneticSource(
        name="dipole_a",
        inputs={
            "magnet_pose_world": NodeFieldRef("motor", "rotor_pose_world"),
            "target_position_world": ExternalInputRef(shape=(3,)),
        },
    )
    src.resolve_all(gm, "magnet", timestep=_DT)
    gm.compile()
    assert gm._external_inputs
    assert any(
        getattr(e, "source_node", None) == "motor" for e in gm._edges
    )
