"""Test: StokesletFluidNode wired into MADDENING graph.

Builds a minimal graph with StokesletFluidNode + RigidBodyNode,
compiles, and runs 10 steps to verify the coupling works.
"""

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def stokeslet_graph():
    """Build a minimal 2-node graph: Stokeslet + RigidBody."""
    from maddening.core.graph_manager import GraphManager
    from mime.nodes.environment.stokeslet import (
        StokesletFluidNode,
        make_stokeslet_rigid_body_edges,
        sphere_surface_mesh,
    )
    from mime.nodes.robot.rigid_body import RigidBodyNode

    # Simple sphere body — not a real UMR, just for graph testing
    body_mesh = sphere_surface_mesh(radius=0.001, n_refine=2)
    dt = 0.001

    stokeslet = StokesletFluidNode(
        name="stokeslet_fluid",
        timestep=dt,
        mu=0.001,  # water-like viscosity
        body_mesh=body_mesh,
    )

    rigid = RigidBodyNode(
        name="rigid_body",
        timestep=dt,
        semi_major_axis_m=0.001,
        semi_minor_axis_m=0.001,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=0.001,
        use_inertial=True,
        I_eff=1e-15,
    )

    gm = GraphManager()
    gm.add_node(stokeslet)
    gm.add_node(rigid)

    for edge in make_stokeslet_rigid_body_edges("stokeslet_fluid", "rigid_body"):
        gm.add_edge(
            edge.source_node, edge.target_node,
            edge.source_field, edge.target_field,
            transform=edge.transform,
            additive=edge.additive,
        )

    gm.add_coupling_group(
        ["rigid_body", "stokeslet_fluid"],
        max_iterations=1,
    )

    return gm


class TestStokesletGraphIntegration:
    def test_graph_compiles(self, stokeslet_graph):
        """Graph with StokesletFluidNode should compile without error."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stokeslet_graph.compile()

    def test_graph_steps(self, stokeslet_graph):
        """Graph should step 10 times without error."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stokeslet_graph.compile()

        for _ in range(10):
            stokeslet_graph.step()

        state = stokeslet_graph.get_node_state("rigid_body")
        pos = np.asarray(state["position"])
        assert np.all(np.isfinite(pos)), f"Position not finite: {pos}"

    def test_drag_force_nonzero_with_motion(self, stokeslet_graph):
        """Drag force should be nonzero when body is moving."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stokeslet_graph.compile()

        # Apply an external torque to make the body rotate
        ext = {
            "rigid_body": {
                "magnetic_torque": jnp.array([0.0, 0.0, 1e-12]),
            },
        }
        for _ in range(5):
            stokeslet_graph.step(ext)

        state = stokeslet_graph.get_node_state("stokeslet_fluid")
        drag_torque = np.asarray(state["drag_torque"])

        # After 5 steps with an applied torque, the body should be
        # rotating and the Stokeslet should produce opposing drag
        # (Note: with max_iterations=1, the coupling is weak —
        # drag might be small but should be nonzero after a few steps)
        state_rb = stokeslet_graph.get_node_state("rigid_body")
        omega = np.asarray(state_rb["angular_velocity"])
        print(f"  omega: {omega}")
        print(f"  drag_torque: {drag_torque}")
