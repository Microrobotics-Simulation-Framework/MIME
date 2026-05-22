"""§8 Step 4 — the shared fluid-node contract.

``src/mime/nodes/environment/FLUID_NODE_CONTRACT.md`` defines the interface a
fluid node exposes so it is interchangeable in an experiment graph: the
``drag_force`` / ``drag_torque`` outputs and the ``body_*`` inputs.

``IBLBMFluidNode``, ``StokesletFluidNode`` and ``DefectCorrectionFluidNode``
already use the contract names — pinned by their own tests (e.g.
``test_iblbm_graph_wiring`` asserts ``drag_force``/``drag_torque`` in the
fluxes, ``test_b2_stokes_drag`` reads ``drag_force``). This pins the additive
reconciliation of ``FVMFluidNode``: a single-body FVM node now also exposes the
contract names, alongside its per-body ``force_<name>`` / ``<name>_*``
multi-body extension.
"""

import jax.numpy as jnp

from mime.nodes.environment.fvm import (
    FVMFluidNode,
    make_cartesian_mesh_3d,
    make_sphere_body_factory,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.piso import PisoConfig


def _single_body_fvm_node() -> FVMFluidNode:
    """A minimal single-body FVM node (one dynamic body named 'sphere')."""
    R_pipe, L = 0.5, 1.0
    Lx = Ly = 2 * 1.2 * R_pipe
    mesh = make_cartesian_mesh_3d(
        8, 8, 6, Lx, Ly, L,
        origin=(-Lx / 2, -Ly / 2, 0.0), periodic_z=True,
    )
    dx = mesh.cartesian_spacing[0]
    wall = IBMBody(
        name="pipe_wall",
        sdf=lambda x: R_pipe - jnp.sqrt(
            x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30),
    )
    bcs = {}
    for name in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(name)
        nbf = int(p.owner.size)
        bcs[name] = VelocityBC(
            u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)),
        )
    cfg = PisoConfig(
        nu=0.005, rho=1.0, gamma_conv=0.5, n_corrector=2,
        pressure_bc=("neumann", "neumann", "periodic"),
        velocity_bc=("dirichlet", "dirichlet", "periodic"),
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    return FVMFluidNode(
        name="fluid", timestep=0.1, mesh=mesh, bcs=bcs, cfg=cfg,
        static_bodies=[wall],
        dynamic_body_factories=[
            ("sphere", make_sphere_body_factory("sphere", radius=0.1)),
        ],
    )


def test_single_body_fvm_node_exposes_contract_names():
    """A single-body FVMFluidNode exposes the shared-contract drag_force /
    drag_torque outputs and body_* inputs — alongside its per-body fields."""
    node = _single_body_fvm_node()

    # Contract inputs are declared (per-body extension still present too).
    inputs = node.boundary_input_spec()
    for name in ("body_position", "body_velocity", "body_angular_velocity"):
        assert name in inputs, f"missing contract input {name!r}"
    assert "sphere_position" in inputs, "per-body extension input lost"

    # Contract outputs are declared (per-body extension still present too).
    fluxes = node.boundary_flux_spec()
    for name in ("drag_force", "drag_torque"):
        assert name in fluxes, f"missing contract output {name!r}"
    assert "force_sphere" in fluxes, "per-body extension output lost"

    # Contract output fields exist in the state pytree.
    state = node.initial_state()
    assert "drag_force" in state and "drag_torque" in state
    assert state["drag_force"].shape == (3,)
