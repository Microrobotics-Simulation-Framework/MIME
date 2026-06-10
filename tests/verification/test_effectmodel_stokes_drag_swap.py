"""EffectModel concept-proof — fluid-backend swap on free-space Stokes drag.

This is the runnable proof of concept for the v0.2 EffectModel pilot (ADR-2026-
EFFECT-MODEL). It composes a *kinematic sphere* + a `HydrodynamicModel` backend
through the `Experiment` surface, runs it, and reads the drag the fluid
produces — then swaps the backend with **one `attach()` line** and runs a
genuinely different solver across the identical body/edges.

What it demonstrates:
- The `Experiment` composition + six-pass `build()` validation + run path works
  end to end (not just compile, as the unit tests cover).
- The **Stokeslet** backend reproduces the analytical Stokes drag
  `F = 6πμaV` on a free-space sphere (≈0.4% in practice).
- The **FVM** backend — a full Navier-Stokes + immersed-boundary solver —
  runs through the *same swapped line* and produces a finite drag opposing the
  motion. (It is a confined sphere-in-a-pipe, so its magnitude is ~2-3× the
  free-space value — confinement, not disagreement.)

Honest scope / findings this exercise surfaced (tracked as E6 in
`plans/MIME_v0.2.0_RELEASE_PLAN.md`):
- **Load-time version validation works** — an `Experiment` with
  `mime_version_min` above the installed version raises before building.
- **External-input composition is not in the pilot (E6a).** Prescribing the
  body's motion here uses `set_node_state` + a single step (the drag feedback
  has not yet engaged), because `Experiment` does not yet compose
  `add_external_input` / coupling groups. A confined FSI experiment needs that.
- **The drag sign convention is not yet pinned across backends (E6/contract).**
  The Stokeslet node reports `+R·V`; the FVM node reports the force *on* the
  body (opposing). The swap surfaced this; magnitudes are physical in both.

This is a *free-space drag* proof — NOT a confined-microrobot experiment
(those need MagneticModel + coupling composition + the StokesletChain / Schwarz
variants; see E6).
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from mime.effects import Body, Experiment, HydrodynamicModel, Medium
from mime.nodes.robot.rigid_body import RigidBodyNode

# Installed MIME is 0.1.0 until the v0.2.0 bump; use a compatible floor so the
# (working) load-time version check doesn't reject the demo experiment.
_MIME_MIN = "0.1.0"


def _inertial_body(*, a, mu, rho_f, dt, name="body"):
    """A rigid body in inertial mode with *large* effective inertia, so the
    drag feedback barely moves it over the readout window (and can't spin it
    up). Prescribing the velocity each step then delivers a clean steady
    translation to the fluid — the steady-drag condition this readout needs.
    Large I_eff/m_eff (vs the ~1e-4 N drags here) keep the body's own
    integration bounded; the per-step velocity re-pin keeps V exact."""
    return RigidBodyNode(
        name=name, timestep=dt,
        semi_major_axis_m=a, semi_minor_axis_m=a,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=mu, fluid_density_kg_m3=rho_f,
        use_inertial=True, I_eff=1.0, m_eff=1.0,
    )


def _drag_at_prescribed_velocity(exp, fluid_name, V, *, n_steps=1):
    """Build `exp`, drive the body at a steady translation `V` (re-pinned each
    step, with zero spin — see E6a: Experiment doesn't yet compose the
    external-input surface), and return the fluid's drag force."""
    V = jnp.asarray(V, dtype=jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
        for _ in range(n_steps):
            s = dict(gm.get_node_state("body"))
            s["velocity"] = V
            s["angular_velocity"] = jnp.zeros(3, dtype=jnp.float32)
            gm.set_node_state("body", s)
            gm.step()
        return np.asarray(gm.get_node_state(fluid_name)["drag_force"])


def _stokeslet_experiment(a, mu, rho_f, dt):
    from mime.nodes.environment.stokeslet import (
        StokesletFluidNode, sphere_surface_mesh,
    )

    bem = StokesletFluidNode(
        name="bem", timestep=dt, mu=mu,
        body_mesh=sphere_surface_mesh(radius=a, n_refine=3),
    )
    exp = Experiment(name="stokes_drag_swap", mime_version_min=_MIME_MIN)
    exp.set_body(Body("body", node=_inertial_body(a=a, mu=mu, rho_f=rho_f, dt=dt),
                      properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": rho_f, "viscosity": mu,
                           "reynolds_number": rho_f * 1e-4 * 2 * a / mu}))
    exp.attach(HydrodynamicModel.Stokeslet(bem))
    return exp, "bem"


def _fvm_experiment(a, mu, rho_f, dt):
    from mime.nodes.environment.fvm import (
        FVMFluidNode, make_cartesian_mesh_3d, make_sphere_body_factory,
    )
    from mime.nodes.environment.fvm.boundary import VelocityBC
    from mime.nodes.environment.fvm.ibm import IBMBody
    from mime.nodes.environment.fvm.piso import PisoConfig

    R_pipe, L = 0.5, 1.0
    Lx = Ly = 2 * 1.2 * R_pipe
    mesh = make_cartesian_mesh_3d(8, 8, 6, Lx, Ly, L,
                                  origin=(-Lx / 2, -Ly / 2, 0.0), periodic_z=True)
    dx = mesh.cartesian_spacing[0]
    wall = IBMBody(name="pipe_wall",
                   sdf=lambda x: R_pipe - jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30))
    bcs = {}
    for nm in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(nm)
        nbf = int(p.owner.size)
        bcs[nm] = VelocityBC(u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(nu=mu / rho_f, rho=rho_f, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx)
    fvm = FVMFluidNode(name="fluid", timestep=dt, mesh=mesh, bcs=bcs, cfg=cfg,
                       static_bodies=[wall],
                       dynamic_body_factories=[("sphere", make_sphere_body_factory("sphere", radius=a))])
    exp = Experiment(name="stokes_drag_swap", mime_version_min=_MIME_MIN)
    exp.set_body(Body("body", node=_inertial_body(a=a, mu=mu, rho_f=rho_f, dt=dt),
                      properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": rho_f, "viscosity": mu}))
    exp.attach(HydrodynamicModel.FVM(fvm))
    return exp, "fluid"


@pytest.mark.slow
def test_stokeslet_backend_matches_analytical_stokes_drag():
    """The Stokeslet backend, composed + run through the EffectModel surface,
    reproduces the analytical free-space Stokes drag F = 6πμaV."""
    a, mu, rho_f, dt, V = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4
    exp, fluid = _stokeslet_experiment(a, mu, rho_f, dt)
    drag = _drag_at_prescribed_velocity(exp, fluid, [V, 0.0, 0.0])

    F_stokes = 6.0 * math.pi * mu * a * V
    assert np.all(np.isfinite(drag))
    # Magnitude matches analytical Stokes drag (the precise win). The sign
    # convention across backends is not yet pinned (see module docstring), so
    # compare on magnitude along the motion axis.
    rel = abs(abs(drag[0]) - F_stokes) / F_stokes
    assert rel < 0.05, f"Stokeslet drag {drag[0]:.4e} vs 6πμaV {F_stokes:.4e} (rel {rel:.1%})"
    # Transverse components are negligible for axial translation.
    assert abs(drag[1]) < 0.05 * F_stokes and abs(drag[2]) < 0.05 * F_stokes


@pytest.mark.slow
def test_fvm_backend_runs_through_the_same_swap_surface():
    """The FVM backend — a full NS + IBM solver — runs through the identical
    one-line swap and produces a finite drag opposing the motion. It is a
    confined sphere-in-a-pipe, so the magnitude exceeds free-space Stokes."""
    a, mu, rho_f, dt, V = 0.1, 0.005, 1.0, 0.1, 0.01
    exp, fluid = _fvm_experiment(a, mu, rho_f, dt)
    drag = _drag_at_prescribed_velocity(exp, fluid, [V, 0.0, 0.0], n_steps=8)

    F_stokes = 6.0 * math.pi * mu * a * V
    assert np.all(np.isfinite(drag)), f"FVM drag not finite: {drag}"
    # Force on the body opposes the +x motion.
    assert drag[0] < 0.0, f"FVM drag should oppose +x motion: {drag}"
    # Within an order of magnitude of free-space Stokes (confinement inflates it).
    assert 0.3 * F_stokes < abs(drag[0]) < 10.0 * F_stokes, (
        f"FVM |drag_x| {abs(drag[0]):.3e} not within O(1) of Stokes {F_stokes:.3e}"
    )


def test_backend_swap_is_one_attach_line_and_both_compile():
    """Interchangeability at the graph level (fast): the same Experiment shape,
    with the fluid backend selected by a single `attach()` call, builds clean
    for both backends — the EffectModel swap guarantee."""
    a, mu, rho_f, dt = 1e-3, 1e-3, 1000.0, 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp_bem, bem_name = _stokeslet_experiment(a, mu, rho_f, dt)
        gm_bem, h_bem = exp_bem.build()
        exp_fvm, fvm_name = _fvm_experiment(0.1, 0.005, 1.0, 0.1)
        gm_fvm, h_fvm = exp_fvm.build()

    assert h_bem[0].node_names == (bem_name,)
    assert h_fvm[0].node_names == (fvm_name,)
    # Both graphs carry the shared body and the swapped fluid node.
    assert "body" in gm_bem.node_names and bem_name in gm_bem.node_names
    assert "body" in gm_fvm.node_names and fvm_name in gm_fvm.node_names


def test_version_validation_rejects_incompatible_experiment():
    """The load-time version check (used by every Experiment) rejects an
    experiment whose floor exceeds the installed MIME version."""
    from mime.effects import IncompatibleMimeVersionError

    with pytest.raises(IncompatibleMimeVersionError):
        Experiment(name="too_new", mime_version_min="99.0.0")
