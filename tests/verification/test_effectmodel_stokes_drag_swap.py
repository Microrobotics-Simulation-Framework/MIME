"""EffectModel concept-proof — fluid-backend swap on free-space Stokes drag.

The runnable proof of concept for the v0.2 EffectModel pilot (ADR-2026-
EFFECT-MODEL): compose a *kinematic sphere* + a `HydrodynamicModel` backend
through the `Experiment` surface, drive it at a prescribed velocity, read the
drag — then swap the backend with **one `attach()` line** and run a genuinely
different solver across the identical body/edges.

What it demonstrates:
- The `Experiment` composition + six-pass `build()` validation + run path
  works end to end, including **graph-external inputs** (the body's prescribed
  velocity is injected via `Experiment.add_external_input` + `step()` — no
  `set_node_state` harness).
- The **Stokeslet** backend reproduces the analytical Stokes drag magnitude
  `F = 6πμaV` on a free-space sphere (≈0.4%).
- The **FVM** backend — a full Navier-Stokes + immersed-boundary solver —
  runs through the *same swapped line* and produces a finite drag (confined
  sphere-in-a-pipe; ~13% under free-space at this resolution).

Sign convention — **normalized in the adapter (v0.2).** The raw fluid nodes do
not share a sign convention: IBLBM and the standalone Stokeslet report
``+R·(motion)`` (the *reaction*, i.e. the force on the fluid), while FVM
reports the force *on* the body (opposing motion). Delivered as-is to a body
that *adds* the drag, the reaction sign is **anti-dissipative** — verified: a
de-Boer UMR diverges when fed IBLBM's ``+R·ω`` with the ``omega_max`` clamp
removed. The `HydrodynamicModel` adapter therefore normalizes every backend to
the contract sign (**force on the body, opposing motion**) via
``native_drag_sign``: IBLBM / Stokeslet flip, FVM is unchanged.
``test_backends_deliver_force_on_body`` asserts the *delivered* drag opposes
the motion for both runnable backends.

Scope: this is a *free-space drag* proof. The confined-microrobot experiments
need the v0.3 magnetic family + coupling composition + StokesletChain/Schwarz
variants (E6).
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


def _kinematic_body(*, a, mu, rho_f, dt, name="body"):
    """A rigid body in kinematic mode: its velocity / angular velocity are
    prescribed each step from graph-external inputs (`external_velocity` /
    `external_angular_velocity`), and it emits them as `velocity` /
    `angular_velocity` for the fluid node to read. The clean way to drive a
    steady translation for a drag readout."""
    return RigidBodyNode(
        name=name, timestep=dt,
        semi_major_axis_m=a, semi_minor_axis_m=a,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=mu, fluid_density_kg_m3=rho_f,
        kinematic_mode=True,
    )


def _with_prescribed_drive(exp):
    """Declare the body's prescribed-motion external inputs on `exp`."""
    exp.add_external_input("body", "external_velocity", (3,))
    exp.add_external_input("body", "external_angular_velocity", (3,))
    return exp


def _drag_at_velocity(exp, fluid_name, V, *, n_steps=1):
    """Build `exp`, drive the body at steady translation `V` via external
    inputs, step, and return the fluid's drag force."""
    ext = {"body": {"external_velocity": jnp.asarray(V, dtype=jnp.float32),
                    "external_angular_velocity": jnp.zeros(3, dtype=jnp.float32)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
        for _ in range(n_steps):
            gm.step(ext)
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
    exp.set_body(Body("body", node=_kinematic_body(a=a, mu=mu, rho_f=rho_f, dt=dt),
                      properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": rho_f, "viscosity": mu,
                           "reynolds_number": rho_f * 1e-4 * 2 * a / mu}))
    _with_prescribed_drive(exp)
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
    exp.set_body(Body("body", node=_kinematic_body(a=a, mu=mu, rho_f=rho_f, dt=dt),
                      properties={"hydrodynamic": {}}))
    exp.set_medium(Medium({"density": rho_f, "viscosity": mu}))
    _with_prescribed_drive(exp)
    exp.attach(HydrodynamicModel.FVM(fvm))
    return exp, "fluid"


@pytest.mark.slow
def test_stokeslet_backend_matches_analytical_stokes_drag():
    """The Stokeslet backend, composed + run through the EffectModel surface
    with a prescribed-velocity drive, reproduces the analytical free-space
    Stokes drag magnitude F = 6πμaV (to a few %)."""
    a, mu, rho_f, dt, V = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4
    exp, fluid = _stokeslet_experiment(a, mu, rho_f, dt)
    drag = _drag_at_velocity(exp, fluid, [V, 0.0, 0.0], n_steps=2)

    F_stokes = 6.0 * math.pi * mu * a * V
    assert np.all(np.isfinite(drag))
    rel = abs(abs(drag[0]) - F_stokes) / F_stokes
    assert rel < 0.05, f"Stokeslet |drag| {abs(drag[0]):.4e} vs 6πμaV {F_stokes:.4e} (rel {rel:.1%})"
    # Transverse components negligible for axial translation.
    assert abs(drag[1]) < 0.05 * F_stokes and abs(drag[2]) < 0.05 * F_stokes


@pytest.mark.slow
def test_fvm_backend_runs_through_the_same_swap_surface():
    """The FVM backend — a full NS + IBM solver — runs through the identical
    one-line swap and produces a finite drag, within an order of magnitude of
    free-space Stokes (it is a confined sphere-in-a-pipe)."""
    a, mu, rho_f, dt, V = 0.1, 0.005, 1.0, 0.1, 0.01
    exp, fluid = _fvm_experiment(a, mu, rho_f, dt)
    drag = _drag_at_velocity(exp, fluid, [V, 0.0, 0.0], n_steps=15)

    F_stokes = 6.0 * math.pi * mu * a * V
    assert np.all(np.isfinite(drag)), f"FVM drag not finite: {drag}"
    assert 0.3 * F_stokes < abs(drag[0]) < 10.0 * F_stokes, (
        f"FVM |drag_x| {abs(drag[0]):.3e} not within O(1) of Stokes {F_stokes:.3e}"
    )


def _delivered_drag_force(exp, fluid_name, raw_drag):
    """Apply the forward `drag_force` edge's transform (the adapter's sign
    normalization + any unit conversion) to a raw fluid drag, i.e. what the
    body actually receives."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm, _ = exp.build()
    edge = next(e for e in gm._edges
                if e.source_node == fluid_name and e.target_node == "body"
                and e.source_field == "drag_force")
    tf = edge.transform
    return np.asarray(tf(jnp.asarray(raw_drag)) if tf is not None else raw_drag)


@pytest.mark.slow
def test_backends_deliver_force_on_body():
    """The adapter normalizes every backend to the contract sign: the drag
    *delivered to the body* opposes the body's motion (force on body) — even
    though the raw node outputs disagree (Stokeslet/IBLBM report `+R·motion`;
    FVM reports force-on-body). This is what makes the swap dissipative and
    sign-safe in a coupled run."""
    # Stokeslet: raw drag is along +x (reaction); delivered opposes.
    a, mu, rho_f, dt, V = 1e-3, 1e-3, 1000.0, 1e-3, 1e-4
    exp, fluid = _stokeslet_experiment(a, mu, rho_f, dt)
    raw = _drag_at_velocity(exp, fluid, [V, 0.0, 0.0], n_steps=2)
    assert raw[0] > 0.0, f"Stokeslet raw convention changed: {raw}"
    delivered = _delivered_drag_force(exp, fluid, raw)
    assert delivered[0] < 0.0, f"Stokeslet delivered drag not opposing: {delivered}"

    # FVM: raw drag already opposes +x; delivered still opposes (no flip).
    a2, mu2, rho2, dt2, V2 = 0.1, 0.005, 1.0, 0.1, 0.01
    exp2, fluid2 = _fvm_experiment(a2, mu2, rho2, dt2)
    raw2 = _drag_at_velocity(exp2, fluid2, [V2, 0.0, 0.0], n_steps=15)
    assert raw2[0] < 0.0, f"FVM raw convention changed: {raw2}"
    delivered2 = _delivered_drag_force(exp2, fluid2, raw2)
    assert delivered2[0] < 0.0, f"FVM delivered drag not opposing: {delivered2}"


def test_backend_swap_is_one_attach_line_and_both_compile():
    """Interchangeability at the graph level (fast): the same Experiment shape,
    with the fluid backend selected by a single `attach()` call, builds clean
    for both backends — the EffectModel swap guarantee."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp_bem, bem_name = _stokeslet_experiment(1e-3, 1e-3, 1000.0, 1e-3)
        gm_bem, h_bem = exp_bem.build()
        exp_fvm, fvm_name = _fvm_experiment(0.1, 0.005, 1.0, 0.1)
        gm_fvm, h_fvm = exp_fvm.build()

    assert h_bem[0].node_names == (bem_name,)
    assert h_fvm[0].node_names == (fvm_name,)
    assert "body" in gm_bem.node_names and bem_name in gm_bem.node_names
    assert "body" in gm_fvm.node_names and fvm_name in gm_fvm.node_names


def test_version_validation_rejects_incompatible_experiment():
    """The load-time version check rejects an experiment whose floor exceeds
    the installed MIME version."""
    from mime.effects import IncompatibleMimeVersionError

    with pytest.raises(IncompatibleMimeVersionError):
        Experiment(name="too_new", mime_version_min="99.0.0")
