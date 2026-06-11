"""Stokes-drag demo — the ``make_experiment(params)`` v1.x contract pilot.

ADR-2026-EFFECT-MODEL §7 (C5): the v1.x experiment contract is a
``make_experiment(params: Params) -> Experiment`` factory plus a ``Params``
TypedDict and a ``PARAMS_SCHEMA`` (the validated, defaulted parameter surface a
runner / config layer reads). This module pilots that contract on the one
experiment already validated through the EffectModel surface — the free-space
Stokes-drag backend swap (``tests/verification/test_effectmodel_stokes_drag_swap``):
a kinematic sphere driven at a prescribed velocity, with the hydrodynamic
backend (Stokeslet BEM or FVM+IBM) selected by a single parameter.

It returns an **unbuilt** ``Experiment`` (the caller / runner calls ``build()``),
so the six-pass validation and graph compile happen at the caller's chosen time.

The real directory experiments (umr_confinement / dejongh) adopt this same
contract once their drag chains are EffectModels (the StokesletChain / Schwarz
variants, E6d, M2); this pilot fixes the pattern + the schema/validation shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from mime.effects import Body, Experiment, HydrodynamicModel, Medium

# Installed MIME predates the v0.3.0 tag in the source tree; keep a permissive
# floor so the load-time version check accepts the demo.
_MIME_MIN = "0.1.0"
_FLUID = "fluid"  # canonical fluid-node name (backend-independent)


# ── parameter surface ───────────────────────────────────────────────────────

class Params(TypedDict, total=False):
    """The Stokes-drag demo parameters (all optional; defaults in
    ``PARAMS_SCHEMA``)."""

    backend: str            # "stokeslet" | "fvm"
    radius_m: float         # sphere radius
    viscosity_pa_s: float   # dynamic viscosity
    fluid_density_kg_m3: float
    timestep_s: float


@dataclass(frozen=True)
class ParamSpec:
    """One parameter's type, default, doc, and (optional) allowed choices."""

    type: type
    default: Any
    description: str
    choices: Optional[tuple] = None


PARAMS_SCHEMA: dict[str, ParamSpec] = {
    "backend": ParamSpec(
        str, "stokeslet", "Hydrodynamic backend", choices=("stokeslet", "fvm")),
    "radius_m": ParamSpec(float, 1e-3, "Sphere radius [m]"),
    "viscosity_pa_s": ParamSpec(float, 1e-3, "Dynamic viscosity [Pa·s]"),
    "fluid_density_kg_m3": ParamSpec(float, 1000.0, "Fluid density [kg/m³]"),
    "timestep_s": ParamSpec(float, 1e-3, "Timestep [s]"),
}


def resolve_params(params: Optional[Params] = None) -> dict[str, Any]:
    """Fill defaults from ``PARAMS_SCHEMA`` and validate keys / types / choices.

    Raises ``ValueError`` for an unknown key, a wrong type, or a choice outside
    the allowed set — so a bad config fails loudly before any graph is built."""
    params = dict(params or {})
    unknown = set(params) - set(PARAMS_SCHEMA)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)}; "
            f"known: {sorted(PARAMS_SCHEMA)}"
        )
    out: dict[str, Any] = {}
    for key, spec in PARAMS_SCHEMA.items():
        val = params.get(key, spec.default)
        if spec.type is float and isinstance(val, int):
            val = float(val)
        if not isinstance(val, spec.type):
            raise ValueError(
                f"parameter {key!r} must be {spec.type.__name__}, got "
                f"{type(val).__name__}"
            )
        if spec.choices is not None and val not in spec.choices:
            raise ValueError(
                f"parameter {key!r}={val!r} not in {spec.choices}"
            )
        out[key] = val
    return out


# ── node construction ───────────────────────────────────────────────────────

def _kinematic_body(*, a, mu, rho_f, dt, name="body"):
    from mime.nodes.robot.rigid_body import RigidBodyNode

    return RigidBodyNode(
        name=name, timestep=dt,
        semi_major_axis_m=a, semi_minor_axis_m=a,
        density_kg_m3=1100.0,
        fluid_viscosity_pa_s=mu, fluid_density_kg_m3=rho_f,
        kinematic_mode=True,
    )


def _stokeslet_backend(a, mu, dt):
    from mime.nodes.environment.stokeslet import (
        StokesletFluidNode, sphere_surface_mesh,
    )

    node = StokesletFluidNode(
        name=_FLUID, timestep=dt, mu=mu,
        body_mesh=sphere_surface_mesh(radius=a, n_refine=3),
    )
    return HydrodynamicModel.Stokeslet(node)


def _fvm_backend(a, mu, rho_f, dt):
    import jax.numpy as jnp

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
    wall = IBMBody(
        name="pipe_wall",
        sdf=lambda x: R_pipe - jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30),
    )
    bcs = {}
    for nm in ("x_min", "x_max", "y_min", "y_max"):
        p = mesh.patch(nm)
        nbf = int(p.owner.size)
        bcs[nm] = VelocityBC(u_wall=jnp.zeros((nbf, 3)), F_through=jnp.zeros((nbf,)))
    cfg = PisoConfig(nu=mu / rho_f, rho=rho_f, gamma_conv=0.5, n_corrector=2,
                     pressure_bc=("neumann", "neumann", "periodic"),
                     velocity_bc=("dirichlet", "dirichlet", "periodic"),
                     ibm_alpha=1e5, ibm_eps=1.0 * dx)
    node = FVMFluidNode(
        name=_FLUID, timestep=dt, mesh=mesh, bcs=bcs, cfg=cfg,
        static_bodies=[wall],
        dynamic_body_factories=[
            ("sphere", make_sphere_body_factory("sphere", radius=a))],
    )
    return HydrodynamicModel.FVM(node)


# ── the factory ─────────────────────────────────────────────────────────────

def make_experiment(params: Optional[Params] = None) -> Experiment:
    """Build the Stokes-drag demo Experiment (unbuilt) from ``params``.

    The body is a kinematic sphere driven at a prescribed velocity via the
    graph-external inputs ``external_velocity`` / ``external_angular_velocity``;
    the hydrodynamic backend is selected by ``params["backend"]``. The returned
    Experiment is configured but not built — call ``.build()`` to validate +
    compile."""
    p = resolve_params(params)
    a = p["radius_m"]
    mu = p["viscosity_pa_s"]
    rho_f = p["fluid_density_kg_m3"]
    dt = p["timestep_s"]

    exp = Experiment(name="stokes_drag_demo", mime_version_min=_MIME_MIN)
    exp.set_body(Body(
        "body", node=_kinematic_body(a=a, mu=mu, rho_f=rho_f, dt=dt),
        properties={"hydrodynamic": {}},
    ))
    medium_props: dict[str, Any] = {"density": rho_f, "viscosity": mu}
    if p["backend"] == "stokeslet":
        medium_props["reynolds_number"] = rho_f * 1e-4 * 2 * a / mu
        exp.attach(_stokeslet_backend(a, mu, dt))
    else:
        exp.attach(_fvm_backend(a, mu, rho_f, dt))
    exp.set_medium(Medium(medium_props))

    exp.add_external_input("body", "external_velocity", (3,))
    exp.add_external_input("body", "external_angular_velocity", (3,))
    return exp
