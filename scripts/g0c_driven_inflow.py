#!/usr/bin/env python3
"""G0-c iliac-Re anchor — DRIVEN-INFLOW sphere drag vs Schiller-Naumann.

The periodic body-force box (p2/a2) cannot reach iliac Re: U_inf is *emergent*
and the box equilibrates on the viscous time L²/ν (~10³ s at ν=1e-4), so it
stalls at Re~27-42. Here U_inf is PRESCRIBED at the inlet, so Re is exact and
steady state is reached on the fast convective time L_z/U. This is the setup the
G0-c plan called for (driven inflow, drag vs SN, <10%).

Setup: uniform freestream U·ẑ past a sphere (radius a) in a non-periodic box
(reuses r6_inlet_outlet's Dirichlet-BC pattern). All six faces carry the
undisturbed freestream U·ẑ (mass-balanced: z_min in, z_max out, lateral normal
flux 0). Drag via surface_integral_force (the p2-validated extraction). Box:
±6a lateral (blockage ~2%), 8a upstream / 16a downstream. Re=100,200 are steady,
axisymmetric (no shedding below ~270) so the steady SN correlation applies.

Pass: |C_D_FVM - C_D_SN(Re)| / C_D_SN < 10% at cpr=4 (gate target).
Resumable jsonl. Run: python scripts/g0c_driven_inflow.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fvm_validation"))
import numpy as np
import jax.numpy as jnp

from mime.nodes.environment.fvm import make_cartesian_mesh_3d
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import IBMBody, surface_integral_force
from mime.nodes.environment.fvm.sdf import sphere_sdf

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "experiments", "schwarz_vessel_helix",
    "output", "g0c_driven_inflow.jsonl"))


def schiller_naumann(Re):
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


def _uniform_bc(mesh, patch, U, axis=2):
    p = mesh.patch(patch)
    u = jnp.zeros((p.owner.size, mesh.dim), dtype=mesh.V.dtype)
    u = u.at[:, axis].set(U)
    F = (U * p.Sf[:, axis]).astype(mesh.V.dtype)   # 0 on lateral faces (Sf_z=0)
    return VelocityBC(u_wall=u, F_through=F)


def run_case(*, Re, a=0.05, U=0.1, cpr=4, up=8.0, down=16.0, half=6.0,
             dt=0.01, n_chunks=12, n_per_chunk=300, ibm_alpha=1e5):
    nu = U * 2 * a / Re                       # exact: Re = U·2a/nu
    Lz = (up + down) * a
    Lxy = 2 * half * a
    Nz = int(round(cpr * Lz / a))
    Nxy = int(round(cpr * Lxy / a))
    mesh = make_cartesian_mesh_3d(
        Nxy, Nxy, Nz, Lxy, Lxy, Lz,
        origin=(-Lxy / 2, -Lxy / 2, -up * a),
        periodic_x=False, periodic_y=False, periodic_z=False)
    dx = mesh.cartesian_spacing[0]
    centre = jnp.zeros(3, dtype=jnp.float32)   # sphere at origin (up·a from inlet)

    def sph(x):
        return sphere_sdf(x, center=centre, radius=a)
    sphere = IBMBody(name="sphere", sdf=sph, ref_point=centre)

    # Inlet + lateral freestream are Dirichlet U·ẑ; OUTLET is zero-gradient
    # (u_wall=None) so it absorbs the through-flux — prescribing Dirichlet on all
    # six faces over-constrains the incompressible solve (flux compatibility) -> NaN.
    bcs = {p: _uniform_bc(mesh, p, U, axis=2)
           for p in ("x_min", "x_max", "y_min", "y_max", "z_min")}
    bcs["z_max"] = VelocityBC(u_wall=None, F_through=None)
    cfg = PisoConfig(nu=nu, rho=1.0, gamma_conv=1.0, n_corrector=2,
                     pressure_bc="neumann", velocity_bc="dirichlet",
                     ibm_alpha=ibm_alpha, ibm_eps=1.0 * dx)

    state = None
    for _ in range(n_chunks):
        state = run_piso(mesh, bcs, cfg, n_steps=n_per_chunk, dt=dt,
                         body_force_fn=None, ibm_bodies=[sphere], initial=state)
    state["u"].block_until_ready()

    F_si, _ = surface_integral_force(
        state["u"], state["p"], mesh, sph, mu=cfg.rho * cfg.nu, dx=dx,
        shell_inner=1.5, shell_outer=3.5, ref_point=centre)
    F_x = float(F_si[2])                        # drag is along the flow (z)
    # freestream sanity: mean u_z in a far upstream slab (z < -4a, |xy|>2a)
    xx = np.asarray(mesh.x)
    far = (xx[:, 2] < -4 * a) & (np.hypot(xx[:, 0], xx[:, 1]) > 2 * a)
    U_far = float(np.mean(np.asarray(state["u"][:, 2])[far]))
    C_D = F_x / (0.5 * cfg.rho * U ** 2 * np.pi * a ** 2)
    return F_x, C_D, U_far, nu, dx, int(mesh.N_cells)


def _done(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); d[r["key"]] = r
    return d


def main():
    configs = [(4, 100.0), (4, 200.0), (6, 200.0)]
    done = _done(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"results -> {OUT} ({len(done)} done)", flush=True)
    for cpr, Re in configs:
        key = f"cpr{cpr}:Re{int(Re)}"
        if key in done:
            print(f"  [skip] {key}", flush=True); continue
        t0 = time.time()
        try:
            F_x, C_D, U_far, nu, dx, ncell = run_case(Re=Re, cpr=cpr)
        except Exception as e:
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
            continue
        C_SN = schiller_naumann(Re)
        err = abs(C_D - C_SN) / C_SN
        rec = {"key": key, "cpr": cpr, "Re": Re, "F_drag": F_x, "C_D_FVM": C_D,
               "C_D_SN": C_SN, "err": err, "U_far": U_far, "nu": nu,
               "n_cells": ncell, "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  cpr={cpr} Re={Re:.0f} (U_far={U_far:.3f}): C_D_FVM={C_D:.3f} "
              f"C_D_SN={C_SN:.3f}  err={err*100:5.1f}%  "
              f"{'PASS' if err < 0.10 else 'FAIL'}  ({rec['wall_s']:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
