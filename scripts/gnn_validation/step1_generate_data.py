"""Step 1 — generate (fine, coarse) training data for the GNN sprint.

Three steady-inlet train configs + one held-out val config. For each:
  1. Build fine mesh (cpr_fine), run PISO to convergence, save state.
  2. Build coarse mesh (cpr_coarse), run PISO to convergence, save state.
  3. Downsample u_fine, p_fine to the coarse-mesh resolution by averaging
     the fine cells that fall inside each coarse cell.
  4. Save K_FVM (drag) for each so improvement can be measured later.

Output: data/gnn_training/<label>_{fine,coarse,fine_downsampled}.npz
        data/gnn_training/manifest.json (config + drag metrics)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, run_piso
from mime.nodes.environment.fvm.ibm import (
    IBMBody, surface_integral_force, momentum_deficit_drag,
)
from mime.nodes.environment.fvm.sdf import sphere_sdf


def happel_brenner(lam):
    return 1.0/(1.0-2.10443*lam+2.08877*lam**3-0.94813*lam**5
                -1.372*lam**6+3.87*lam**8-4.19*lam**10)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
TRAIN_CONFIGS = [
    dict(label="train_A", lambda_=0.2, Re=50,  cpr_fine=8, cpr_coarse=4),
    dict(label="train_B", lambda_=0.3, Re=100, cpr_fine=8, cpr_coarse=4),
    dict(label="train_C", lambda_=0.2, Re=200, cpr_fine=8, cpr_coarse=4),
]
VAL_CONFIGS = [
    # Held out by Re (train_B is λ=0.3 Re=100). Brief originally
    # specified λ=0.25 but that gives a non-integer fine/coarse mesh
    # ratio (1.97×), breaking block-mean downsampling. λ=0.3 keeps
    # the 2:1 ratio while remaining held out from training.
    dict(label="val_A", lambda_=0.3, Re=150, cpr_fine=8, cpr_coarse=4),
]

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gnn_training"


def downsample_to_coarse(field_fine, mesh_fine, mesh_coarse):
    """Average fine-mesh cells that fall inside each coarse cell.

    Both meshes share the same domain (same origin and extents). For
    each coarse cell, find which fine cells are inside it (by integer
    index ratios) and take the mean of ``field_fine`` over them.
    """
    Nx_f, Ny_f, Nz_f = mesh_fine.cartesian_shape
    Nx_c, Ny_c, Nz_c = mesh_coarse.cartesian_shape
    rx, ry, rz = Nx_f // Nx_c, Ny_f // Ny_c, Nz_f // Nz_c
    assert Nx_f == rx * Nx_c and Ny_f == ry * Ny_c and Nz_f == rz * Nz_c, (
        "Fine and coarse mesh shapes must be integer multiples; got "
        f"fine={mesh_fine.cartesian_shape}, coarse={mesh_coarse.cartesian_shape}"
    )
    if field_fine.ndim == 1:
        f3 = np.asarray(field_fine).reshape(Nx_f, Ny_f, Nz_f)
        # block-mean
        f3 = f3.reshape(Nx_c, rx, Ny_c, ry, Nz_c, rz).mean(axis=(1, 3, 5))
        return f3.reshape(-1)
    else:
        f3 = np.asarray(field_fine).reshape(Nx_f, Ny_f, Nz_f, -1)
        f3 = f3.reshape(Nx_c, rx, Ny_c, ry, Nz_c, rz, -1).mean(axis=(1, 3, 5))
        return f3.reshape(-1, f3.shape[-1])


def run_one(*, lambda_, Re, cpr, label, n_steps=400, U_dc=None):
    """Build mesh + bodies + PISO config, run to (steady) convergence."""
    r_b = 1e-3
    R_pipe = r_b / lambda_
    sphere_margin = 5.0; bc_margin = 5.0
    L_pipe = 2.0 * (sphere_margin + bc_margin) * r_b + 2.0 * r_b
    nu = 1e-3
    rho = 1.0
    mu = rho * nu

    # Re_R = U_dc · R / ν → U_dc = Re · ν / R
    if U_dc is None:
        U_dc = Re * nu / R_pipe

    mesh = make_pipe_mesh(pipe_radius=R_pipe, pipe_length=L_pipe,
                          robot_radius=r_b, cpr=cpr)
    dx = mesh.cartesian_spacing[0]
    Nz = mesh.cartesian_shape[2]
    L_actual = Nz * dx
    sphere_centre = jnp.array([0.0, 0.0, L_actual / 2], dtype=mesh.V.dtype)

    def pipe_wall_sdf(x):
        rxy = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-30)
        return R_pipe - rxy
    def sphere_sdf_fn(x):
        return sphere_sdf(x, center=sphere_centre, radius=r_b)
    bodies = [
        IBMBody(name="pipe_wall", sdf=pipe_wall_sdf),
        IBMBody(name="sphere",    sdf=sphere_sdf_fn),
    ]

    bcs = {}
    for name in ("x_min","x_max","y_min","y_max","z_min","z_max"):
        nb = int(mesh.patch(name).owner.size)
        bcs[name] = VelocityBC(u_wall=jnp.zeros((nb,3)),
                                F_through=jnp.zeros((nb,)))

    cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2)

    dt = min(0.5, 0.5 * dx / max(2 * U_dc, 1e-30))
    t0 = time.time()
    state = run_piso(mesh, bcs, cfg, n_steps=n_steps, dt=dt,
                     body_force_fn=None, ibm_bodies=bodies, lifting=L_lift)
    state["u"].block_until_ready()
    wall = time.time() - t0

    # Force extraction: surface_integral with shell (0.5, 2.5)
    u_phys = state["u"] + L_lift.u_lift_static
    p_lift_fn = make_poiseuille_p_lift(mu=mu, U_mean=U_dc, pipe_radius=R_pipe)
    F_si_vec, _ = surface_integral_force(
        u_phys, state["p"], mesh, sphere_sdf_fn,
        mu=mu, dx=dx, shell_inner=0.5, shell_outer=2.5,
        ref_point=sphere_centre, p_lift_fn=p_lift_fn, pipe_axis=2,
    )
    F_si = float(F_si_vec[2])
    F_uncon = 6.0 * np.pi * mu * r_b * (2 * U_dc)
    K_FVM = F_si / F_uncon
    K_h = happel_brenner(lambda_)
    return dict(
        mesh=mesh, state=state, u_phys=u_phys,
        L_lift=L_lift, sphere_centre=sphere_centre,
        cells=mesh.N_cells, wall_s=wall, dt=dt,
        F_z=F_si, K_FVM=K_FVM, K_Happel=K_h,
        U_dc=U_dc, R_pipe=R_pipe, r_b=r_b, mu=mu,
        L_pipe_actual=L_actual,
    )


def save_state_npz(path: Path, state: dict, u_phys, mesh):
    np.savez_compressed(
        path,
        u=np.asarray(u_phys),
        p=np.asarray(state["p"]),
        cartesian_shape=np.asarray(mesh.cartesian_shape, dtype=np.int32),
        cartesian_spacing=np.asarray(mesh.cartesian_spacing, dtype=np.float32),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Step 1 — generate fine + coarse training data")
    print("=" * 78)

    manifest = []
    for cfg in TRAIN_CONFIGS + VAL_CONFIGS:
        label   = cfg["label"]
        lambda_ = cfg["lambda_"]
        Re      = cfg["Re"]
        cpr_f   = cfg["cpr_fine"]
        cpr_c   = cfg["cpr_coarse"]
        print(f"\n--- {label} (λ={lambda_}, Re={Re}) ---")

        fine = run_one(lambda_=lambda_, Re=Re, cpr=cpr_f, label=label)
        coarse = run_one(lambda_=lambda_, Re=Re, cpr=cpr_c, label=label,
                          U_dc=fine["U_dc"])

        # Downsample fine to coarse resolution
        u_fine_ds = downsample_to_coarse(
            fine["u_phys"], fine["mesh"], coarse["mesh"],
        )
        p_fine_ds = downsample_to_coarse(
            fine["state"]["p"], fine["mesh"], coarse["mesh"],
        )

        save_state_npz(OUT_DIR / f"{label}_fine.npz",
                        fine["state"], fine["u_phys"], fine["mesh"])
        save_state_npz(OUT_DIR / f"{label}_coarse.npz",
                        coarse["state"], coarse["u_phys"], coarse["mesh"])
        # The downsampled fine reference shares the coarse mesh
        np.savez_compressed(
            OUT_DIR / f"{label}_fine_downsampled.npz",
            u=u_fine_ds, p=p_fine_ds,
            cartesian_shape=np.asarray(coarse["mesh"].cartesian_shape,
                                       dtype=np.int32),
            cartesian_spacing=np.asarray(coarse["mesh"].cartesian_spacing,
                                         dtype=np.float32),
        )

        # Coarse vs fine drag error (matched-reference K)
        rel_err = abs(coarse["K_FVM"] - fine["K_FVM"]) / abs(fine["K_FVM"])
        print(f"  Fine   : {fine['cells']:>8d} cells, drag = {fine['F_z']:.4e} N, "
              f"K_FVM = {fine['K_FVM']:.4f}, K_Happel = {fine['K_Happel']:.4f}")
        print(f"  Coarse : {coarse['cells']:>8d} cells, drag = {coarse['F_z']:.4e} N, "
              f"K_FVM = {coarse['K_FVM']:.4f}, K_Happel = {coarse['K_Happel']:.4f}")
        print(f"  Coarse error vs fine: {rel_err*100:.2f}%")
        print(f"  Wall time: fine {fine['wall_s']:.1f}s, coarse {coarse['wall_s']:.1f}s")

        manifest.append(dict(
            label=label, lambda_=lambda_, Re=Re,
            cpr_fine=cpr_f, cpr_coarse=cpr_c,
            U_dc=fine["U_dc"], R_pipe=fine["R_pipe"], r_b=fine["r_b"],
            mu=fine["mu"], L_pipe=fine["L_pipe_actual"],
            cells_fine=fine["cells"], cells_coarse=coarse["cells"],
            cartesian_shape_fine=list(fine["mesh"].cartesian_shape),
            cartesian_shape_coarse=list(coarse["mesh"].cartesian_shape),
            F_z_fine=fine["F_z"], F_z_coarse=coarse["F_z"],
            K_FVM_fine=fine["K_FVM"], K_FVM_coarse=coarse["K_FVM"],
            K_Happel=fine["K_Happel"],
            coarse_vs_fine_err_pct=rel_err * 100,
            wall_s_fine=fine["wall_s"], wall_s_coarse=coarse["wall_s"],
        ))

    # Save manifest
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n" + "=" * 78)
    print("Step 1 complete — manifest saved.")
    print("=" * 78)


if __name__ == "__main__":
    main()
