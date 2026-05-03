"""Step 2 — train the GNN flux corrector on the local 3 train configs.

For each config: load (coarse_state, fine_downsampled), build mesh +
cfg matching the manifest, run N_inner=5 PISO steps with the corrector
injected as a per-cell body force, MSE the result against the fine
reference, backprop into the corrector weights via optax.adam.

Pass criteria:
  - loss decreases over 50 epochs (monotonically modulo small jitter)
  - final loss < 10% of initial
  - no NaN at any epoch
  - completes in < 30 min on RTX 2060
"""
from __future__ import annotations
import json
import time
import pickle
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
import optax

from mime.nodes.environment.fvm import (
    make_pipe_mesh, make_poiseuille_lift, make_poiseuille_p_lift,
    init_gnn_flux_corrector,
)
from mime.nodes.environment.fvm.boundary import VelocityBC
from mime.nodes.environment.fvm.piso import PisoConfig, make_piso_step
from mime.nodes.environment.fvm.ibm import IBMBody
from mime.nodes.environment.fvm.sdf import sphere_sdf


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gnn_training"


def build_mesh_and_step(cfg_dict, *, coarse=True):
    """Re-create the coarse-mesh PISO step for one manifest entry.

    Returns (mesh, bcs, piso_cfg, ibm_bodies, lifting, dt).
    """
    r_b = cfg_dict["r_b"]
    R_pipe = cfg_dict["R_pipe"]
    cpr = cfg_dict["cpr_coarse"] if coarse else cfg_dict["cpr_fine"]
    L_pipe = cfg_dict["L_pipe"]
    nu = cfg_dict["mu"]   # rho=1.0
    rho = 1.0
    U_dc = cfg_dict["U_dc"]

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
    piso_cfg = PisoConfig(
        nu=nu, rho=rho, gamma_conv=0.0, n_corrector=2,
        pressure_bc="neumann", velocity_bc="dirichlet",
        ibm_alpha=1e5, ibm_eps=1.0 * dx,
    )
    L_lift = make_poiseuille_lift(mesh, R_pipe=R_pipe, U_mean=U_dc, axis=2)
    dt = min(0.5, 0.5 * dx / max(2 * U_dc, 1e-30))
    return mesh, bcs, piso_cfg, bodies, L_lift, dt, sphere_centre


def correction_body_force(corrector, u, p, mesh, rho):
    """Per-cell body force from GNN correction (matches
    GNNFluxCorrectedFVMNode.compute_correction_force)."""
    delta_u_face = corrector.apply(u, p, mesh, correction_weight=1.0)
    Sf = mesh.Sf
    F_face = jnp.einsum("fi,fi->f", delta_u_face, Sf)
    flux = rho * F_face[:, None] * delta_u_face
    out_o = jax.ops.segment_sum(flux, mesh.owner, num_segments=mesh.N_cells)
    out_n = jax.ops.segment_sum(flux, mesh.neighbour, num_segments=mesh.N_cells)
    return -(out_o - out_n) / mesh.V[:, None]


def make_loss_fn(mesh, bcs, piso_cfg, bodies, L_lift, dt, n_inner=5):
    """Build a JIT-able loss function for one training config.

    Closes over mesh / cfg / lifting so only (corrector, init_state,
    fine_ref) vary across calls.
    """
    rho = piso_cfg.rho

    @jax.jit
    def loss(corrector, init_state, fine_ref_u, fine_ref_p):
        # Single body_force_fn closure that recomputes f_gnn each call;
        # in a fori_loop carry the state, recompute per step.
        def step_fn(_, state):
            f_gnn = correction_body_force(
                corrector, state["u"], state["p"], mesh, rho,
            )
            body_force_fn = lambda t: f_gnn
            step = make_piso_step(
                mesh, bcs, piso_cfg,
                body_force_fn=body_force_fn,
                ibm_bodies=bodies, lifting=L_lift,
            )
            return step(state, dt)

        final_state = jax.lax.fori_loop(0, n_inner, step_fn, init_state)
        # u in state is u_hom; physical = u_hom + u_lift
        u_phys = final_state["u"] + L_lift.u_lift_static
        loss_u = jnp.mean((u_phys - fine_ref_u) ** 2) / (
            jnp.mean(fine_ref_u ** 2) + 1e-12
        )
        loss_p = jnp.mean((final_state["p"] - fine_ref_p) ** 2) / (
            jnp.mean(fine_ref_p ** 2) + 1e-12
        )
        return loss_u + loss_p

    return loss


def load_initial_state(label, mesh, L_lift):
    """Load coarse state .npz; convert u_phys back to u_hom for PISO."""
    data = np.load(DATA_DIR / f"{label}_coarse.npz")
    u_phys = jnp.asarray(data["u"], dtype=mesh.V.dtype)
    u_hom = u_phys - L_lift.u_lift_static
    p = jnp.asarray(data["p"], dtype=mesh.V.dtype)
    state = {
        "u": u_hom,
        "u_pre_ibm": u_phys,
        "u_after_explicit": u_phys,
        "p": p,
        "F": jnp.zeros((mesh.N_faces,), dtype=mesh.V.dtype),
        "t": jnp.asarray(0.0, dtype=mesh.V.dtype),
        "i_step": jnp.asarray(0, dtype=jnp.int32),
    }
    return state


def load_fine_ref(label, mesh):
    data = np.load(DATA_DIR / f"{label}_fine_downsampled.npz")
    return (
        jnp.asarray(data["u"], dtype=mesh.V.dtype),
        jnp.asarray(data["p"], dtype=mesh.V.dtype),
    )


def main():
    print("=" * 78)
    print("Step 2 — GNN flux correction training (50 epochs, 3 configs)")
    print("=" * 78)

    with open(DATA_DIR / "manifest.json") as f:
        manifest = json.load(f)

    train_labels = ["train_A", "train_B", "train_C"]
    cfg_by_label = {m["label"]: m for m in manifest}

    # Pre-build per-config (mesh, loss_fn, init_state, fine_ref).
    # Each config has a different mesh shape, so each loss_fn is a
    # separate JIT trace.
    per_cfg = {}
    for lbl in train_labels:
        m, bcs, piso, bodies, L_lift, dt, sphere = build_mesh_and_step(
            cfg_by_label[lbl], coarse=True,
        )
        loss_fn = make_loss_fn(m, bcs, piso, bodies, L_lift, dt, n_inner=5)
        init_state = load_initial_state(lbl, m, L_lift)
        fine_u, fine_p = load_fine_ref(lbl, m)
        per_cfg[lbl] = dict(mesh=m, loss_fn=loss_fn,
                             init_state=init_state,
                             fine_u=fine_u, fine_p=fine_p)
        print(f"  {lbl}: mesh {m.cartesian_shape} ({m.N_cells} cells)")

    # ---- Init corrector and optimiser ----
    # last_layer_scale=1.0 (vs default 1e-3): the body-force projection
    # (compute_correction_force) is quadratic in the corrector output,
    # so a 1e-3 init makes the correction O(1e-6) — well below float32
    # gradient noise. Init at full Glorot scale; the optimiser then
    # tunes from a meaningful baseline. M2 chose 1e-3 for *deployment*
    # bit-equivalence at correction_weight=0; for training we override.
    rng = jax.random.PRNGKey(7)
    corrector = init_gnn_flux_corrector(
        rng, hidden=32, n_rounds=3, last_layer_scale=1.0,
    )
    print(f"  corrector params: {corrector.param_count()}")
    LR = 1e-2
    opt = optax.adam(LR)
    opt_state = opt.init(corrector)

    grad_fns = {
        lbl: jax.jit(jax.value_and_grad(per_cfg[lbl]["loss_fn"]))
        for lbl in train_labels
    }

    # ---- Training loop ----
    N_EPOCHS = 50
    print(f"\n{'epoch':>5} | {'train_A':>10} | {'train_B':>10} | "
          f"{'train_C':>10} | {'mean':>10} | rel/init")
    print("-" * 78)
    initial_mean = None
    history = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        losses = {}
        for lbl in train_labels:
            d = per_cfg[lbl]
            loss, grad = grad_fns[lbl](
                corrector, d["init_state"], d["fine_u"], d["fine_p"],
            )
            updates, opt_state = opt.update(grad, opt_state, corrector)
            corrector = optax.apply_updates(corrector, updates)
            losses[lbl] = float(loss)
        mean_loss = sum(losses.values()) / len(losses)
        if initial_mean is None:
            initial_mean = mean_loss
        rel = mean_loss / initial_mean
        history.append(dict(epoch=epoch, **losses, mean=mean_loss, rel_init=rel))
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"{epoch:>5d} | {losses['train_A']:>10.4e} | "
                  f"{losses['train_B']:>10.4e} | {losses['train_C']:>10.4e} | "
                  f"{mean_loss:>10.4e} | {rel:.4f}")
        if not np.isfinite(mean_loss):
            print(f"  NaN at epoch {epoch} — abort")
            break

    wall = time.time() - t0
    print(f"\n  Wall time: {wall:.0f}s ({wall/60:.1f} min)")

    # ---- Save ----
    final_loss = history[-1]["mean"]
    print(f"  Initial loss: {initial_mean:.4e}")
    print(f"  Final loss  : {final_loss:.4e}")
    print(f"  Reduction   : {initial_mean / final_loss:.2f}× "
          f"({'PASS' if final_loss < 0.1 * initial_mean else 'FAIL'} target <0.1)")

    out_path = DATA_DIR / "gnn_params_local.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(corrector, f)
    with open(DATA_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved corrector → {out_path}")
    print(f"  Saved history    → {DATA_DIR/'training_history.json'}")


if __name__ == "__main__":
    main()
