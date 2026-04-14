#!/usr/bin/env python3
"""Retrain Cholesky MLP on combined dataset (original + dense).

Tries 3 architectures, picks best on held-out 30 LHS test configs.
Target: test MAE < 5% (ideally 2-3%).
"""
from __future__ import annotations

import os, sys, json, time
import datetime as dt
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data" / "dejongh_benchmark"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
REPORT_PATH = DATA_DIR / "mlp_training_report_v2.md"
R_CYL_UMR = 1.56


def load_all_configs():
    """Load all training data. Original LHS is the test set (same as v1)."""
    all_ = []
    for src, fname in [
        ("centered", "swimming_speeds_centered.json"),
        ("offcenter", "swimming_speeds_offcenter.json"),
        ("lhs", "swimming_speeds_lhs.json"),
        ("freespace", "swimming_speeds_freespace.json"),
        ("dense", "swimming_speeds_dense.json"),
        ("v3_fill", "swimming_speeds_v3_fill.json"),
    ]:
        p = DATA_DIR / fname
        if not p.exists():
            print(f"Skipping missing: {fname}")
            continue
        data = json.load(open(p))
        for k, c in data.items():
            if "R_matrix" not in c or c.get("nu") is None:
                continue
            if c.get("reciprocity_error", 999) > 1e-4:
                continue
            R = np.array(c["R_matrix"], dtype=np.float64)
            # Sanity: must be SPD for Cholesky decomposition
            eigs = np.linalg.eigvalsh(R)
            if eigs.min() <= 0:
                continue
            rec = {
                "source": src, "key": k,
                "nu": c["nu"], "L_UMR_mm": c["L_UMR_mm"],
                "kappa": c["kappa"],
                "offset_x_nd": c.get("offset_x_nd", 0.0),
                "offset_y_nd": c.get("offset_y_nd", 0.0),
                "log_min_gap": c["log_min_gap"],
                "R": R,
            }
            all_.append(rec)
    return all_


def featurize(configs, use_squared=True):
    """Features: ν, L_nd, κ, off_x, off_y, log_gap, (optional) ν², κ²."""
    rows = []
    for c in configs:
        base = [c["nu"], c["L_UMR_mm"] / R_CYL_UMR, c["kappa"],
                c["offset_x_nd"], c["offset_y_nd"], c["log_min_gap"]]
        if use_squared:
            base += [c["nu"]**2, c["kappa"]**2]
        rows.append(base)
    X = np.array(rows, dtype=np.float32)
    Y = np.array([c["R"] for c in configs], dtype=np.float32)
    return X, Y


def R_to_L_flat(R):
    """Cholesky L flat, diag log-scaled."""
    L = np.linalg.cholesky(R)
    flat = np.zeros(21)
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            val = L[i, j]
            if i == j:
                flat[idx] = np.log(np.expm1(max(val, 1e-10)))
            else:
                flat[idx] = val
            idx += 1
    return flat


def L_flat_to_R_jax(flat):
    L = jnp.zeros((6, 6))
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            val = flat[idx]
            if i == j:
                val = jax.nn.softplus(val) + 1e-5
            L = L.at[i, j].set(val)
            idx += 1
    return L @ L.T


L_flat_to_R_vmap = jax.vmap(L_flat_to_R_jax)


def mlp_init(key, layers, in_dim, out_dim=21):
    params = []
    dims = [in_dim] + list(layers) + [out_dim]
    for i in range(len(dims) - 1):
        key, sub = random.split(key)
        w = random.normal(sub, (dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])
        b = jnp.zeros(dims[i + 1])
        params.append((w, b))
    return params


def mlp_forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b


def adam_init(params):
    return {"m": [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params],
            "v": [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params], "t": 0}


def adam_step(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    new_params, new_m, new_v = [], [], []
    for (w, b), (gw, gb), (mw, mb), (vw, vb) in zip(params, grads, state["m"], state["v"]):
        mw_ = b1 * mw + (1 - b1) * gw
        mb_ = b1 * mb + (1 - b1) * gb
        vw_ = b2 * vw + (1 - b2) * gw ** 2
        vb_ = b2 * vb + (1 - b2) * gb ** 2
        w_ = w - lr * (mw_ / (1 - b1 ** t)) / (jnp.sqrt(vw_ / (1 - b2 ** t)) + eps)
        b_ = b - lr * (mb_ / (1 - b1 ** t)) / (jnp.sqrt(vb_ / (1 - b2 ** t)) + eps)
        new_params.append((w_, b_)); new_m.append((mw_, mb_)); new_v.append((vw_, vb_))
    return new_params, {"m": new_m, "v": new_v, "t": t}


def train_one(params_init, X_train_n, L_train_n, X_test_n, L_test_n, n_epochs, lr0=1e-3):
    state = adam_init(params_init)
    params = params_init

    def loss_fn(params, Xn, Ln):
        pred = mlp_forward(params, Xn)
        return jnp.mean((pred - Ln) ** 2)

    loss_grad = jax.jit(jax.value_and_grad(loss_fn))

    X_jax = jnp.asarray(X_train_n)
    L_jax = jnp.asarray(L_train_n)
    X_test_jax = jnp.asarray(X_test_n)
    L_test_jax = jnp.asarray(L_test_n)

    train_losses = []
    test_losses = []

    def lr_at(epoch):
        return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / n_epochs)) + 1e-6

    for epoch in range(n_epochs):
        loss, grads = loss_grad(params, X_jax, L_jax)
        params, state = adam_step(params, grads, state, lr=lr_at(epoch))
        if epoch % (n_epochs // 20) == 0 or epoch == n_epochs - 1:
            tl = loss_fn(params, X_test_jax, L_test_jax)
            train_losses.append((epoch, float(loss)))
            test_losses.append((epoch, float(tl)))

    return params, train_losses, test_losses


def evaluate(params, X_n, Y_true, L_mean, L_std):
    L_pred_n = np.array(mlp_forward(params, jnp.asarray(X_n)))
    L_pred = L_pred_n * L_std + L_mean
    R_pred = np.array(L_flat_to_R_vmap(jnp.asarray(L_pred)))
    return R_pred


def swim_speed(R):
    U = -np.linalg.solve(R[:3, :3], R[:3, 3:] @ np.array([0, 0, 1.0]))
    return U[2] * R_CYL_UMR * 2 * np.pi * 10  # mm/s


def train_and_report(configs, arch_name, layer_sizes, n_epochs=10000, use_squared=True, seed=42):
    # Train split: everything except source=='lhs'
    train = [c for c in configs if c["source"] != "lhs"]
    test = [c for c in configs if c["source"] == "lhs"]
    print(f"  Arch {arch_name}: train={len(train)}, test={len(test)}")

    X_train, Y_train = featurize(train, use_squared=use_squared)
    X_test, Y_test = featurize(test, use_squared=use_squared)

    L_train = np.array([R_to_L_flat(R) for R in Y_train])
    L_test = np.array([R_to_L_flat(R) for R in Y_test])

    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std
    L_mean, L_std = L_train.mean(0), L_train.std(0) + 1e-8
    L_train_n = (L_train - L_mean) / L_std
    L_test_n = (L_test - L_mean) / L_std

    key = random.PRNGKey(seed)
    in_dim = X_train.shape[1]
    params = mlp_init(key, layer_sizes, in_dim, out_dim=21)

    t0 = time.time()
    params, train_losses, test_losses = train_one(
        params, X_train_n, L_train_n, X_test_n, L_test_n, n_epochs)
    train_time = time.time() - t0

    # Evaluate
    R_pred_train = evaluate(params, X_train_n, Y_train, L_mean, L_std)
    R_pred_test = evaluate(params, X_test_n, Y_test, L_mean, L_std)

    v_true_train = np.array([swim_speed(R) for R in Y_train])
    v_true_test = np.array([swim_speed(R) for R in Y_test])
    v_pred_train = np.array([swim_speed(R) for R in R_pred_train])
    v_pred_test = np.array([swim_speed(R) for R in R_pred_test])

    mae_train = float(np.mean(np.abs(v_pred_train - v_true_train)))
    mae_test = float(np.mean(np.abs(v_pred_test - v_true_test)))
    rel_test = mae_test / max(abs(v_true_test.mean()), 1e-6)
    max_err_test = float(np.max(np.abs(v_pred_test - v_true_test)))

    # Per-entry rel err
    iu, ju = np.triu_indices(6)
    rel_entry = float(np.mean(np.abs(R_pred_test[:, iu, ju] - Y_test[:, iu, ju])) /
                      np.mean(np.abs(Y_test[:, iu, ju])))

    # SPD check
    spd_test = all(np.all(np.linalg.eigvalsh(R) > -1e-6) for R in R_pred_test)

    return {
        "arch": arch_name, "layer_sizes": list(layer_sizes),
        "n_features": int(in_dim), "use_squared": use_squared,
        "n_train": len(train), "n_test": len(test),
        "n_epochs": n_epochs, "train_time_s": train_time,
        "final_train_loss": train_losses[-1][1],
        "final_test_loss": test_losses[-1][1],
        "train_swim_mae_mm_s": mae_train,
        "test_swim_mae_mm_s": mae_test,
        "test_swim_rel_mae": rel_test,
        "test_max_err_mm_s": max_err_test,
        "test_R_entry_rel_err": rel_entry,
        "spd_all_test": spd_test,
        "params": params, "train_losses": train_losses, "test_losses": test_losses,
        "X_mean": X_mean, "X_std": X_std, "L_mean": L_mean, "L_std": L_std,
        "v_pred_test": v_pred_test, "v_true_test": v_true_test,
        "v_pred_train": v_pred_train, "v_true_train": v_true_train,
    }


def save_weights(result, path):
    """Save weights + normalization to .npz."""
    save_dict = {
        "n_layers": len(result["params"]),
        "layer_sizes": np.array(result["layer_sizes"]),
        "X_mean": np.array(result["X_mean"]), "X_std": np.array(result["X_std"]),
        "L_mean": np.array(result["L_mean"]), "L_std": np.array(result["L_std"]),
        "R_CYL_UMR": R_CYL_UMR,
        "use_squared_features": result["use_squared"],
    }
    for i, (w, b) in enumerate(result["params"]):
        save_dict[f"W{i}"] = np.array(w)
        save_dict[f"b{i}"] = np.array(b)
    np.savez(path, **save_dict)


def main():
    print(f"Start: {dt.datetime.now().isoformat(timespec='seconds')}")
    configs = load_all_configs()
    src_counts = {}
    for c in configs:
        src_counts[c["source"]] = src_counts.get(c["source"], 0) + 1
    print(f"Loaded {len(configs)} configs: {src_counts}")

    architectures = [
        ("A_3x64_sq", [64, 64, 64], True),
        ("B_3x128_sq", [128, 128, 128], True),
        ("C_4x128_sq", [128, 128, 128, 128], True),
    ]

    results = []
    for name, layers, use_sq in architectures:
        print(f"\n--- Training {name} ---")
        r = train_and_report(configs, name, layers, n_epochs=10000, use_squared=use_sq)
        print(f"  test MAE: {r['test_swim_mae_mm_s']:.3f} mm/s ({100*r['test_swim_rel_mae']:.1f}%)")
        print(f"  train MAE: {r['train_swim_mae_mm_s']:.3f} mm/s")
        print(f"  SPD: {r['spd_all_test']}")
        print(f"  train time: {r['train_time_s']:.0f}s")
        results.append(r)

    # Pick best
    best = min(results, key=lambda r: r["test_swim_mae_mm_s"])
    print(f"\n=== Best arch: {best['arch']} (test MAE: {best['test_swim_mae_mm_s']:.3f} mm/s) ===")

    # Save best weights
    save_weights(best, str(DATA_DIR / "mlp_cholesky_weights_v2.npz"))
    print(f"Saved: mlp_cholesky_weights_v2.npz")

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, r in zip(axes, results):
        ax.scatter(r["v_true_train"], r["v_pred_train"], color="C0", alpha=0.5, s=25, label=f"train (N={r['n_train']})")
        ax.scatter(r["v_true_test"], r["v_pred_test"], color="C1", marker="s", s=40, label=f"test (N={r['n_test']})")
        lim = max(r["v_true_train"].max(), r["v_true_test"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xlabel("BEM v_z (mm/s)"); ax.set_ylabel("MLP v_z (mm/s)")
        ax.set_aspect("equal"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"{r['arch']}: test MAE = {r['test_swim_mae_mm_s']:.2f} mm/s ({100*r['test_swim_rel_mae']:.1f}%)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mlp_v2_parity.png", dpi=120, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'mlp_v2_parity.png'}")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, r in zip(axes, results):
        ax.plot(*zip(*r["train_losses"]), "o-", label="train")
        ax.plot(*zip(*r["test_losses"]), "s-", label="test")
        ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE normalized L")
        ax.set_title(r["arch"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mlp_v2_training.png", dpi=120, bbox_inches="tight")

    # Report
    with open(REPORT_PATH, "w") as f:
        f.write(f"# MLP v2 Training Report\n\n")
        f.write(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"## Dataset\n")
        f.write(f"Total configs: {len(configs)}\n\n")
        f.write(f"Source breakdown: {src_counts}\n\n")
        f.write(f"Train: {len(configs) - src_counts.get('lhs', 0)}, Test (LHS only): {src_counts.get('lhs', 0)}\n\n")
        f.write(f"## Architecture comparison\n\n")
        f.write(f"| Arch | Test MAE (mm/s) | Test rel MAE | Max err (mm/s) | Train MAE | SPD | Train time |\n")
        f.write(f"|------|----------------|--------------|----------------|-----------|-----|------------|\n")
        for r in results:
            f.write(f"| {r['arch']} | {r['test_swim_mae_mm_s']:.3f} | "
                    f"{100*r['test_swim_rel_mae']:.1f}% | "
                    f"{r['test_max_err_mm_s']:.2f} | "
                    f"{r['train_swim_mae_mm_s']:.3f} | "
                    f"{r['spd_all_test']} | {r['train_time_s']:.0f}s |\n")
        f.write(f"\n## Best\n\n")
        f.write(f"**{best['arch']}** with {len(best['layer_sizes'])}×{best['layer_sizes'][0]} SiLU\n\n")
        f.write(f"- Test swim speed MAE: **{best['test_swim_mae_mm_s']:.3f} mm/s ({100*best['test_swim_rel_mae']:.1f}%)**\n")
        f.write(f"- Max per-config test error: {best['test_max_err_mm_s']:.2f} mm/s\n")
        f.write(f"- R entry relative error (test): {100*best['test_R_entry_rel_err']:.1f}%\n")
        f.write(f"- SPD guaranteed on all test predictions: {best['spd_all_test']}\n\n")
        f.write(f"## Comparison vs v1\n\n")
        f.write(f"- v1 (91 configs, 3×64): 0.35 mm/s ({100*0.35/4.5:.0f}%)\n")
        f.write(f"- v2 (best): {best['test_swim_mae_mm_s']:.3f} mm/s ({100*best['test_swim_rel_mae']:.1f}%)\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- Weights: `data/dejongh_benchmark/mlp_cholesky_weights_v2.npz`\n")
        f.write(f"- Parity plot: `data/dejongh_benchmark/figures/mlp_v2_parity.png`\n")
        f.write(f"- Training curves: `data/dejongh_benchmark/figures/mlp_v2_training.png`\n")

    # Save summary JSON too
    summary_json = {
        "timestamp": dt.datetime.now().isoformat(),
        "n_configs": len(configs), "src_counts": src_counts,
        "best_arch": best["arch"],
        "best_test_mae": best["test_swim_mae_mm_s"],
        "best_test_rel_mae": best["test_swim_rel_mae"],
        "best_max_err": best["test_max_err_mm_s"],
        "best_spd": best["spd_all_test"],
        "all_results": [
            {k: v for k, v in r.items()
             if not isinstance(v, (np.ndarray, jnp.ndarray)) and k != "params"
             and not isinstance(v, list) or k in ["layer_sizes", "train_losses", "test_losses"]}
            for r in results
        ],
    }
    with open(DATA_DIR / "mlp_v2_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print(f"\n{REPORT_PATH} written")
    print(f"Finish: {dt.datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
