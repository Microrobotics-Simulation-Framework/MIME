"""Spike W2F, the MAC grid's arms: MIME's resolved-body flow node (IMEX Euler, a penalty sphere held by feeding its pose
and a zero twist) on CORE spike 2's case (``probe/c2/macref.py``, generalised): a periodic box ``1.5 D x 1.5 D x 4 D``
holding the tube, the sphere of radius 0.4 R on its axis offset by (0.31, 0.17, 0.23) cells, a Poiseuille start.

A configuration is ``<cells across D>:<Re_D>:<drive>:<t_end ms>[:nu105]``; drive ``force`` (spike 2's fixed body force
at the nominal Poiseuille gradient) or ``flux`` (the body force set every step by a PI controller holding the lumen's
flux at the nominal one, read from the node's own ``lumen_flux``). Step ``0.25 h / (3 U)``, spike 2's rule.
Results: ``<tag>.npz`` (every step: the body's load, its displaced share, the lumen's mean velocity, the drive) and
``<tag>.json`` in ``W2F_OUT``."""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("W2F_OUT", os.path.join(os.path.dirname(HERE), "out"))
NU, RHO, RADIUS = 3.3e-6, 1060.0, 1.75e-3
A = 0.4 * RADIUS


def run(config, budget_s):
    import jax
    import jax.numpy as jnp

    from mime.core import run_dtype
    from mime.nodes.flow.fvm import band_named
    from mime.nodes.flow.fvm._resolved import window_cells
    from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode, ResolvedSpec
    parts = config.split(":")
    cells, re, drive, t_end = int(parts[0]), float(parts[1]), parts[2], float(parts[3]) * 1e-3
    nu_scale = 1.05 if "nu105" in parts[4:] else 1.0
    tag = "mac_" + config.replace(":", "_")
    dtype = run_dtype()
    h = 2 * RADIUS / cells
    nx = cells + cells // 2
    nz = 4 * cells
    U = re * NU / (2 * RADIUS)
    dt = 0.25 * h / (3.0 * U)
    chunk_steps = 25
    steps = int(math.ceil(t_end / dt / chunk_steps)) * chunk_steps
    band = band_named("smooth_2h_0.3h")
    spec = ResolvedSpec("ball", "drag:ball", "sphere", {"centre": (0.0, 0.0, 0.0), "radius": A}, (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), window_cells(A, h, band), 4.0 / 3.0 * math.pi * A ** 3)
    centre = (0.5 * nx * h, 0.5 * nx * h)
    x = (np.arange(nx) + 0.5) * h - centre[0]
    r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2
    w0 = np.broadcast_to(2.0 * U * np.clip(1.0 - r2 / RADIUS ** 2, 0.0, None), (nx, nx, nz)).copy()
    node = ResolvedFlowNode("flow", dt, cells_x=nx, cells_y=nx, cells_z=nz, spacing=h, viscosity=NU * nu_scale * RHO, density=RHO,
                            tube_centre=(centre[0], centre[1], 0.0), tube_axis=(0.0, 0.0, 1.0), tube_radius=RADIUS,
                            scheme="imex_euler", convection=True, bodies=[spec], initial_velocity_z=w0)
    pose = jnp.asarray([centre[0] + 0.31 * h, centre[1] + 0.17 * h, 0.5 * nz * h + 0.23 * h, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
    mass = RHO * 4.0 / 3.0 * math.pi * A ** 3
    mass6 = jnp.asarray(np.diag([mass] * 3 + [0.4 * mass * A ** 2] * 3), dtype=dtype)
    f0 = 8.0 * NU * U / RADIUS ** 2
    base = {"body_0_pose": pose, "body_0_twist": jnp.zeros(6, dtype=dtype), "body_0_mass": mass6}
    unfed = [k for k in node.own_inputs() if k not in base and k != "body_force"]
    area = math.pi * RADIUS ** 2
    tau = 2e-3
    kp, ki = 2.0 / tau, 1.0 / tau ** 2
    control = drive == "flux"

    def one(c, _):
        state, drv, integ = c
        inputs = dict(base, body_force=jnp.stack([jnp.zeros((), dtype), jnp.zeros((), dtype), drv]))
        new = node.update(state, inputs, dt)
        new = {**state, **new}
        um = new["lumen_flux"] / area
        if control:
            e = U - um
            integ = integ + e * dt
            drv = f0 + kp * e + ki * integ
        return (new, drv, integ), (new["body_0_load"], new["body_0_share"], um, drv)

    chunk = jax.jit(lambda c: jax.lax.scan(one, c, None, length=chunk_steps))
    c = (dict(node.initial_state()), jnp.asarray(f0, dtype), jnp.zeros((), dtype))
    loads, shares, means, drives, walls = [], [], [], [], []
    t0 = time.time()
    info = {"tag": tag, "re": re, "drive": drive, "cells_across": cells, "box": [nx, nx, nz], "h": h, "dt": dt, "steps_planned": steps,
            "U": U, "f0": f0, "unfed_inputs": unfed, "cells_a_radius": A / h, "nu_scale": nu_scale, "dtype": str(dtype)}
    print(json.dumps(info), flush=True)
    for _ in range(steps // chunk_steps):
        t1 = time.time()
        c, (ld, sh, mn, dv) = chunk(c)
        jax.block_until_ready(ld)
        walls.append(time.time() - t1)
        loads.append(np.asarray(ld)), shares.append(np.asarray(sh)), means.append(np.asarray(mn)), drives.append(np.asarray(dv))
        if not np.isfinite(loads[-1]).all() or time.time() - t0 > budget_s:
            break
    loads, shares, means, drives = (np.concatenate(v) for v in (loads, shares, means, drives))
    t = (np.arange(len(loads)) + 1) * dt
    state = c[0]
    info.update(steps_done=int(len(loads)), t_done=float(t[-1]), s_per_step=float(np.median(walls[1:]) / chunk_steps) if len(walls) > 1 else None,
                stopped_at=float(np.asarray(state.get("stopped_at", -1))), fz_load_end=float(loads[-1, 2]), fz_share_end=float(shares[-1, 2]),
                um_end=float(means[-1]), drive_end=float(drives[-1]), finite=bool(np.isfinite(loads).all()), total_s=time.time() - t0,
                device=str(jax.devices()[0]))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), t=t, load=loads, share=shares, um=means, drive=drives, dt=dt)
    with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    jax.clear_caches()


def main():
    configs = os.environ["W2F_CONFIGS"].split(",")
    budget = float(os.environ.get("W2F_BUDGET_S", "17000"))
    for config in configs:
        try:
            run(config, budget / len(configs))
        except Exception as e:  # noqa: BLE001
            import traceback
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, "mac_" + config.replace(":", "_") + ".failed.json"), "w", encoding="utf-8") as f:
                json.dump({"config": config, "failed": repr(e)[:500], "trace": traceback.format_exc()[-3000:]}, f)
            print("FAILED", config, repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
