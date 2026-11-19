"""The MAC grid's force history on the sphere case: MIME's resolved-body flow node (IMEX Euler, the penalty body held by
feeding its pose and a zero twist), the same periodic tube, drive and Poiseuille start as the graph's runs.

``C2_MAC_RE`` (260 or 650), ``C2_MAC_CELLS`` (cells across the diameter), ``C2_MAC_TIME`` (s), ``C2_MAC_BUDGET_S``."""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("C2_OUT", os.path.join(os.path.dirname(HERE), "out"))
NU, RHO, RADIUS = 3.3e-6, 1060.0, 1.75e-3
A = 0.4 * RADIUS


def main():
    import jax
    import jax.numpy as jnp

    from mime.core import run_dtype
    from mime.nodes.flow.fvm import band_named
    from mime.nodes.flow.fvm._resolved import window_cells
    from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode, ResolvedSpec
    re = float(os.environ.get("C2_MAC_RE", "260"))
    cells = int(os.environ.get("C2_MAC_CELLS", "32"))
    t_end = float(os.environ.get("C2_MAC_TIME", "0.04"))
    budget = float(os.environ.get("C2_MAC_BUDGET_S", "9000"))
    nu_scale = float(os.environ.get("C2_MAC_NU_SCALE", "1.0"))
    tag = os.environ.get("C2_MAC_TAG", f"mac_re{int(re)}_n{cells}" + ("_nu105" if nu_scale != 1.0 else ""))
    dtype = run_dtype()
    h = 2 * RADIUS / cells
    nx = cells + cells // 2
    nz = 4 * cells
    U = re * NU / (2 * RADIUS)
    dt = float(os.environ.get("C2_MAC_DT", str(0.25 * h / (3.0 * U))))
    steps = int(round(t_end / dt))
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
    force = jnp.asarray([0.0, 0.0, 8.0 * NU * U / RADIUS ** 2], dtype=dtype)
    inputs = {"body_0_pose": pose, "body_0_twist": jnp.zeros(6, dtype=dtype), "body_0_mass": mass6, "body_force": force}
    wanted = [k for k in node.own_inputs() if k not in inputs]
    chunk_steps = 25

    def one(state, _):
        new = node.update(state, inputs, dt)
        new = {**state, **new}
        return new, (new["body_0_load"], new["body_0_share"], jnp.mean(new["velocity_z"]))

    chunk = jax.jit(lambda state: jax.lax.scan(one, state, None, length=chunk_steps))
    state = dict(node.initial_state())
    loads, shares, means, walls = [], [], [], []
    t0 = time.time()
    info = {"tag": tag, "re": re, "cells_across": cells, "box": [nx, nx, nz], "h": h, "dt": dt, "steps_planned": steps, "U": U,
            "unfed_inputs": wanted, "cells_a_radius": A / h, "nu_scale": nu_scale}
    print(json.dumps(info), flush=True)
    for c in range(steps // chunk_steps):
        t1 = time.time()
        state, (ld, sh, mn) = chunk(state)
        jax.block_until_ready(ld)
        walls.append(time.time() - t1)
        loads.append(np.asarray(ld)), shares.append(np.asarray(sh)), means.append(np.asarray(mn))
        if not np.isfinite(loads[-1]).all() or time.time() - t0 > budget:
            break
    loads, shares, means = np.concatenate(loads), np.concatenate(shares), np.concatenate(means)
    info.update(steps_done=int(len(loads)), s_per_step=float(np.median(walls[1:]) / chunk_steps) if len(walls) > 1 else None,
                stopped_at=float(np.asarray(state.get("stopped_at", -1))), fz_load_end=float(loads[-1, 2]), fz_share_end=float(shares[-1, 2]),
                finite=bool(np.isfinite(loads).all()), state_keys=sorted(k for k in state if not k.startswith("velocity"))[:40])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), load=loads, share=shares, mean_w=means, dt=dt, box_mean_to_lumen=nx * nx * h * h / (math.pi * RADIUS ** 2))
    with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
