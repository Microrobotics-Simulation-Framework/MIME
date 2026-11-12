"""Camera spike R1, items 1 and 2: the run-level scalar's gradient through the in-loop camera by vjp, by jvp and by central
differences, in one dtype (one process per dtype: a node's step is traced once a process).

    python probe/r1_grad.py --dtype float64 --out OUT.json [--width 160] [--steps 12] [--period 3] [--draw SEED]
        [--continued N0] [--per-step]

``--draw SEED`` draws the operating point (the ball's start along and across the tube, its speed, the frame period's
phase) at physical scales; without it the fixed point below. ``--continued N0`` starts the differentiated run from the state
after ``N0`` steps (the cache already made and its numbers equal: a continued run). ``--per-step`` adds each step's
centroid's Jacobian by reverse and by forward mode (item 2's held frames). Writes JSON; judged by ``r1_judge.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


#: The central differences' steps, as fractions of a pixel of image motion (``PIXEL_SCALE``), each sqrt(10) from the next
#: so that Richardson's estimate of the truncation at h reads D(h) - D(sqrt(10) h) over 9.
LADDER = (1e-1, 10 ** -1.5, 1e-2, 10 ** -2.5, 1e-3, 10 ** -3.5, 1e-4, 10 ** -4.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float64", "float32"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--period", type=int, default=3)
    parser.add_argument("--draw", type=int, default=None)
    parser.add_argument("--continued", type=int, default=0)
    parser.add_argument("--per-step", action="store_true")
    parser.add_argument("--no-ladder", action="store_true")
    arguments = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np

    import r1_core as core

    core.plant()
    point = {"x0_um": -30.0, "z0_um": 15.0, "speed": 2e-3, "period": arguments.period}
    if arguments.draw is not None:
        rng = np.random.default_rng(arguments.draw)
        # The ball inside the lumen (radius 60 um, the ball's 20) and in view at 160 px (half-field 107 um) over the run.
        point = {"x0_um": float(rng.uniform(-45.0, 0.0)), "z0_um": float(rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 30.0)),
                 "speed": float(rng.uniform(0.5e-3, 3e-3)), "period": int(rng.integers(2, 5))}
    built = core.case(width=arguments.width, period_steps=point["period"], x0_um=point["x0_um"], z0_um=point["z0_um"],
                      speed=point["speed"]).build()
    graph = built.graph
    from mime.build import _maddening as mad

    assert not [i for i in mad.external_inputs(graph) if not i[0].startswith(core.CAMERA)], mad.external_inputs(graph)
    state0, params0, theta0 = core.start_of(built)
    steps = arguments.steps
    if arguments.continued:
        warm = core.scalar_run(graph, arguments.continued, core.per_step_objective(graph), checkpoint=False)
        _, _, state0 = jax.jit(warm)(state0, params0)
        state0 = jax.tree_util.tree_map(lambda a: a, state0)
    term = core.per_step_objective(graph)
    run = core.scalar_run(graph, steps, term, checkpoint=True)

    def objective(theta):
        state, params = core.inserted(theta, state0, params0)
        total, kept, final = run(state, params)
        return total, kept, final

    def scalar(theta):
        return objective(theta)[0]

    started = time.perf_counter()
    report: dict = {"dtype": arguments.dtype, "width": arguments.width, "height": arguments.width * 3 // 4,
                    "steps": steps, "continued_from": arguments.continued, "point": point, "names": list(core.NAMES),
                    "theta0": np.asarray(theta0, dtype=np.float64).tolist(), "plant": core.PLANT,
                    "render_checkpoint": os.environ.get("MIME_SPIKE_RENDER_CHECKPOINT") == "1",
                    "cpu": cpu_model(), "jax": jax.__version__, "eps": float(jnp.finfo(theta0.dtype).eps)}
    t0 = time.perf_counter()
    forward = jax.jit(objective)
    total, kept, final = forward(theta0)
    jax.block_until_ready(total)
    report["seconds_first_forward"] = time.perf_counter() - t0
    report["J"] = float(total)
    report["c"] = np.asarray(kept["c"], dtype=np.float64).tolist()
    report["p"] = np.asarray(kept["p"], dtype=np.float64).tolist()
    camera = final[core.CAMERA]
    report["flags_last"] = {name: float(camera[name]) for name in
                            ("beyond_distortion", "saturated", "outside_parent", "small_silhouette", "window_overflow",
                             "behind", "ghosting", "thin_cap")}
    report["frame_steps_seen"] = int(camera["frame_step"])

    # Reverse mode: the gradient by vjp (MIME's chunked checkpoint, as a study's reverse mode takes it).
    t0 = time.perf_counter()
    grad = jax.jit(jax.grad(scalar))
    g_vjp = grad(theta0)
    jax.block_until_ready(g_vjp)
    report["seconds_vjp_first"] = time.perf_counter() - t0
    report["g_vjp"] = np.asarray(g_vjp, dtype=np.float64).tolist()

    # Forward mode: one jvp per parameter, the tangents of every step's frame and projected centre kept for the scale.
    frames_run = core.scalar_run(graph, steps, core.per_step_objective(graph, keep_frames=True), checkpoint=False)

    def frames(theta):
        state, params = core.inserted(theta, state0, params0)
        total, kept, final = frames_run(state, params)
        held = final[core.CAMERA]
        return total, kept["frame"], kept["p"], kept["c"], (held["cache_transmission"], held["cache_rays"])

    tangent = jax.jit(lambda theta, t: jax.jvp(frames, (theta,), (t,)))
    light = core.backlight(graph.get_node(core.CAMERA)._spec())  # noqa: SLF001

    def loss_of(frame_stack, p_stack):
        c = jax.vmap(lambda f: core.soft_centroid(f, light))(frame_stack)
        return jnp.sum((c - p_stack) ** 2)

    cotangent = jax.jit(jax.grad(loss_of, argnums=(0, 1)))
    g_jvp, scale, per_step_fwd, cache_tangent = [], [], [], []
    for index in range(len(core.NAMES)):
        e = jnp.zeros_like(theta0).at[index].set(1.0)
        (total_, frame_stack, p_stack, c_stack, _), (dtotal, dframes, dp, dc, dcache) = tangent(theta0, e)
        # The static cache's tangent at the run's end: zero in the ball's parameters (nothing of a moving body may reach
        # the static structure's image), and in the leaves only where the run made the cache.
        cache_tangent.append(float(max(jnp.max(jnp.abs(part)) for part in dcache)))
        g_jvp.append(float(dtotal))
        bar_f, bar_p = cotangent(frame_stack, p_stack)
        # The scale of the sum's terms (CLAUDE.md: judge a result on the scale it came from): every pixel's and every
        # projected centre's contribution to dJ/dtheta, in absolute value, summed.
        scale.append(float(jnp.sum(jnp.abs(bar_f * dframes)) + jnp.sum(jnp.abs(bar_p * dp))))
        per_step_fwd.append(np.asarray(dc, dtype=np.float64).tolist())
    report["g_jvp"] = g_jvp
    report["scale"] = scale
    report["dc_fwd"] = per_step_fwd  # [parameter][step][u, v]
    report["cache_tangent_maxabs"] = cache_tangent

    if arguments.per_step:
        def centroids(theta):
            return objective(theta)[1]["c"]

        t0 = time.perf_counter()
        jac = jax.jit(jax.jacrev(centroids))(theta0)  # (steps, 2, parameters)
        report["seconds_jacrev"] = time.perf_counter() - t0
        report["dc_rev"] = np.moveaxis(np.asarray(jac, dtype=np.float64), -1, 0).tolist()
        report["frame_step_of"] = [int(((arguments.continued + t) // point["period"]) * point["period"])
                                   for t in range(steps)]

    if not arguments.no_ladder:
        values = jax.jit(lambda theta: objective(theta)[:2])
        ladder: dict = {}
        for index, name in enumerate(core.NAMES):
            unit = core.PIXEL_SCALE[name] * (160.0 / arguments.width)
            rows = []
            for fraction in LADDER:
                h = fraction * unit
                e = jnp.zeros_like(theta0).at[index].set(1.0)
                up_total, up_kept = values(theta0 + h * e)
                down_total, down_kept = values(theta0 - h * e)
                h_actual = float((theta0 + h * e)[index] - theta0[index])
                rows.append({"fraction": fraction, "h": h_actual, "J_up": float(up_total), "J_down": float(down_total),
                             "D": (float(up_total) - float(down_total)) / (2.0 * h_actual) if h_actual else None,
                             "dc": ((np.asarray(up_kept["c"], dtype=np.float64) - np.asarray(down_kept["c"],
                                                                                              dtype=np.float64))
                                    / (2.0 * h_actual)).tolist() if h_actual else None})
            ladder[name] = rows
        report["ladder"] = ladder
    report["seconds_total"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1))
    print(f"r1_grad {arguments.dtype} J={report['J']:.6g} g_vjp={report['g_vjp']} g_jvp={report['g_jvp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
