"""The closed loop's gradient check (camera spike R2): the tracking objective's derivative in the controller's gains
(``kp, ki, kd``) and one allocation parameter (``limit``, A), in reverse mode, against forward mode (jvp) and central
differences, through the built graph's own differentiated run (``mime.build._maddening.differentiated_run``: the
program a study's derivative runs), in the run's dtype (JAX_ENABLE_X64).

The objective: ``J = mean_n |p_n - r(t_n)|^2 / (1 mm)^2`` over the run's steps, ``p_n`` the sphere's true position
after step ``n`` and ``r`` the reference (``loop_case.reference``).

**Tolerances, from the mechanism.**
- *reverse against jvp*: both are the exact derivative of one piecewise-smooth program at one point, summed in another
  order: they differ by rounding accumulated along the run, at most about ``eps`` per operation on the scale of the
  absolute sum of the per-step contributions, ``S = sum_n |dj_n/dtheta|`` (read from the jvp of the trajectory). The
  line is ``16 eps N S`` (linear accumulation over ``N`` steps, sixteen operations a step).
- *reverse against central differences*: ``CD(h)`` at ``h`` and ``h/2``; for a smooth objective ``CD(h/2)`` errs by its
  truncation, ``tau = |CD(h) - CD(h/2)| / 3`` (Richardson), and by rounding, ``rho = 3 sigma_J / (h/2)``, ``sigma_J``
  the objective's own rounding noise (ECnoise: fourth differences of seven values). The line is ``3 tau + 3 rho``. A
  kink within ``h`` (an active-set change, a limit switching on, a pixel corner) shows as a second difference that
  falls as ``h`` rather than ``h^2``; there the central difference errs by up to half the jump in slope, and the line is
  ``|D(h/2)| / h + 3 rho`` instead. Where the line is above a tenth of the derivative the check cannot discriminate and
  says so (exit 3).

Draws (DETECTION_SEED): the gains within 20 % of their defaults, the reference's step 4.5 to 6 mm in a drawn direction
(every draw holds a coil at its limit early on). ``--arm camera|state``, ``--steps N``.

Exit 0: no check failed and at least one discriminated; 1 a check failed or a derivative is not finite (a control's
catch); 3 no parameter's check could discriminate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_case  # noqa: E402

jnp: Any = jax.numpy
NAMES = ("kd", "ki", "kp", "limit")
#: The relative spacing of the noise table, by dtype: about 1e7 units in the last place in float64, 300 in float32.
NOISE_SPACING = {np.dtype(np.float64): 1e-9, np.dtype(np.float32): 3e-5}


def drawn(seed: int) -> tuple[dict[str, float], tuple[float, float]]:
    rng = np.random.default_rng(7000 + seed)
    gains = {name: float(loop_case.GAINS[name] * rng.uniform(0.8, 1.2)) for name in NAMES if name != "limit"}
    gains["limit"] = loop_case.GAINS["limit"]
    angle, size = rng.uniform(0, 2 * math.pi), rng.uniform(4.5e-3, 6e-3)
    return gains, (size * math.cos(angle), size * math.sin(angle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="camera")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="", help="writes <arm>_<dtype>_<variant>_<plant>_<steps>_<seed>.json")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="the drawn kd times this (a loop near instability)")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DETECTION_SEED", "0") or 0))
    parser.add_argument("--h", type=float, default=0.0, help="relative step of the central differences (0: the dtype's)")
    args = parser.parse_args()
    from mime.build._maddening import differentiated_run, external_inputs, full_state, graph_params

    dtype = np.dtype(jnp.zeros(()).dtype)
    eps = float(np.finfo(dtype).eps)
    gains, step = drawn(args.seed)
    gains["kd"] *= args.kd_scale
    loop_case.STEP = step
    clock = {"start": time.time()}
    built = loop_case.experiment(args.arm, **gains).build()
    graph = built.graph
    assert external_inputs(graph) == [], external_inputs(graph)
    graph.reset_state()
    state = {node: dict(fields) for node, fields in full_state(graph).items()}
    params = graph_params(graph)
    clock["build"] = time.time() - clock["start"]
    steps = args.steps
    times = (np.arange(steps) + 1) * loop_case.DT
    reference = jnp.asarray(np.asarray(loop_case.reference(jnp.asarray(times))).T, dtype=dtype)
    kept = (("sphere", "pose"), ("controller", "write_currents"))

    def objective(checkpoint: bool) -> Any:
        run = differentiated_run(graph, steps, (), kept, checkpoint=checkpoint)

        def at(theta: Any) -> tuple[Any, Any]:
            nodes = {node: dict(leaves) for node, leaves in params["nodes"].items()}
            for name in NAMES:
                nodes["controller"][f"setting_{name}"] = theta[name]
            _, _, out = run(state, {**params, "nodes": nodes})
            miss = out["sphere"]["pose"][:, :3] - reference
            per_step = jnp.sum(miss * miss, axis=1) / 1e-6 / steps
            return jnp.sum(per_step), (per_step, out["controller"]["write_currents"])

        return at

    theta0 = {name: jnp.asarray(gains[name], dtype=dtype) for name in NAMES}
    forward = jax.jit(lambda theta: objective(False)(theta)[0])
    reverse = jax.jit(jax.value_and_grad(lambda theta: objective(True)(theta)[0]))

    def tangent(name: str) -> Any:
        direction = {k: jnp.asarray(1.0 if k == name else 0.0, dtype=dtype) for k in NAMES}
        return jax.jvp(objective(False), (theta0,), (direction,))

    jvp = jax.jit(tangent, static_argnums=0)
    value, grad = reverse(theta0)
    grad = {k: float(v) for k, v in grad.items()}
    clock["reverse"] = time.time() - clock["start"] - clock["build"]
    (j0, (per_step, currents)), (jt, (per_step_t, _)) = jvp("kp")
    rows: dict[str, Any] = {}
    value = float(value)
    at_limit = np.abs(np.asarray(currents)) >= np.asarray(gains["limit"]) * (1 - 1e-6)
    relative = args.h or (1e-4 if dtype == np.float64 else 1e-2)
    verdict = "pass"
    for name in NAMES:
        (_, _), (tj, (tper, _)) = jvp(name)
        absum = float(np.sum(np.abs(np.asarray(tper, dtype=np.float64))))
        h1 = relative * gains[name]
        h2 = h1 / 2

        def moved(by: float) -> float:
            return float(forward({**theta0, name: jnp.asarray(gains[name] + by, dtype=dtype)}))

        j_up1, j_dn1, j_up2, j_dn2 = moved(h1), moved(-h1), moved(h2), moved(-h2)
        cd1, cd2 = (j_up1 - j_dn1) / (2 * h1), (j_up2 - j_dn2) / (2 * h2)
        second1, second2 = j_up1 - 2 * value + j_dn1, j_up2 - 2 * value + j_dn2
        # The objective's rounding noise (More and Wild's ECnoise: the fourth differences of seven values spaced far
        # enough apart, in the dtype, that their roundings are independent, and close enough that the smooth part's
        # fourth difference is far below them).
        spacing = NOISE_SPACING[dtype] * gains[name]
        table = np.asarray([moved(k * spacing) for k in range(-3, 4)], dtype=np.float64)
        fourth = np.diff(table, 4)
        noise = float(math.sqrt(np.mean(fourth ** 2) * (math.factorial(4) ** 2) / math.factorial(8)))
        tau, rho = abs(cd1 - cd2) / 3.0, 3.0 * noise / h2
        # A kink within h (an active-set change, a limit switching, a pixel corner): the second difference falls as h
        # rather than h^2 (ratio a half rather than a quarter), and the central difference errs by up to half the jump
        # in slope, which the one-sided differences bound: |D(h)| / (2 h).
        kinked = abs(second2) > 6 * noise and abs(second2 / second1) > 0.375 if second1 else False
        line_cd = (abs(second2) / (2 * h2) if kinked else 3.0 * tau) + 3.0 * rho + 16 * eps * abs(cd2)
        line_ad = 16 * eps * steps * absum
        g, t = grad[name], float(tj)
        finite = all(math.isfinite(x) for x in (g, t, cd1, cd2))
        ad_ok = finite and abs(g - t) <= line_ad
        cd_ok = finite and abs(g - cd2) <= line_cd
        discriminates = line_cd <= 0.1 * max(abs(cd2), 1e-300)
        rows[name] = {"reverse": g, "jvp": t, "cd_h": cd1, "cd_h2": cd2, "h": h1, "tau": tau, "rho": rho,
                      "noise_J": noise, "second_h": second1, "second_h2": second2, "kinked": kinked,
                      "line_cd": line_cd, "line_ad": line_ad, "abs_sum": absum,
                      "rev_minus_jvp": g - t, "rev_minus_cd": g - cd2, "ratio_ad": abs(g - t) / line_ad if line_ad else None,
                      "ratio_cd": abs(g - cd2) / line_cd if line_cd else None, "finite": finite, "ad_ok": ad_ok,
                      "cd_ok": cd_ok, "discriminates": discriminates}
        if not finite or not ad_ok or (discriminates and not cd_ok):
            verdict = "fail"
    if verdict == "pass" and not any(row["discriminates"] for row in rows.values()):
        verdict = "cannot_discriminate"
    clock["total"] = time.time() - clock["start"]
    poses = np.asarray(jax.jit(lambda th: objective(False)(th)[1][0])(theta0))
    summary = {"arm": args.arm, "dtype": str(dtype), "steps": steps, "seed": args.seed, "gains": gains, "step": step,
               "variant": loop_case.VARIANT, "plant": loop_case.PLANT, "J": value, "verdict": verdict,
               "ticks_at_limit": int(at_limit.any(axis=1).sum()) // loop_case.PERIOD_STEPS,
               "steps_at_limit": int(at_limit.any(axis=1).sum()), "clock_s": clock, "rows": rows,
               "per_step_last": float(poses[-1])}
    print(json.dumps(summary, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        tag = f"{args.arm}_{dtype}_{loop_case.VARIANT}_{loop_case.PLANT or 'none'}_{steps}_kd{args.kd_scale:g}_{args.seed}"
        (Path(args.output_dir) / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    return {"pass": 0, "fail": 1, "cannot_discriminate": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
