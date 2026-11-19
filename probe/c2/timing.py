"""The step's time split: for each mesh and configuration, the whole step, the step with the pressure iterations
skipped (the explicit stages, the projection's bookkeeping and the first guess), hence the time an iteration; and the
MAC grid's stencil stage and transform solve at the nearest box, in the same process.

``C2_TIME_CASES`` (comma separated cases), ``C2_TIME_TAG``, ``C2_TIME_SAVE`` (write the state after the warm-up and 20
steps of the first case's l1rc:proj8, for the device comparison), ``C2_TIME_DTYPE``."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402

CONFIGS = ["p3:zero:1e-8", "p3:proj8:1e-8", "l1rc:zero:1e-8", "l1rc:proj8:1e-8", "l1rc:proj4:1e-8", "l1rc:projR8:1e-8",
           "l1pf:proj8:1e-8", "l1rc:proj8:1e-8:c32", "l1rc:proj8:1e-6", "l1rc:proj8:1e-8:st16"]
BOX = {"T": 16, "S": 16, "D": 16, "SL": 32, "SX": 64}
BOXES = {16: (27, 27, 64), 32: (43, 43, 128), 64: (85, 85, 256)}


def timed(f, *args, reps=3):
    import jax
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - t0)
    return out, min(times), first


def mac_parts(size, dtype):
    import jax
    import jax.numpy as jnp
    from gfv import stag
    box = BOXES[size]
    h = 3.5e-3 / size
    rng = np.random.default_rng(0)
    u = tuple(jnp.asarray(0.1 * rng.standard_normal(box), dtype) for _ in range(3))
    p = jnp.asarray(0.01 * rng.standard_normal(box), dtype)
    par = (h, run.NU, 1e-5, 0.0)
    n_stage = 50
    stage = jax.jit(lambda u, p: jax.lax.scan(lambda c, _: (stag.mac_stage(par, *c), None), (u, p), None, length=n_stage)[0])
    _, t_stage, _ = timed(stage, u, p, reps=5)
    nx, ny, nz = box
    s = lambda k, c: jnp.sin(jnp.pi * jnp.arange(k, dtype=dtype) / c) ** 2
    symbol = (4.0 / h ** 2) * (s(nx, nx)[:, None, None] + s(ny, ny)[None, :, None] + s(nz // 2 + 1, nz)[None, None, :])
    symbol = symbol.at[0, 0, 0].set(1.0)

    def transform(src):
        solved = jnp.fft.rfftn(src) / symbol
        return jnp.fft.irfftn(solved.at[0, 0, 0].set(0.0), s=box).astype(src.dtype)

    loop = jax.jit(lambda b: jax.lax.fori_loop(0, 20, lambda i, v: transform(b + 1e-3 * v), b))
    _, t_loop, _ = timed(loop, p, reps=5)
    cells = int(np.prod(box))
    return {"mac_box": list(box), "mac_cells": cells, "mac_stage_ms": 1e3 * t_stage / n_stage, "mac_transform_ms": 1e3 * t_loop / 20,
            "mac_step_ms": 3e3 * (t_stage / n_stage + t_loop / 20), "mac_ns_cell_step": 3e9 * (t_stage / n_stage + t_loop / 20) / cells}


def main():
    import jax
    import jax.numpy as jnp
    tag = os.environ.get("C2_TIME_TAG", "cpu")
    dtype = os.environ.get("C2_TIME_DTYPE", "float64")
    cases = os.environ.get("C2_TIME_CASES", "s260").split(",")
    configs = os.environ.get("C2_TIME_CONFIGS", ",".join(CONFIGS)).split(",")
    warm = int(os.environ.get("C2_TIME_WARM", "300"))
    n_time = int(os.environ.get("C2_TIME_STEPS", "20"))
    os.makedirs(run.OUT, exist_ok=True)
    path = os.path.join(run.OUT, f"timing_{tag}_{dtype}.jsonl")
    device = str(jax.devices()[0])

    def write(row):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k not in ("amg", "quality")}, default=float), flush=True)

    for ci, case_name in enumerate(cases):
        mesh_name = run.CASES[case_name][0]
        try:
            mac = mac_parts(BOX[mesh_name], jnp.float64 if dtype == "float64" else jnp.float32)
        except Exception as e:  # noqa: BLE001
            mac = {"mac_failed": repr(e)[:300]}
        write(dict(mac, row="mac", case=case_name, device=device, tag=tag, dtype=dtype))
        # one developed state for every configuration of this case
        first = run.Case(run.parse(f"{case_name}:l1rc:proj8:1e-8"), dtype=dtype)
        t0 = time.time()
        state, _ = first.steps_only(first.tab, first.T, first.par, first.start(), warm)
        jax.block_until_ready(state["u"])
        base = {k: v for k, v in state.items() if k not in ("hist", "k")}
        warm_s = time.time() - t0
        if ci == 0 and os.environ.get("C2_TIME_SAVE"):
            s20, o20 = first.steps_only(first.tab, first.T, first.par, state, 20)
            np.savez(os.path.join(run.OUT, f"equal_{tag}_{dtype}.npz"), u=np.asarray(s20["u"]), p=np.asarray(s20["p"]), F=np.asarray(s20["F"]),
                     its=np.asarray(o20["its"]))
        for config in configs:
            try:
                cfg = run.parse(f"{case_name}:{config}")
                stride = cfg["stride"]
                rows = {}
                for skip in (False, True):
                    case = run.Case(cfg, dtype=dtype, skip_solve=skip)
                    s = dict(case.start(), **base)
                    pre = 40 * stride
                    s, _ = case.steps_only(case.tab, case.T, case.par, s, pre)     # fills this configuration's history
                    (s2, o), t, first_s = timed(case.steps_only, case.tab, case.T, case.par, s, n_time)
                    its = np.asarray(o["its"])
                    used = np.asarray(o["use"]).astype(bool)
                    rows[skip] = {"ms_step": 1e3 * t / n_time, "its_step": float(its.sum(axis=1).mean()),
                                  "its_step_used": float(its[used].sum(axis=1).mean()) if used.any() else None,
                                  "res_max": float(np.asarray(o["res"]).max()), "compile_s": first_s}
                    facts = case.facts()
                    del case
                full, bare = rows[False], rows[True]
                ms_it = (full["ms_step"] - bare["ms_step"]) / max(full["its_step"], 1e-9)
                row = {"row": "graph", "case": case_name, "config": config, "device": device, "tag": tag, "dtype": dtype, "cells": facts["cells"],
                       "ms_step": full["ms_step"], "ms_step_no_iterations": bare["ms_step"], "its_step": full["its_step"],
                       "its_step_used": full["its_step_used"], "ms_iteration": ms_it, "res_max": full["res_max"],
                       "ns_cell_step": 1e6 * full["ms_step"] / facts["cells"], "compile_s": full["compile_s"], "warm_s": warm_s,
                       "amg_levels": facts["amg"]["levels"], "amg_complexity": facts["amg"]["operator_complexity"], "dt": facts["dt"]}
                if "mac_ns_cell_step" in mac:
                    row["c_as_run"] = row["ns_cell_step"] / mac["mac_ns_cell_step"]
                    row["c_no_iterations"] = 1e6 * bare["ms_step"] / facts["cells"] / mac["mac_ns_cell_step"]
                    row["c_per_iteration"] = 1e6 * ms_it / facts["cells"] / mac["mac_ns_cell_step"]
                write(row)
            except Exception as e:  # noqa: BLE001
                import traceback
                write({"row": "failed", "case": case_name, "config": config, "failed": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
            jax.clear_caches()


if __name__ == "__main__":
    main()
