"""The temporal order of each projection scheme by step halving on one mesh from one developed state.

A p3 run of ``C2_ORDER_PRE`` steps gives the start (velocity, projected flux, pressure and the pressure one step back).
Each scheme then covers the same time with 32, 64, 128 and 256 steps (tolerance 1e-11, first guess zero). The pressure
one step back is interpolated linearly to each step. Orders are log2 of successive differences."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402


def main():
    import jax
    import jax.numpy as jnp
    case_name = os.environ.get("C2_ORDER_CASE", "t260")
    pre = int(os.environ.get("C2_ORDER_PRE", "600"))
    base = run.Case(run.parse(f"{case_name}:p3:zero:1e-11"))
    state = base.start()
    state, _ = base.steps_only(base.tab, base.T, base.par, state, pre)
    vol = np.asarray(base.mesh.V)
    l2 = lambda a: float(np.sqrt((vol.reshape((-1,) + (1,) * (a.ndim - 1)) * a * a).sum() / vol.sum()))
    start = {k: np.asarray(v) for k, v in state.items() if k != "hist"}
    out = {"case": case_name, "cells": int(base.mesh.n), "dt0": base.dt, "pre_steps": pre, "schemes": {}}
    ends = {}
    for scheme in ["p3", "l1pf", "l1rc", "l1rcc", "x1pf", "x1rc", "x1rcc"]:
        case = run.Case(run.parse(f"{case_name}:{scheme}:zero:1e-11"))
        rows = []
        for j in range(4):
            f = 0.5 ** j
            s = case.start()
            s = dict(s, **{k: jnp.asarray(v) for k, v in start.items()})
            s["p_old"] = jnp.asarray(start["p"] - f * (start["p"] - start["p_old"]))
            s["gp_old"] = jnp.asarray(start["gp"] - f * (start["gp"] - start["gp_old"]))
            par = dict(case.par, dt=case.par["dt"] * f)
            s, o = case.steps_only(case.tab, case.T, par, s, 32 * 2 ** j)
            d = case.diag(case.tab, par, s)
            u, p = np.asarray(s["u"]), np.asarray(s["p"])
            rows.append({"u": u, "p": p - (vol * p).sum() / vol.sum(), "fz": float(d["fp"][2] + d["fv"][2]),
                         "finite": bool(np.isfinite(u).all()), "its": float(np.asarray(o["its"]).mean())})
        ends[scheme] = rows
        du = [l2(rows[j]["u"] - rows[j + 1]["u"]) for j in range(3)]
        dp = [l2(rows[j]["p"] - rows[j + 1]["p"]) for j in range(3)]
        df = [abs(rows[j]["fz"] - rows[j + 1]["fz"]) for j in range(3)]
        order = lambda e: [float(np.log2(e[j] / e[j + 1])) if e[j + 1] > 0 and np.isfinite(e[j]) else None for j in range(2)]
        out["schemes"][scheme] = {"finite": [r["finite"] for r in rows], "du": du, "dp": dp, "dfz": df, "order_u": order(du),
                                  "order_p": order(dp), "order_fz": order(df), "u_scale": l2(rows[3]["u"]), "p_scale": l2(rows[3]["p"]),
                                  "fz": [r["fz"] for r in rows], "its": [r["its"] for r in rows]}
        print(scheme, json.dumps(out["schemes"][scheme], default=float), flush=True)
    ref = ends["p3"][3]
    for scheme, rows in ends.items():
        out["schemes"][scheme]["u_against_p3_finest"] = [l2(r["u"] - ref["u"]) for r in rows]
        out["schemes"][scheme]["p_against_p3_finest"] = [l2(r["p"] - ref["p"]) for r in rows]
    os.makedirs(run.OUT, exist_ok=True)
    with open(os.path.join(run.OUT, f"order_{case_name}{os.environ.get('C2_ORDER_TAG', '')}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
