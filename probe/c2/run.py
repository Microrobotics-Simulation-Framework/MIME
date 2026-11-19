"""CORE spike 2's runner: one group of configurations a process (``C2_GROUP``), results in ``C2_OUT``.

A configuration is ``case:scheme:guess:tol[:option...]``:
  case     ``s260``, ``s650`` (sphere, mesh S), ``S650`` (sphere, mesh SL), ``d260``, ``d650`` (thin spheroid, mesh D), ``t260`` (mesh T)
  scheme   ``p3``, ``l1pf``, ``l1rc``, ``l1rcc``, ``x1pf``, ``x1rc``, ``x1rcc`` (see flow.py; ``pf`` carries the projected
           flux, ``rc`` interpolates with the Rhie-Chow term, ``rcc`` does both)
  guess    ``zero``, ``prev``, ``proj4``, ``proj8`` (the last 4 or 8 solutions), ``projR4``, ``projR8`` (Fischer's restart)
  tol      the relative residual, or ``e<eta>`` for the energy-norm stop
  options  ``pert`` (a 2 % cross-flow in the start, so the wake may leave the axis), ``c32`` (float32 V-cycle), ``nu105`` (the planted viscosity), ``dt<f>`` (the step times f), ``n<steps>``,
           ``st<m>`` (the first guess and its history use every m-th step only; the other steps start from zero and are
           left out of the iteration counts: the guess's quality when successive solves are m steps apart)"""
import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
OUT = os.environ.get("C2_OUT", os.path.join(os.path.dirname(HERE), "out"))
NU, RHO, RADIUS = 3.3e-6, 1060.0, 1.75e-3
A = 0.4 * RADIUS
LENGTH = 8.0 * RADIUS
MESHES = {
    "T": dict(aspect=1.0, h_body=A / 5, h_rim=A / 5, h_far=320e-6, sectors=8),
    "S": dict(aspect=1.0, h_body=A / 8, h_rim=A / 8, h_far=140e-6, sectors=16),
    "SL": dict(aspect=1.0, h_body=A / 14, h_rim=A / 14, h_far=75e-6, sectors=24),
    "SX": dict(aspect=1.0, h_body=A / 24, h_rim=A / 24, h_far=46e-6, sectors=48),
    "D": dict(aspect=0.05, h_body=A / 10, h_rim=A / 40, h_far=165e-6, sectors=16),
    "DT": dict(aspect=0.05, h_body=A / 6, h_rim=A / 20, h_far=320e-6, sectors=8),
}
CASES = {"t260": ("T", 260.0), "t650": ("T", 650.0), "s260": ("S", 260.0), "s650": ("S", 650.0), "S260": ("SL", 260.0), "S650": ("SL", 650.0),
         "d260": ("D", 260.0), "d650": ("D", 650.0), "dt260": ("DT", 260.0), "X260": ("SX", 260.0)}
SCHEMES = {"p3": ("p3", False, True), "l1pf": ("l1", False, True), "l1rc": ("l1", True, False), "l1rcc": ("l1", True, True),
           "x1pf": ("x1", False, True), "x1rc": ("x1", True, False), "x1rcc": ("x1", True, True)}
ONE = ["l1pf", "l1rc", "l1rcc", "x1pf", "x1rc", "x1rcc"]
GROUPS = {
    "smoke": ["t260:" + s + ":zero:1e-8:n200" for s in ["p3"] + ONE] + ["t260:l1rc:proj8:1e-8:n200", "t260:l1rc:projR4:e1e-3:c32:n200"],
    "a_p3": ["s260:p3:zero:1e-8", "s260:p3:prev:1e-8", "s260:p3:proj4:1e-8", "s260:p3:projR8:1e-8"],
    "a_one": ["s260:" + s + ":zero:1e-8" for s in ONE],
    "a_one4": ["s260:" + s + ":proj4:1e-8" for s in ONE],
    "b_guess": ["s260:l1rc:prev:1e-8", "s260:l1rc:proj8:1e-8", "s260:l1rc:projR8:1e-8", "s260:l1rc:projR4:1e-8"],
    "c_tol": ["s260:l1rc:zero:1e-6", "s260:l1rc:zero:1e-4", "s260:l1rc:proj4:1e-6", "s260:l1rc:proj4:1e-4", "s260:p3:proj4:1e-6",
              "s260:p3:proj4:1e-4"],
    "d_energy": ["s260:l1rc:zero:e1e-2", "s260:l1rc:zero:e1e-3", "s260:l1rc:zero:e1e-4", "s260:l1rc:proj4:e1e-2", "s260:l1rc:proj4:e1e-3",
                 "s260:l1rc:proj4:e1e-4"],
    "e_c32": ["s260:l1rc:zero:1e-8:c32", "s260:l1rc:proj4:1e-8:c32", "s260:l1rc:proj4:1e-6:c32", "s260:p3:zero:1e-8:c32"],
    "f_plant": ["s260:l1rc:proj4:1e-8:nu105", "s260:p3:zero:1e-8:nu105"],
    "g_stride": ["s260:l1rc:proj4:1e-8:st4", "s260:l1rc:proj4:1e-8:st16", "s260:l1rc:proj4:1e-8:st64", "s260:l1rc:proj4:e1e-3:st16"],
    "g_stride2": ["s260:p3:proj4:1e-8:st4", "s260:p3:proj4:1e-8:st16", "s260:p3:proj4:1e-8:st64", "s260:l1rc:proj8:1e-8:st16"],
    "h_s650": ['s650:p3:zero:1e-8', 's650:l1rc:zero:1e-8', 's650:l1rc:proj4:1e-8', 's650:x1rc:proj4:1e-8', 's650:l1rc:proj4:1e-8:st16', 's650:l1rc:proj4:1e-6'],
    "i_d260": ['d260:p3:zero:1e-8', 'd260:l1rc:zero:1e-8', 'd260:l1rc:proj4:1e-8', 'd260:x1rc:proj4:1e-8', 'd260:l1rc:proj4:1e-8:st16', 'd260:l1rc:proj4:1e-6'],
    "j_d650": ['d650:p3:zero:1e-8', 'd650:l1rc:zero:1e-8', 'd650:l1rc:proj4:1e-8', 'd650:x1rc:proj4:1e-8', 'd650:l1rc:proj4:1e-8:st16', 'd650:l1rc:proj4:1e-6'],
    "k_S650a": ["S650:l1rc:proj4:1e-8:n1500", "S650:l1rc:zero:1e-8:n800", "S650:l1rc:proj4:1e-8:st16:n1500"],
    "k_S650b": ["S650:p3:zero:1e-8:n500", "S650:p3:proj4:1e-8:n800"],
    "k_S260": ["S260:l1rc:proj4:1e-8:n1500", "S260:l1rc:zero:1e-8:n800"],
    "m_s1": ["s650:l1rc:proj4:1e-8:pert:n40000"], "m_s16": ["s650:l1rc:proj4:1e-8:pert:st16:n40000"],
    "m_s64": ["s650:l1rc:proj4:1e-8:pert:st64:n40000"], "m_se": ["s650:l1rc:proj4:e1e-3:pert:st16:n40000"],
    "m_d1": ["d650:l1rc:proj4:1e-8:pert:n50000"], "m_d16": ["d650:l1rc:proj4:1e-8:pert:st16:n50000"],
    "m_d64": ["d650:l1rc:proj4:1e-8:pert:st64:n50000"], "m_p64": ["s650:p3:proj4:1e-8:pert:st64:n40000"],
    "m_t": ["t650:l1rc:proj4:1e-8:pert:n60000", "t650:l1rc:proj4:1e-8:pert:st16:n60000", "t650:l1rc:proj4:1e-8:pert:st64:n60000",
            "t650:l1rc:zero:1e-8:pert:n60000"],
    "e_c32b": ["s260:l1rc:proj8:1e-8:c32", "s260:l1rc:projR8:1e-8:c32", "s260:l1rc:proj4:1e-7:c32", "s260:l1rc:prev:1e-8:c32"],
    "l_dev": ["t650:l1rc:proj4:1e-8:n30000", "t650:l1rc:proj4:1e-8:st16:n30000", "t650:l1rc:proj4:1e-8:st64:n30000", "t650:l1rc:zero:1e-8:n30000"],
    "l_dev2": ["t650:p3:zero:1e-8:n30000", "t650:p3:proj4:1e-8:st16:n30000", "t650:l1rc:proj4:e1e-3:st16:n30000", "t650:l1rc:proj4:1e-8:nu105:n30000"],
}
STEPS, BLOCK, BLOCKS = int(os.environ.get("C2_STEPS", "4000")), 10, 10
BUDGET_S = float(os.environ.get("C2_BUDGET_S", "5400"))


def parse(config):
    parts = config.split(":")
    case, scheme, guess, tol = parts[:4]
    cfg = {"config": config, "case": case, "scheme": scheme, "guess": "proj" if guess.startswith("proj") else guess,
           "depth": int(guess.lstrip("projR")) if guess.startswith("proj") else 0, "fold": "restart" if guess.startswith("projR") else "solutions", "stop": "energy" if tol.startswith("e") else "res",
           "tol": 1e-12 if tol.startswith("e") else float(tol), "eta": float(tol[1:]) if tol.startswith("e") else 0.0,
           "c32": False, "nu_scale": 1.0, "dt_scale": 1.0, "steps": STEPS, "stride": 1, "pert": False}
    for o in parts[4:]:
        if o == "pert":
            cfg["pert"] = True
        elif o == "c32":
            cfg["c32"] = True
        elif o == "nu105":
            cfg["nu_scale"] = 1.05
        elif o.startswith("st"):
            cfg["stride"] = int(o[2:])
        elif o.startswith("dt"):
            cfg["dt_scale"] = float(o[2:])
        elif o.startswith("n"):
            cfg["steps"] = int(o[1:])
        else:
            raise ValueError(o)
    return cfg


_MESH = {}


def get_mesh(name):
    if name not in _MESH:
        import tube
        t0 = time.time()
        spec = dict(MESHES[name], radius=RADIUS, length=LENGTH, a=A)
        folder = os.environ.get("C2_MESHDIR", "")
        path = os.path.join(folder, name + ".npz") if folder else ""
        if path and os.path.exists(path):          # the section as meshed elsewhere (a container without gmsh)
            z = np.load(path, allow_pickle=False)
            nodes, tris, version = z["nodes"], z["tris"], str(z["gmsh"])
        else:
            nodes, tris, version = tube.meridional(RADIUS, LENGTH, A, spec["aspect"], spec["h_body"], spec["h_rim"], spec["h_far"])
            if path:
                np.savez(path, nodes=nodes, tris=tris, gmsh=version)
        m = tube.revolved(nodes, tris, spec["sectors"], RADIUS, LENGTH, A, spec["aspect"])
        m.info["gmsh"] = version
        q = m.quality()
        q.update(m.info, build_s=time.time() - t0, mesh=name)
        _MESH[name] = (m, q)
    return _MESH[name]


class Case:
    """A mesh, its operators, a solver and the compiled step of one configuration."""

    def __init__(self, cfg, impl=None, dtype="float64", skip_solve=False):
        import jax
        import jax.numpy as jnp

        import flow as fl
        self.cfg = cfg
        mesh_name, re = CASES[cfg["case"]]
        self.mesh, self.quality = get_mesh(mesh_name)
        self.sectors = MESHES[mesh_name]["sectors"]
        impl = impl or ("padded" if jax.devices()[0].platform == "gpu" else "scatter")
        self.dtype = jnp.float64 if dtype == "float64" else jnp.float32
        self.flow = fl.Flow(self.mesh, impl)
        self.U = re * NU / (2 * RADIUS)
        dt, self.dt_facts = self.flow.stable_dt(NU, 3.0 * self.U)
        self.dt = dt * cfg["dt_scale"]
        self.nu = NU * cfg["nu_scale"]
        self.solver = fl.Solver(self.flow.A, self.dtype, guess=cfg["guess"], depth=cfg["depth"], tol=cfg["tol"], stop=cfg["stop"],
                                eta=cfg["eta"], cycle32=cfg["c32"], fold=cfg["fold"])
        self.tab = fl.prims.as_device(self.flow.tab, self.dtype)
        self.T = self.solver.tables()
        f = lambda v: jnp.asarray(v, self.dtype)
        self.par = {"nu": f(self.nu), "dt": f(self.dt), "force": f([0.0, 0.0, 8.0 * NU * self.U / RADIUS ** 2])}
        kind, rc, carry = SCHEMES[cfg["scheme"]]
        self.rc = rc
        step = fl.make_step(self.flow, self.solver, kind, rc=rc, carry=carry, skip_solve=skip_solve, stride=cfg["stride"])
        self.step = step
        flow, sectors = self.flow, self.sectors

        def block(tab, T, par, state):
            def one(s, _):
                s, o = step(tab, T, par, s)
                return s, o
            state, outs = jax.lax.scan(one, state, None, length=BLOCK)
            return state, (outs, fl.diagnostics(flow, tab, par, state, sectors, rc))

        self.chunk = jax.jit(lambda tab, T, par, state: jax.lax.scan(lambda s, _: block(tab, T, par, s), state, None, length=BLOCKS))
        self.steps_only = jax.jit(lambda tab, T, par, state, n: jax.lax.scan(lambda s, _: step(tab, T, par, s), state, None, length=n),
                                  static_argnums=4)
        self.diag = jax.jit(lambda tab, par, state: fl.diagnostics(flow, tab, par, state, sectors, rc))
        self.fl = fl

    def start(self):
        rho2 = self.mesh.xc[:, 0] ** 2 + self.mesh.xc[:, 1] ** 2
        u0 = np.zeros((self.mesh.n, 3))
        u0[:, 2] = 2.0 * self.U * np.clip(1.0 - rho2 / RADIUS ** 2, 0.0, None)
        if self.cfg.get("pert"):      # 2 % of the mean speed across the axis, one wave along the tube: breaks the symmetry
            zz = 2 * np.pi * self.mesh.xc[:, 2] / LENGTH
            shape = np.clip(1.0 - rho2 / RADIUS ** 2, 0.0, None)
            u0[:, 0] = 0.02 * self.U * np.sin(zz) * shape
            u0[:, 1] = 0.02 * self.U * np.cos(zz) * shape
        return self.fl.initial_state(self.flow, self.solver, u0, self.dtype)

    def facts(self):
        return {"cells": int(self.mesh.n), "inner_faces": int(self.flow.m), "dt": self.dt, "U": self.U, "nu": self.nu,
                "dt_facts": self.dt_facts, "amg": self.solver.facts, "quality": self.quality,
                "cfl_bulk": 2 * self.U * self.dt / MESHES[CASES[self.cfg["case"]][0]]["h_far"]}


def plant_checkerboard(case, state, kind, eps):
    """The state with a checkerboard of relative size ``eps`` added to its pressure (and to the pressure's gradient)."""
    import jax.numpy as jnp
    p = state["p"]
    scale = float(case.diag(case.tab, case.par, state)["p_scale"])
    n2 = case.mesh.n // case.sectors
    if kind == "odd":
        mode = np.tile(np.where(np.arange(case.sectors) % 2 == 0, 1.0, -1.0), n2)
    else:
        rng = np.random.default_rng(0)
        mode = rng.choice([-1.0, 1.0], size=case.mesh.n)
    p2 = p + jnp.asarray(eps * scale * mode, p.dtype)
    gp2, _ = case.flow.grad_and_jump(case.tab, p2)
    return dict(state, p=p2, gp=gp2)


def run_history(config):
    import jax
    cfg = parse(config)
    t0 = time.time()
    case = Case(cfg)
    state = case.start()
    rows, diags, walls, plants = [], [], [], []
    chunk_steps = BLOCK * BLOCKS
    name = config.replace(":", "_")
    blown = False
    for c in range(cfg["steps"] // chunk_steps):
        t1 = time.time()
        state, (outs, dg) = case.chunk(case.tab, case.T, case.par, state)
        jax.block_until_ready(state["u"])
        walls.append(time.time() - t1)
        rows.append({k: np.asarray(v).reshape((chunk_steps,) + v.shape[2:]) for k, v in outs.items()})
        diags.append({k: np.asarray(v) for k, v in dg.items()})
        if not np.isfinite(diags[-1]["ke"][-1]):
            blown = True
            break
        if c % 5 == 4:   # the checkerboard detector on this state as it is and with each plant
            clean = case.diag(case.tab, case.par, state)
            row = {"step": (c + 1) * chunk_steps, "clean_odd": float(clean["cb_odd"]), "clean_rough": float(clean["cb_rough"])}
            for kind in ("odd", "random"):
                for eps in (0.01, 0.1):
                    d = case.diag(case.tab, case.par, plant_checkerboard(case, state, kind, eps))
                    row[f"{kind}_{eps}_odd"], row[f"{kind}_{eps}_rough"] = float(d["cb_odd"]), float(d["cb_rough"])
            plants.append(row)
        if time.time() - t0 > BUDGET_S / max(1, len(GROUPS.get(os.environ.get("C2_GROUP", ""), [1]))):
            break
    done = len(rows) * chunk_steps
    # persistence: a planted odd-even pressure of 10 % followed for 100 steps, against the same 100 steps without it
    persist = {}
    if not blown and done > 0:
        for label, st in (("clean", state), ("planted", plant_checkerboard(case, state, "odd", 0.1))):
            series = [float(case.diag(case.tab, case.par, st)["cb_odd"])]
            s = st
            for _ in range(20):
                s, _ = case.steps_only(case.tab, case.T, case.par, s, 5)
                series.append(float(case.diag(case.tab, case.par, s)["cb_odd"]))
            persist[label] = series
    cat = lambda key, src: np.concatenate([r[key] for r in src]) if src else np.zeros(0)
    its, res = cat("its", rows), cat("res", rows)
    used = cat("use", rows).astype(bool)
    half = np.arange(len(its)) >= len(its) // 2
    half = half & used if its.size else half
    dkeys = list(diags[0].keys()) if diags else []
    D = {k: np.concatenate([d[k] for d in diags]) for k in dkeys}
    summary = dict(cfg, **case.facts(), steps_done=done, blown=blown, wall_s=float(sum(walls)), first_chunk_s=float(walls[0]) if walls else None,
                   s_per_step=float(np.median(walls[1:]) / chunk_steps) if len(walls) > 1 else None,
                   its_per_solve_all=float(its.mean()) if its.size else None, its_per_solve_late=float(its[half].mean()) if its.size else None,
                   its_per_step_late=float(its[half].sum(axis=1).mean()) if its.size else None, its_max=int(its.max()) if its.size else None,
                   res_late_max=float(res[half].max()) if res.size else None, plants=plants, persist=persist,
                   device=str(jax.devices()[0]), total_s=time.time() - t0)
    if D:
        scale = RADIUS / case.U
        late = slice(len(D["ke"]) // 2, None)
        fz = RHO * (D["fp"][:, 2] + D["fv"][:, 2])
        summary.update(div_l2_max=float(np.nanmax(D["div_l2"]) * scale), div_max_max=float(np.nanmax(D["div_max"]) * scale),
                       div_l2_late=float(np.nanmean(D["div_l2"][late]) * scale),
                       divc_l2_late=float(np.nanmean(D["divc_l2"][late]) * scale), cb_odd_late=float(np.nanmean(D["cb_odd"][late])),
                       cb_odd_max=float(np.nanmax(D["cb_odd"])), cb_rough_late=float(np.nanmean(D["cb_rough"][late])),
                       fz_end=float(fz[-1]), fz_late_mean=float(fz[late].mean()), uz_mean_end=float(D["uz_mean"][-1]),
                       f_lateral_end=float(RHO * np.hypot(*(D["fp"][-1, :2] + D["fv"][-1, :2]))),
                       share=float(RHO * 8.0 * NU * case.U / RADIUS ** 2 * case.mesh.info["body_volume"]))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, name + ".npz"), its=its, res=res, bnorm=cat("bnorm", rows), walls=np.asarray(walls),
                        dt=case.dt, u_end=np.asarray(state["u"], np.float32) if case.mesh.n < 6e4 else np.zeros(0),
                        p_end=np.asarray(state["p"], np.float32) if case.mesh.n < 6e4 else np.zeros(0), **D)
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    keys = ("config", "cells", "dt", "steps_done", "blown", "s_per_step", "its_per_solve_late", "its_per_step_late", "res_late_max",
            "div_l2_late", "divc_l2_late", "cb_odd_late", "cb_rough_late", "fz_late_mean", "fz_end")
    print(json.dumps({k: summary.get(k) for k in keys}, default=float), flush=True)
    jax.clear_caches()
    return summary


def main():
    group = os.environ.get("C2_GROUP", "smoke")
    os.makedirs(OUT, exist_ok=True)
    for config in GROUPS[group]:
        try:
            run_history(config)
        except Exception as e:  # noqa: BLE001
            import traceback
            with open(os.path.join(OUT, config.replace(":", "_") + ".failed.json"), "w", encoding="utf-8") as f:
                json.dump({"config": config, "failed": repr(e)[:500], "trace": traceback.format_exc()[-3000:]}, f)
            print("FAILED", config, repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
