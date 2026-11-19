"""Spike W2F, the graph's arms: CORE spike 2's unsteady step (``probe/c2/flow.py``, unchanged) on a held body in a tube, with
three drives:

  ``force``   spike 2's: a uniform body force fixed at the Poiseuille value of the nominal flow rate (the flow rate floats);
  ``flux``    the same body force, set every step by a PI controller holding the lumen's mean axial velocity at U;
  ``settle``  the settling sphere in the sphere's frame: the tube wall moves at V, no body force, and a fringe region
              (Nordstrom, Nordin and Henningson 1999) forces the flow back to the plug V upstream of the body, so the
              periodic tube stands in for an infinite one.

A configuration is ``<mesh>:<Re>:<drive>:<t_end ms>[:option]``; options ``nu105`` (planted viscosity x 1.05). ``Re`` is
spike 2's ``Re_D`` (tube diameter, mean velocity) for ``force`` and ``flux``, and the sphere's ``Re = V d / nu`` for
``settle``. Results: ``<tag>.npz`` (series every 10 steps) and ``<tag>.json`` in ``W2F_OUT``."""
import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "c2"))
sys.path.insert(0, HERE)
OUT = os.environ.get("W2F_OUT", os.path.join(os.path.dirname(HERE), "out"))
NU, RHO, RADIUS = 3.3e-6, 1060.0, 1.75e-3
A = 0.4 * RADIUS
# spike 2's periodic tube (8 R long, body at its centre) and the long one of the settling arm (16 R, body 4 R from its
# inlet end, the fringe over its last 3 R)
SHORT = dict(length=8.0 * RADIUS, zc=0.0)
LONG = dict(length=16.0 * RADIUS, zc=4.0 * RADIUS)
MESHES = {
    # spike 2's, unchanged
    "T": dict(SHORT, aspect=1.0, h_body=A / 5, h_rim=A / 5, h_far=320e-6, sectors=8),
    "S": dict(SHORT, aspect=1.0, h_body=A / 8, h_rim=A / 8, h_far=140e-6, sectors=16),
    "SL": dict(SHORT, aspect=1.0, h_body=A / 14, h_rim=A / 14, h_far=75e-6, sectors=24),
    "D": dict(SHORT, aspect=0.05, h_body=A / 10, h_rim=A / 40, h_far=165e-6, sectors=16),
    "DT": dict(SHORT, aspect=0.05, h_body=A / 6, h_rim=A / 20, h_far=320e-6, sectors=8),
    # added: a rung between S and SL, and a rung above the spheroid's D
    "M": dict(SHORT, aspect=1.0, h_body=A / 11, h_rim=A / 11, h_far=100e-6, sectors=20),
    "DM": dict(SHORT, aspect=0.05, h_body=A / 14, h_rim=A / 56, h_far=120e-6, sectors=20),
    # the settling arm's long tube
    "BT": dict(LONG, aspect=1.0, h_body=A / 5, h_rim=A / 5, h_far=320e-6, sectors=8),
    "BS": dict(LONG, aspect=1.0, h_body=A / 8, h_rim=A / 8, h_far=140e-6, sectors=16),
    "BM": dict(LONG, aspect=1.0, h_body=A / 11, h_rim=A / 11, h_far=100e-6, sectors=20),
    "BL": dict(LONG, aspect=1.0, h_body=A / 14, h_rim=A / 14, h_far=75e-6, sectors=24),
}
BLOCK, CHUNK = 10, 20          # diagnostics every BLOCK steps; CHUNK blocks a compiled call


def parse(config):
    parts = config.split(":")
    cfg = {"config": config, "mesh": parts[0], "re": float(parts[1]), "drive": parts[2], "t_end": float(parts[3]) * 1e-3,
           "nu_scale": 1.0, "dt_scale": 1.0}
    for o in parts[4:]:
        if o == "nu105":
            cfg["nu_scale"] = 1.05
        elif o.startswith("dt"):
            cfg["dt_scale"] = float(o[2:])
        else:
            raise ValueError(o)
    return cfg


_MESH = {}


def get_mesh(name):
    if name not in _MESH:
        import tube2
        t0 = time.time()
        spec = MESHES[name]
        nodes, tris, version = tube2.meridional(RADIUS, spec["length"], A, spec["aspect"], spec["h_body"], spec["h_rim"],
                                                spec["h_far"], zc=spec["zc"])
        m = tube2.revolved(nodes, tris, spec["sectors"], RADIUS, spec["length"], A, spec["aspect"], zc=spec["zc"])
        m.info["gmsh"] = version
        q = m.quality()
        q.update(m.info, build_s=time.time() - t0, mesh=name)
        _MESH[name] = (m, q)
    return _MESH[name]


def make_flow_class():
    import flow as fl
    import jax.numpy as jnp
    from tube2 import WALL

    class Flow2(fl.Flow):
        """Spike 2's operators plus a tube wall moving along the axis at ``uwall`` (its viscous flux and its value in the
        cells' Gauss gradient) and a fringe forcing ``-sigma(z) (u - u_target)``."""

        def __init__(self, mesh, impl, uwall=0.0, sigma=None, target=None):
            super().__init__(mesh, impl)
            n = mesh.n
            inner = mesh.inner
            bo, Sb, xfb, pb = mesh.own[~inner], mesh.S[~inner], mesh.xf[~inner], mesh.patch[~inner]
            areab = np.linalg.norm(Sb, axis=1)
            gb = areab / (np.einsum("fi,fi->f", xfb - mesh.xc[bo], Sb) / areab)
            wall = pb == WALL
            ez = np.array([0.0, 0.0, 1.0])
            self.tab["wall_flux_uw"] = np.bincount(bo[wall], gb[wall], n)[:, None] * (uwall * ez)[None, :]
            wS = np.zeros((n, 3))
            np.add.at(wS, bo[wall], Sb[wall])
            self.tab["wall_grad_uw"] = (uwall * ez)[None, :, None] * wS[:, None, :]          # u_w (x) S, summed per cell
            self.tab["sigma"] = np.zeros(n) if sigma is None else np.asarray(sigma, float)
            self.tab["target"] = np.zeros((n, 3)) if target is None else np.asarray(target, float)

        def rhs(self, tab, par, u, flux, extra, nonorth=True):
            nu, force = par["nu"], par["force"]
            uo, un = self.ends(tab, u)
            w = tab["w"][:, None]
            ufl = w * uo + (1.0 - w) * un
            fc = (ufl * tab["S"]).sum(axis=-1)
            F = fc + extra if flux is None else flux
            msg = nu * tab["g"][:, None] * (un - uo) - F[:, None] * (0.5 * (uo + un))
            if nonorth:
                S = tab["S"]
                grad = (self._net(tab["net"], ufl[:, :, None] * S[:, None, :]) + tab["wall_grad_uw"]) / tab["vol"][:, None, None]
                go, gn = self.ends(tab, grad.reshape(self.n, 9))
                gf = (w * go + (1.0 - w) * gn).reshape(-1, 3, 3)
                k = tab["k"]
                msg = msg + nu * (gf[:, :, 0] * k[:, 0:1] + gf[:, :, 1] * k[:, 1:2] + gf[:, :, 2] * k[:, 2:3])
            R = (self._net(tab["net"], msg) - nu * tab["wall_g"][:, None] * u + nu * tab["wall_flux_uw"]) / tab["vol"][:, None] + force
            R = R - tab["sigma"][:, None] * (u - tab["target"])
            return R, fc

    return Flow2


class Case:
    def __init__(self, cfg, impl=None):
        import jax.numpy as jnp

        import flow as fl
        self.cfg, self.fl = cfg, fl
        spec = MESHES[cfg["mesh"]]
        self.mesh, self.quality = get_mesh(cfg["mesh"])
        m = self.mesh
        self.sectors = spec["sectors"]
        impl = impl or ("padded" if jax.devices()[0].platform == "gpu" else "scatter")
        drive = cfg["drive"]
        rho2 = m.xc[:, 0] ** 2 + m.xc[:, 1] ** 2
        z = m.xc[:, 2]
        self.nu = NU * cfg["nu_scale"]
        if drive == "settle":
            self.V = cfg["re"] * NU / (2 * A)            # the sphere's speed through the tube; Re on the true viscosity
            speed = 2.5 * self.V
            zlo, zhi = spec["zc"] - spec["length"] / 2, spec["zc"] + spec["length"] / 2
            # fringe over the last 3 R before the inlet end: rising over R, flat, falling over 0.5 R
            s = lambda x: np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, 1.0 / (1.0 + np.exp(np.clip(1.0 / (x - 1.0) + 1.0 / x, -700, 700)))))
            f0, f1 = zhi - 3.0 * RADIUS, zhi
            shape = s((z - f0) / RADIUS) - s((z - (f1 - 0.5 * RADIUS)) / (0.5 * RADIUS))
            self.sigma_max = max(6.7 * self.V / RADIUS, 50.0 * self.nu / RADIUS ** 2)
            sigma = self.sigma_max * shape
            target = np.zeros((m.n, 3))
            target[:, 2] = self.V
            self.flow = make_flow_class()(m, impl, uwall=self.V, sigma=sigma, target=target)
            u0 = target.copy()
            self.U = self.V
            self.f0 = 0.0
        else:
            self.U = cfg["re"] * NU / (2 * RADIUS)
            speed = 3.0 * self.U
            self.flow = make_flow_class()(m, impl)
            u0 = np.zeros((m.n, 3))
            u0[:, 2] = 2.0 * self.U * np.clip(1.0 - rho2 / RADIUS ** 2, 0.0, None)
            self.f0 = 8.0 * NU * self.U / RADIUS ** 2       # spike 2's drive: the nominal Poiseuille gradient on the true viscosity
        dt, self.dt_facts = self.flow.stable_dt(self.nu, speed)
        self.dt = dt * cfg["dt_scale"]
        self.steps = int(np.ceil(cfg["t_end"] / self.dt / (BLOCK * CHUNK))) * BLOCK * CHUNK
        self.solver = fl.Solver(self.flow.A, jnp.float64, guess="proj", depth=4, tol=1e-8)
        self.tab = fl.prims.as_device(self.flow.tab, jnp.float64)
        self.T = self.solver.tables()
        f = lambda v: jnp.asarray(v, jnp.float64)
        self.par = {"nu": f(self.nu), "dt": f(self.dt), "force": f([0.0, 0.0, self.f0])}
        self.u0 = u0
        step = fl.make_step(self.flow, self.solver, "l1", rc=True, carry=False)     # spike 2's l1rc
        flow, f0 = self.flow, self.f0
        total_np = float(np.sum(self.flow.tab["vol"]))
        # the controller holds the lumen's volume mean, which is the flux's mean over the tube's section times the tube's
        # volume over the lumen's (the body displaces 1.1 % of spike 2's tube)
        U_ctrl = self.U * np.pi * RADIUS ** 2 * spec["length"] / total_np
        self.U_ctrl = U_ctrl
        tau = 2e-3                                           # the flux controller's time constant (s), critically damped
        kp, ki = 2.0 / tau, 1.0 / tau ** 2
        control = drive == "flux"

        def one(tab, T, par0, c):
            s, drv, integ = c
            vol = tab["vol"]
            par = dict(par0, force=jnp.asarray([0.0, 0.0, 1.0], jnp.float64) * drv)
            s, o = step(tab, T, par, s)
            uz = jnp.sum(vol * s["u"][:, 2]) / jnp.sum(vol)
            if control:
                e = U_ctrl - uz
                integ = integ + e * par["dt"]
                drv = f0 + kp * e + ki * integ
            return (s, drv, integ), o["its"][0]

        def block(tab, T, par, c):
            c, its = jax.lax.scan(lambda cc, _: one(tab, T, par, cc), c, None, length=BLOCK)
            s, drv, _ = c
            vol = tab["vol"]
            total = jnp.sum(vol)
            fp, fv = flow.body_force(tab, par, s["u"], s["p"])
            div = flow.divergence(tab, s["F"])
            d = {"fp": fp, "fv": fv, "uz_mean": jnp.sum(vol * s["u"][:, 2]) / total, "drive": drv, "its": jnp.sum(its),
                 "ke": 0.5 * jnp.sum(vol[:, None] * s["u"] ** 2), "div_l2": jnp.sqrt(jnp.sum(vol * div * div) / total)}
            return c, d

        self.chunk = jax.jit(lambda tab, T, par, c: jax.lax.scan(lambda cc, _: block(tab, T, par, cc), c, None, length=CHUNK))

    def start(self):
        import jax.numpy as jnp
        st = self.fl.initial_state(self.flow, self.solver, self.u0, jnp.float64)
        return (st, jnp.asarray(self.f0, jnp.float64), jnp.zeros((), jnp.float64))

    def facts(self):
        return {"cells": int(self.mesh.n), "inner_faces": int(self.flow.m), "dt": self.dt, "U": self.U, "nu": self.nu, "f0": self.f0,
                "steps": self.steps, "dt_facts": self.dt_facts, "quality": self.quality, "amg": self.solver.facts,
                "body_volume": float(self.mesh.info["body_volume"]), "sigma_max": getattr(self, "sigma_max", 0.0),
                "U_ctrl": self.U_ctrl, "lumen_volume": float(np.sum(self.flow.tab["vol"]))}


def run(config, budget_s):
    cfg = parse(config)
    t0 = time.time()
    case = Case(cfg)
    c = case.start()
    tag = config.replace(":", "_")
    D, walls, blown = [], [], False
    n_chunks = case.steps // (BLOCK * CHUNK)
    print(json.dumps({"start": tag, "cells": case.mesh.n, "dt": case.dt, "steps": case.steps}), flush=True)
    for i in range(n_chunks):
        t1 = time.time()
        c, d = case.chunk(case.tab, case.T, case.par, c)
        jax.block_until_ready(c[0]["u"])
        walls.append(time.time() - t1)
        D.append({k: np.asarray(v) for k, v in d.items()})
        if not np.isfinite(D[-1]["ke"]).all():
            blown = True
            break
        if time.time() - t0 > budget_s:
            break
    S = {k: np.concatenate([d[k] for d in D]) for k in D[0]}
    t = (np.arange(len(S["ke"])) + 1) * BLOCK * case.dt
    fz = RHO * (S["fp"][:, 2] + S["fv"][:, 2])
    share = RHO * S["drive"] * case.mesh.info["body_volume"]
    summary = dict(cfg, **case.facts(), tag=tag, steps_done=int(len(t) * BLOCK), t_done=float(t[-1]), blown=blown,
                   s_per_step=float(np.median(walls[1:]) / (BLOCK * CHUNK)) if len(walls) > 1 else None, wall_s=float(sum(walls)),
                   total_s=time.time() - t0, device=str(jax.devices()[0]), fz_end=float(fz[-1]), uz_end=float(S["uz_mean"][-1]),
                   drive_end=float(S["drive"][-1]), share_end=float(share[-1]), its_mean=float(S["its"].mean() / BLOCK))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), t=t, fz=fz, share=share, **S)
    with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("tag", "cells", "dt", "steps_done", "t_done", "blown", "s_per_step", "fz_end", "uz_end",
                                               "drive_end", "its_mean")}, default=float), flush=True)
    jax.clear_caches()
    return summary


def main():
    configs = os.environ["W2F_CONFIGS"].split(",")
    budget = float(os.environ.get("W2F_BUDGET_S", "17000"))
    for config in configs:
        try:
            run(config, budget / len(configs))
        except Exception as e:  # noqa: BLE001
            import traceback
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, config.replace(":", "_") + ".failed.json"), "w", encoding="utf-8") as f:
                json.dump({"config": config, "failed": repr(e)[:500], "trace": traceback.format_exc()[-3000:]}, f)
            print("FAILED", config, repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
