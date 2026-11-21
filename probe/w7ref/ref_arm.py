"""One REF arm: a mesh level of one case, a continuation in Re, results written after every Re (incremental JSON).

python ref_arm.py <arm-name> <out-dir>     (the arm's spec comes from arms_spec.json beside this file)
"""
import json
import os
import resource
import sys
import time

import numpy as np
import ngsolve as ng

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_core import build_mesh, Flow  # noqa: E402

R_TUBE = 1.75 / 0.675  # tube radius over sphere radius (3.5 mm tube, 1.35 mm sphere)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def case_setup(spec):
    """Dirichlet data, gauge flag and the map Re -> code viscosity for the case."""
    kind = spec["kind"]
    gap = spec["gap"]
    nuf = spec.get("nu_factor", 1.0)
    if kind == "tube":
        R = spec.get("R", R_TUBE)
        cf = {"inlet": ng.CoefficientFunction((2 * (1 - (ng.y ** 2 + ng.z ** 2) / R ** 2), 0, 0))}
        nu_of = lambda Re: nuf * 2 * R / Re  # Re = Re_D = U_m D / nu, U_m = 1, a = 1
        return cf, False, nu_of
    if kind == "shear":
        R = spec.get("R", R_TUBE)
        G = 4.0 / R  # the tube's wall shear rate in U_m / a
        cf = {"box": ng.CoefficientFunction((G * ng.z, 0, 0))}
        nu_of = lambda Re: nuf * 2 * R / Re  # Re is the tube's Re_D at the matching shear
        return cf, False, nu_of
    if kind == "eka":
        l = 1.0 + gap
        cfv = ng.CoefficientFunction((ng.z - l, 0, 0))  # gamma = 1, sphere frame, zero slip at the centre
        cf = {"box": cfv, "wall": cfv}
        nu_of = lambda Re: nuf * 1.0 / Re  # Re = Re_gamma = gamma a^2 / nu
        return cf, True, nu_of
    if kind == "res":
        return {}, True, None
    if kind == "free":
        cf = {"box": ng.CoefficientFunction((1, 0, 0))}
        nu_of = lambda Re: nuf * 2.0 / Re  # Re = U d / nu
        return cf, False, nu_of
    raise ValueError(kind)


def main():
    name, out_dir = sys.argv[1], sys.argv[2]
    specs = json.load(open(os.path.join(HERE, "arms_spec.json")))
    spec = specs[name]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    nthreads = int(os.environ.get("REF_THREADS", "4"))
    ng.SetNumThreads(nthreads)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    rec = dict(arm=name, spec=spec, started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), ngsolve=ng.__version__,
               threads=nthreads, results=[])

    def dump():
        rec["peak_rss_gib"] = rss_gib()
        with open(out_path + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(out_path + ".tmp", out_path)

    t0 = time.time()
    dom = {k: spec[k] for k in ("Lu", "Ld", "Ly", "H", "Rf", "R", "hfar") if k in spec}
    if spec["kind"] in ("tube", "shear") and "R" not in dom:
        dom["R"] = R_TUBE
    mesh, info = build_mesh(spec["kind"], spec["gap"], s=spec["s"], order=spec.get("order", 2), full=spec.get("full", False),
                            wake=spec.get("wake"), **dom)
    info.pop("centre", None)
    rec["mesh"] = info
    log(f"mesh {info['ne']} elements, {info['nv']} vertices, {info['t_mesh']:.1f} s")
    cf, gauge, nu_of = case_setup(spec)
    inverse = spec.get("inverse", "pypardiso")
    with ng.TaskManager():
        fl = Flow(mesh, info | {"centre": build_centre(spec)}, order=spec.get("order", 2), dirichlet_cf=cf, gauge=gauge, log=log)
        nfree, ntot = fl.ndof()
        rec["ndof_free"], rec["ndof_total"] = nfree, ntot
        log(f"dofs free {nfree} total {ntot}; rss {rss_gib():.2f} GiB")
        dump()
        half = 1.0 if spec.get("full", False) else 2.0
        if spec["kind"] == "res":
            rec["resistance"] = resistance(fl, spec, half, log)
            dump()
        for Re in spec.get("Re", []):
            t1 = time.time()
            if Re == 0:
                nu, conv = 1.0, 0.0
            else:
                nu, conv = nu_of(Re), 1.0
            st = fl.newton(nu, conv=conv, inverse=inverse, reuse=spec.get("reuse", False))
            f = fl.forces()
            full = {k: v * (half if k[:2] in ("Fx", "Fz", "Ty") else 1.0) for k, v in f.items()}
            row = dict(Re=Re, nu=nu, conv=conv, code=full, per_mu=({k: v / nu for k, v in full.items()}),
                       newton=st, t=time.time() - t1, rss_gib=rss_gib())
            rec["results"].append(row)
            log(f"Re {Re}: conv {st['converged']} its {st['iters']} factors {st['n_factor']} t {row['t']:.1f}s | per mu: Fx {row['per_mu']['Fx']:.6f}"
                f" Fz {row['per_mu']['Fz']:.6f} Ty {row['per_mu']['Ty']:.6f} Ty_sym {row['per_mu']['Ty_sym']:.6f} (trac g {row['per_mu']['Fx_g']:.5f}); rss {row['rss_gib']:.2f}")
            dump()
            if spec.get("stability") and Re in spec["stability"].get("at", []):
                rec.setdefault("stability", []).append(stability(fl, spec["stability"], Re, log))
                dump()
    rec["t_total"] = time.time() - t0
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    rec["log"] = log_lines[-400:]
    dump()


def build_centre(spec):
    kind, gap = spec["kind"], spec["gap"]
    if kind == "tube":
        R = spec.get("R", R_TUBE)
        return (0.0, 0.0, -(R - 1.0 - gap))
    if kind in ("shear", "eka", "res"):
        return (0.0, 0.0, 1.0 + gap)
    return (0.0, 0.0, 0.0)


def resistance(fl, spec, half, log):
    """Stokes resistances of the held-sphere mesh: the sphere moves rigidly in fluid at rest (wall and box at rest).
    One factorisation, one linear solve per motion; forces by the residual method (on the sphere, by the fluid)."""
    zc = 1.0 + spec["gap"]
    motions = {"tx": ng.CoefficientFunction((1, 0, 0)), "tz": ng.CoefficientFunction((0, 0, 1)),
               "ry": ng.CoefficientFunction((ng.z - zc, 0, -ng.x))}
    out = {}
    fl.nu.Set(1.0)
    fl.conv.Set(0.0)
    fl.a.AssembleLinearization(fl.gfu.vec)
    t0 = time.time()
    from ref_core import _PyPardiso
    inv = _PyPardiso(fl.a.mat, fl.free_mask)
    t_fac = time.time() - t0
    du = fl.gfu.vec.CreateVector()
    for key, cf in motions.items():
        fl.dirichlet_cf = {"sphere": cf}
        fl.set_bc()
        r = fl.residual()
        inv.solve_into(r, du)
        fl.gfu.vec.data -= du
        res_after = fl.free_norm()
        f = fl.forces()
        full = {k: v * (half if k[:2] in ("Fx", "Fz", "Ty") else 1.0) for k, v in f.items()}
        out[key] = dict(code=full, residual_after=res_after)
        log(f"  resistance {key}: Fx {full['Fx_sym']:.6f} Fz {full['Fz_sym']:.6f} Ty {full['Ty_sym']:.6f} (lap Ty {full['Ty']:.6f};"
            f" traction s Ty {full['Ty_s']:.6f}; residual {res_after:.1e})")
    out["t_factor"] = t_fac
    return out


def stability(fl, sspec, Re, log):
    """Linear stability of the steady state by a time march of the linearised equations: Crank-Nicolson on
    M d/dt x + J x = 0 (M the velocity mass), which grows exactly when an eigenvalue has a positive real part.
    Starts from a seeded random velocity made solenoidal by one implicit Euler step."""
    dt = sspec.get("dt", 0.1)
    T = sspec.get("T", 200.0)
    X = fl.X
    tr, te = X.TrialFunction(), X.TestFunction()
    m = ng.BilinearForm(X)
    m += ng.InnerProduct(tr[0], te[0]) * ng.dx
    m.Assemble()
    fl.a.AssembleLinearization(fl.gfu.vec)
    J = fl.a.mat
    fd = X.FreeDofs()
    A_be = m.mat.CreateMatrix()
    A_be.AsVector().data = (1.0 / dt) * m.mat.AsVector() + J.AsVector()
    A_cn = m.mat.CreateMatrix()
    A_cn.AsVector().data = (1.0 / dt) * m.mat.AsVector() + 0.5 * J.AsVector()
    B_cn = m.mat.CreateMatrix()
    B_cn.AsVector().data = (1.0 / dt) * m.mat.AsVector() - 0.5 * J.AsVector()
    inv_be = A_be.Inverse(fd, inverse="pardiso")
    inv_cn = A_cn.Inverse(fd, inverse="pardiso")
    x = ng.GridFunction(X)
    rng = np.random.default_rng(sspec.get("seed", 0))
    arr = x.vec.FV().NumPy()
    nv = fl.V.ndof
    arr[:nv] = rng.standard_normal(nv) * fl.free_mask[:nv]
    rhs = x.vec.CreateVector()
    tmp = x.vec.CreateVector()
    rhs.data = (1.0 / dt) * m.mat * x.vec
    x.vec.data = inv_be * rhs
    norms, times, fyz = [], [], []
    n = int(round(T / dt))
    t0 = time.time()
    for k in range(n):
        rhs.data = B_cn * x.vec
        x.vec.data = inv_cn * rhs
        if k % 10 == 0 or k == n - 1:
            tmp.data = m.mat * x.vec
            e = float(np.sqrt(abs(ng.InnerProduct(tmp, x.vec))))
            norms.append(e)
            times.append((k + 1) * dt)
            if e > 0:
                x.vec.data *= 1.0 / e  # renormalise; growth is accumulated in the log
                norms[-1] = e
    # growth rate: each recorded norm is the growth since the last renormalisation (10 steps, or fewer at the end)
    lg = np.log(np.array(norms[1:]))
    dts = np.diff(np.array(times))
    rates = lg / dts
    res = dict(Re=Re, dt=dt, T=T, rate_last=float(rates[-5:].mean()) if len(rates) >= 5 else None,
               rates_tail=[float(r) for r in rates[-20:]], rates_head=[float(r) for r in rates[:10]], t=time.time() - t0)
    log(f"  stability Re {Re}: late growth rate {res['rate_last']} (per a/U), {res['t']:.0f} s")
    return res


if __name__ == "__main__":
    main()
