"""Driver for linear arms. Usage: python -I run_lin.py out.jsonl '<json spec>' ...
spec keys: geom (G0, G0b, G1, G1h (a/R 0.5), G2, G3, G4), pose (G4), body (SPH, H, F), gap, cfac, hbody, maxh, orders [2,3],
s [0,130,1381], Lc (container), win (window half-length in R), cuts ("auto" or list), F0 (bool), spin (H, F), tag."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G
import occ_build as B
import lin
from ngsolve import CoefficientFunction, x, y, z, SetNumThreads

SetNumThreads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def build(spec):
    gname, gap = spec["geom"], spec["gap"]
    body = {"SPH": G.Sphere(), "H": G.Helix(), "F": G.Fins()}[spec.get("body", "SPH")]
    Rm = np.eye(3)
    bg, bgpar = None, None
    if gname == "G0":
        w = G.Plane(); c = G.sphere_centre(w, gap); lum = B.lumen_G0(spec.get("Lc", 30.0)); cuts = ["y", "x"]
    elif gname == "G0b":
        w = G.HeldSphere(spec.get("b", 4.0)); c = G.sphere_centre(w, gap); lum = B.lumen_G0b(w.b, spec.get("Lc", 30.0)); cuts = ["y", "x"]
    elif gname == "Bunb":     # unbounded sphere in a spherical container of radius Lc
        w = None; c = np.zeros(3)
        from netgen.occ import Sphere as S_, Pnt
        lum = S_(Pnt(0, 0, 0), spec.get("Lc", 30.0)); lum.faces.name = "outer"; cuts = ["y", "x", "z"]
    elif gname in ("G1", "G1h"):
        R = G.R_SPH if (gname == "G1" and body.name == "SPH") else G.R_HF
        w = G.Tube(R)
        L = spec.get("win", 6.0) * R
        if body.name == "SPH":
            c = G.sphere_centre(w, gap); cuts = ["y", "x"]
        else:
            Rm = G.rot_body_to_lab(spec.get("spin", 0.0))
            c = G.press(w, body, np.zeros(3), [0, 0, -1], gap, Rm, lo=0.0, hi=R - 0.5)
            cuts = []
        lum = B.lumen_G1(R, c[0] - L, c[0] + L)
        bg = CoefficientFunction((2.0 * (1 - (y * y + z * z) / R ** 2), 0, 0)); bgpar = {"y": +1, "x": -1, "z": +1}
    elif gname in ("G2", "G3", "G4"):
        R = G.R_SPH if body.name == "SPH" else G.R_HF
        if gname == "G2":
            w = G.Bend(R); lum = B.lumen_G2(w); cuts = ["y"]
            inlet = (w.P_in - w.L_in * w.t_in, w.t_in)
        elif gname == "G3":
            Rs = 0.9 * R if body.name == "SPH" else 1.5
            w = G.Stenosis(R, Rs, R, R); lum = B.lumen_G3(w, -w.L_in, w.L_out); cuts = ["y"]
            inlet = (np.array([-w.L_in, 0, 0]), np.array([1.0, 0, 0]))
        else:
            w = G.Bifurcation(R); lum = B.lumen_G4(w); cuts = ["y", "z"] if spec.get("pose") == "P1" else ["y"]
            inlet = (np.array([-w.L_in, 0, 0]), np.array([1.0, 0, 0]))
        if body.name == "SPH":
            c = G.sphere_centre(w, gap, spec.get("pose"))
        else:
            Rm = G.rot_body_to_lab(spec.get("spin", 0.0))
            c = G.press(w, body, np.zeros(3), [0, 0, -1], gap, Rm, lo=0.0, hi=R - 0.5)
            cuts = []
        bgpar = {"y": +1, "z": +1}
        bg = ("background", inlet, R)
    else:
        raise ValueError(gname)
    if spec.get("cuts", "auto") != "auto":
        cuts = spec["cuts"]
    if body.name == "SPH":
        bsh = B.body_sphere(c)
    elif body.name == "H":
        bsh = B.body_helix(c, Rm, body)
    else:
        bsh = B.body_fins(c, Rm, body)
    occ_cuts = [("x:%.17g" % c[0]) if k == "x" else k for k in cuts]
    fl = B.fluid(lum, bsh, occ_cuts)
    if isinstance(bg, tuple):           # two materials: fluid and the body's interior, for the body-free background solve
        from netgen.occ import Glue
        sol = B.fluid(bsh, None, occ_cuts)
        fl.mat("fluid"); sol.mat("solid")
        fl = Glue([fl, sol])
    if w is None:
        pts = [(c + p, spec["hbody"]) for p in np.random.default_rng(0).normal(size=(4000, 3))]
        pts = [(c + (p - c) / np.linalg.norm(p - c), h) for p, h in pts]
    elif body.name == "SPH":
        pts = lin.grade_points_sphere(w, c, gap, spec["cfac"], spec["hbody"], gmax=spec.get("gmax", 0.6))
    elif body.name == "H":
        H = body
        sc_u = float(np.sqrt(H.rh ** 2 + (H.pitch / (2 * np.pi)) ** 2)) * (1 + H.rho / 0.846)
        surf = lambda u, v: c + lin.helix_surface(H, u, v) @ Rm.T
        pts = lin.grade_points_param(w, surf, (0.0, H.t1), (0.0, 2 * np.pi), sc_u, H.rho, spec["cfac"], spec["hbody"],
                                     gmax=spec.get("gmax", 0.6), n0=(int(H.t1 * sc_u / spec["hbody"]) + 1, 16))
        u = np.random.default_rng(2).normal(size=(400, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        for e in (H.X(np.array([0.0]))[0], H.X(np.array([H.t1]))[0]):
            for q in u:
                pts.append((c + Rm @ (e + H.rho * q), spec["hbody"]))
    else:
        raise NotImplementedError("grading for F: not built")
    return dict(w=w, c=c, Rm=Rm, fluid=fl, pts=pts, cuts=cuts, bg=bg if spec.get("F0") or isinstance(bg, tuple) else None,
                bgpar=bgpar, body=body)


def main():
    out = sys.argv[1]
    for sj in sys.argv[2:]:
        spec = json.loads(sj)
        t0 = time.time()
        b = build(spec)
        mult = 2 ** len(b["cuts"])
        for order in spec.get("orders", [2]):
            mesh, tm = lin.make_mesh(b["fluid"], b["pts"], spec.get("maxh", 4.0), spec.get("grading", 0.3), order=order)
            geo_fix = {}
            corr = {}
            import exactgeo
            if spec["geom"] == "G4" and "carina" in mesh.GetBoundaries():
                corr["carina"] = exactgeo.carina_correction(b["w"])
            if b["body"].name == "H":
                corr["body"] = exactgeo.helix_correction(b["body"], b["c"], b["Rm"])
            if corr:
                mag, _ = exactgeo.apply(mesh, order, corr)
                geo_fix = {"l2_disp_" + "+".join(corr): mag}
            bgcf, bginfo = b["bg"], {}
            if isinstance(bgcf, tuple):
                bgcf, bginfo = lin.background(mesh, order, bgcf[1], bgcf[2], b["cuts"], log=lambda m: print(m, flush=True))
                if not spec.get("F0"):
                    bgcf = None
            for s in spec.get("s", [0]):
                rec = dict(spec=spec, order=order, s=s, centre=b["c"].tolist(), nv=mesh.nv, ne=mesh.ne, t_mesh=tm, mult=mult)
                try:
                    info = lin.solve_linear(mesh, order, s, b["c"], b["cuts"], mult, bg=bgcf, bg_parity=b["bgpar"],
                                            log=lambda m: print(m, flush=True), fluid_only=bool(bginfo))
                    info["background"] = bginfo
                    info["geo_fix"] = geo_fix
                    rec.update(info)
                except Exception as e:
                    import traceback
                    rec["error"] = repr(e); rec["tb"] = traceback.format_exc()[-1200:]
                rec["t_total"] = time.time() - t0
                rec["rss_gib"] = lin.rss_gib()
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("DONE", spec.get("tag"), order, s, rec.get("rss_gib"), flush=True)


if __name__ == "__main__":
    main()
