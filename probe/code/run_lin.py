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
    if w is None:
        pts = [(c + p, spec["hbody"]) for p in np.random.default_rng(0).normal(size=(4000, 3))]
        pts = [(c + (p - c) / np.linalg.norm(p - c), h) for p, h in pts]
    elif body.name == "SPH":
        pts = lin.grade_points_sphere(w, c, gap, spec["cfac"], spec["hbody"], gmax=spec.get("gmax", 0.6))
    else:
        raise NotImplementedError("grading for H, F: next step")
    return dict(w=w, c=c, Rm=Rm, fluid=fl, pts=pts, cuts=cuts, bg=bg if spec.get("F0") else None, bgpar=bgpar, body=body)


def main():
    out = sys.argv[1]
    for sj in sys.argv[2:]:
        spec = json.loads(sj)
        t0 = time.time()
        b = build(spec)
        mult = 2 ** len(b["cuts"])
        for order in spec.get("orders", [2]):
            mesh, tm = lin.make_mesh(b["fluid"], b["pts"], spec.get("maxh", 4.0), spec.get("grading", 0.3), order=order)
            for s in spec.get("s", [0]):
                rec = dict(spec=spec, order=order, s=s, centre=b["c"].tolist(), nv=mesh.nv, ne=mesh.ne, t_mesh=tm, mult=mult)
                try:
                    info = lin.solve_linear(mesh, order, s, b["c"], b["cuts"], mult, bg=b["bg"], bg_parity=b["bgpar"],
                                            log=lambda m: print(m, flush=True))
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
