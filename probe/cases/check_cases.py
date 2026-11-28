"""Check every declared case: build, mesh coarsely, compare boundary vertices with the analytic definitions in geom.py.
Writes one JSON line per case. Usage: python -I check_cases.py out.jsonl [case ...]"""
import sys, json, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G
import occ_build as B
from netgen.occ import OCCGeometry
from netgen.meshing import MeshingParameters
from ngsolve import Mesh, Integrate, CoefficientFunction, BND, VOL


def vertex_check(mesh, name, fn):
    ng = mesh.ngmesh
    P = np.array([p.p for p in ng.Points()])
    idx = set()
    for el in mesh.Elements(BND):
        if el.mat == name:
            for v in el.vertices:
                idx.add(v.nr)
    idx = np.array(sorted(idx))
    if len(idx) == 0:
        return None, 0
    r = fn(P[idx])
    return float(np.max(np.abs(r))), len(idx)


def run(case):
    t0 = time.time()
    out = {"case": case}
    gap = 0.1
    Rm = np.eye(3)
    body = G.Sphere()
    if case == "G0":
        w = G.Plane(); c = G.sphere_centre(w, gap); lum = B.lumen_G0(10.0); cuts = ("y", "x:0")
    elif case == "G0b":
        w = G.HeldSphere(4.0); c = G.sphere_centre(w, gap); lum = B.lumen_G0b(4.0, 12.0); cuts = ("y", "x:0")
    elif case == "G1":
        w = G.Tube(G.R_SPH); c = G.sphere_centre(w, gap); lum = B.lumen_G1(w.R, -3 * w.R, 3 * w.R); cuts = ("y", "x:0")
    elif case == "G2":
        w = G.Bend(G.R_SPH); c = G.sphere_centre(w, gap); lum = B.lumen_G2(w); cuts = ("y",)
    elif case == "G3":
        w = G.Stenosis(G.R_SPH, 0.9 * G.R_SPH, G.R_SPH, G.R_SPH); c = G.sphere_centre(w, gap)
        lum = B.lumen_G3(w, -w.L_in, w.L_in); cuts = ("y", "x:0")
    elif case.startswith("G4"):
        w = G.Bifurcation(G.R_SPH); pose = case.split("-")[1]; c = G.sphere_centre(w, gap, pose)
        lum = B.lumen_G4(w); cuts = ("y", "z") if pose == "P1" else ("y",)
    elif case == "H-G1":
        w = G.Tube(G.R_HF); body = G.Helix(); Rm = G.rot_body_to_lab(0.3)
        c = G.press(w, body, np.zeros(3), [0, 0, -1], gap, Rm, lo=0.0, hi=1.5)
        lum = B.lumen_G1(w.R, -3 * w.R, 3 * w.R); cuts = ()
    elif case == "F-G1":
        w = G.Tube(G.R_HF); body = G.Fins(); Rm = G.rot_body_to_lab(0.3)
        c = G.press(w, body, np.zeros(3), [0, 0, -1], gap, Rm, lo=0.0, hi=1.5)
        lum = B.lumen_G1(w.R, -3 * w.R, 3 * w.R); cuts = ()
    else:
        raise ValueError(case)
    out["centre"] = c.tolist()
    if body.name == "SPH":
        bsh = B.body_sphere(c)
        g_exact = float(w.sdf(np.atleast_2d(c))[0] - 1.0)
        # brute-force check that the nearest wall point is where the closed form says: sample the sphere surface
        u = np.random.default_rng(0).normal(size=(200000, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        out["gap_closed_form"] = g_exact
        out["gap_sampled_min"] = float(np.min(w.sdf(c + u)))
    elif body.name == "H":
        bsh = B.body_helix(c, Rm, body)
        out["gap"] = G.body_gap(w, body, c, Rm)
    else:
        bsh = B.body_fins(c, Rm, body)
        out["gap"] = G.body_gap(w, body, c, Rm)
    fl = B.fluid(lum, bsh, cuts)
    out["occ_volume"] = fl.mass
    out["face_names"] = sorted(set(f.name for f in fl.faces))
    geo = OCCGeometry(fl)
    mp = MeshingParameters(maxh=0.6, grading=0.3)
    # refine at the contact: points on the body nearest the wall
    pts = (c + np.random.default_rng(1).normal(size=(4000, 3)) * 0.5)
    for p in pts:
        g = float(w.sdf(np.atleast_2d(p))[0])
        if 0 < g < 0.4:
            mp.RestrictH(float(p[0]), float(p[1]), float(p[2]), max(0.5 * g, 0.03))
    mesh = Mesh(geo.GenerateMesh(mp=mp))
    out["nv"], out["ne"] = mesh.nv, mesh.ne
    mesh.Curve(3)
    out["mesh_volume_curved3"] = Integrate(CoefficientFunction(1.0), mesh, VOL)
    out["wall_vertex_maxdev"], out["wall_nv"] = vertex_check(mesh, "wall", w.sdf)
    out["body_vertex_maxdev"], out["body_nv"] = vertex_check(mesh, "body", lambda P: body.sdf_body((P - c) @ Rm))
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    cases = sys.argv[2:] or ["G0", "G0b", "G1", "G2", "G3", "G4-P1", "G4-P2", "G4-P3", "H-G1", "F-G1"]
    with open(sys.argv[1], "a") as f:
        for cs in cases:
            try:
                r = run(cs)
            except Exception as e:
                import traceback
                r = {"case": cs, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
            print(json.dumps(r), flush=True)
            f.write(json.dumps(r) + "\n"); f.flush()
