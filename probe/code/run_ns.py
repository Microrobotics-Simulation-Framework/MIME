"""NS arms for SPH in G1 (half domain y >= 0). Usage: python -I run_ns.py out.jsonl '<spec json>'
spec: gap, cfac, hbody, hwake, order, Lin (10), Lout (30), points: [{Re, V, Om, flow, label}], tag.
Frame: Type B (body frame): wall at -V, inlet Poiseuille*flow - V, body spins at Om about y through its centre."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom as G
import occ_build as B
import lin, ns
from ngsolve import SetNumThreads

SetNumThreads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def main():
    out, spec = sys.argv[1], json.loads(sys.argv[2])
    R = spec.get("R", G.R_SPH)
    w = G.Tube(R)
    gap = spec["gap"]
    c = G.sphere_centre(w, gap)
    lum = B.lumen_G1(R, c[0] - spec.get("Lin", 10.0), c[0] + spec.get("Lout", 30.0))
    fl = B.fluid(lum, B.body_sphere(c), ["y"])
    pts = lin.grade_points_sphere(w, c, gap, spec["cfac"], spec["hbody"], gmax=spec.get("gmax", 0.6))
    hw = spec.get("hwake", 0.3)
    for xx in np.arange(c[0], c[0] + spec.get("Lwake", 12.0), hw / 2):
        grow = hw * (1 + 0.08 * (xx - c[0]))
        for rr in np.linspace(0, 1.6, 9):
            for ph in np.linspace(0, np.pi, 7):
                p = c + np.array([xx - c[0], rr * np.sin(ph), rr * np.cos(ph)])
                if w.sdf(np.atleast_2d(p))[0] > 0.05:
                    pts.append((p, grow))
    t0 = time.time()
    mesh, tm = lin.make_mesh(fl, pts, spec.get("maxh", 1.5), spec.get("grading", 0.3), order=spec["order"])
    head = dict(spec={k: v for k, v in spec.items() if k != "points"}, centre=c.tolist(), nv=mesh.nv, ne=mesh.ne, t_mesh=tm)
    with open(out, "a") as f:
        f.write(json.dumps(dict(head, kind="mesh")) + "\n")

    def log(m):
        print(m, flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(dict(json.loads(m), tag=spec.get("tag"), order=spec["order"], gap=gap)) + "\n")
    ns.solve_ns(mesh, spec["order"], R, c, spec["points"], log=log)


if __name__ == "__main__":
    main()
