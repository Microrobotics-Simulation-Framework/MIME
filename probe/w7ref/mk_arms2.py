"""Launch 2 (w7ref): where the high-Re error lives (localisation at P3 s1.6) and a p-ladder (P3, P4) on fixed meshes."""
import json
import sys

RE_D = [0, 130, 260, 455, 650]
TUBE = dict(kind="tube", Lu=10.0, Ld=30.0)
SHEAR = dict(kind="shear", Lu=12.0, Ld=36.0, Ly=16.0, H=16.0)
arms = {}
# localisation: one mesh ingredient refined at a time (P3, s 1.6, tube 0.1a); base = TP3-g0.1-s1.6 of launch 1
loc = {"sph": dict(h_sph=0.075), "wake": dict(h_w=0.15), "gap": dict(k=4.0), "wall": dict(h_wall=0.3), "grad": dict(grading=0.2),
       "sphwake": dict(h_sph=0.075, h_w=0.15)}
for key, b in loc.items():
    arms[f"L-{key}"] = dict(TUBE, gap=0.1, s=1.6, order=3, base=b, Re=[0, 260, 650])
arms["L-noise"] = dict(TUBE, gap=0.1, s=1.6, order=3, Lu=10.37, Re=[0, 260, 650])  # another realisation, same sizes
# p-ladder on fixed meshes
for g in (0.1, 0.03, 0.01):
    arms[f"TP4-g{g}-s1.6"] = dict(TUBE, gap=g, s=1.6, order=4, Re=RE_D)
for g in (0.03, 0.01):
    arms[f"TP3-g{g}-s1.6"] = dict(TUBE, gap=g, s=1.6, order=3, Re=RE_D)
arms["TP3-g0.01-s1.26"] = dict(TUBE, gap=0.01, s=1.26, order=3, Re=RE_D)
for g in (0.1, 0.03, 0.01):
    arms[f"SP3-g{g}-s1.6"] = dict(SHEAR, gap=g, s=1.6, order=3, Re=RE_D)
# domain at P3 (less mesh noise than P2 s1.0)
arms["TDP3-long-g0.1"] = dict(TUBE, gap=0.1, s=1.6, order=3, Lu=15.0, Ld=45.0, wake=dict(x_near=8.0, x_mid=30.0), Re=RE_D)
arms["TDP3-short-g0.1"] = dict(TUBE, gap=0.1, s=1.6, order=3, Lu=6.0, Ld=20.0, Re=RE_D)
for n in arms:
    assert len(n) <= 40, n
out = sys.argv[1]
spec = json.load(open(f"{out}/arms_spec.json"))
spec.update(arms)
json.dump(spec, open(f"{out}/arms_spec.json", "w"), indent=1)
open(f"{out}/arms2.txt", "w").write("\n".join(arms) + "\n")
print(len(arms), "arms")
