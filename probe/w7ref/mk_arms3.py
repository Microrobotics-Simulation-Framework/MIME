"""Launch 3 (w7ref): the sphere-surface-refined P3 family for the tube, its noise and domain; shear stability; a slimmer
shear mesh that lets P3 fit a runner."""
import json
import sys

RE_D = [0, 130, 260, 455, 650]
TUBE = dict(kind="tube", Lu=10.0, Ld=30.0)
SPH2 = dict(h_sph=0.075)
arms = {}
for g in (0.1, 0.03, 0.01):
    arms[f"LS-g{g}"] = dict(TUBE, gap=g, s=1.6, order=3, base=SPH2, Re=RE_D)
arms["LS-noise-g0.1"] = dict(TUBE, gap=0.1, s=1.6, order=3, base=SPH2, Lu=10.37, Re=RE_D)
arms["LS-long-g0.1"] = dict(TUBE, gap=0.1, s=1.6, order=3, base=SPH2, Lu=15.0, Ld=45.0, wake=dict(x_near=8.0, x_mid=30.0), Re=RE_D)
SLIM = dict(kind="shear", Lu=10.0, Ld=30.0, Ly=12.0, H=12.0, wake=dict(x_near=6.0, x_mid=16.0, far=False))
for g in (0.1, 0.03):
    arms[f"SS2-g{g}"] = dict(SLIM, gap=g, s=1.6, full=True, Re=RE_D, stability=dict(at=[260, 455, 650], dt=0.1, T=150.0, seed=1))
    arms[f"SL-g{g}"] = dict(SLIM, gap=g, s=1.6, order=2, Re=RE_D)
    arms[f"SLP3-g{g}"] = dict(SLIM, gap=g, s=1.6, order=3, Re=RE_D)
arms["SLP3-sph-g0.1"] = dict(SLIM, gap=0.1, s=1.6, order=3, base=SPH2, Re=RE_D)
arms["SL-g0.1-s1.26"] = dict(SLIM, gap=0.1, s=1.26, order=2, Re=RE_D)
for n in arms:
    assert len(n) <= 40, n
out = sys.argv[1]
spec = json.load(open(f"{out}/arms_spec.json"))
spec.update(arms)
json.dump(spec, open(f"{out}/arms_spec.json", "w"), indent=1)
open(f"{out}/arms3.txt", "w").write("\n".join(arms) + "\n")
print(len(arms), "arms")
