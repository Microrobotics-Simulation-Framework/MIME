"""Writes arms_spec.json and arms.txt for the w7ref probe (launch 1)."""
import json
import sys

SL = [1.6, 1.26, 1.0, 0.8, 0.63]
RE_D = [0, 130, 260, 455, 650]
arms = {}
for g in (0.1, 0.03, 0.01):
    for s in SL:
        arms[f"T-g{g}-s{s}"] = dict(kind="tube", gap=g, s=s, Lu=10.0, Ld=30.0, Re=RE_D)
        if s >= 0.8:  # the plane-wall box at s 0.63 is about 0.9M unknowns: over a runner's 16 GB
            arms[f"S-g{g}-s{s}"] = dict(kind="shear", gap=g, s=s, Lu=12.0, Ld=36.0, Ly=16.0, H=16.0, Re=RE_D)
# domain checks (s = 1.0, gap 0.1)
arms["TD-long-g0.1"] = dict(kind="tube", gap=0.1, s=1.0, Lu=15.0, Ld=45.0, wake=dict(x_near=8.0, x_mid=30.0), Re=RE_D)
arms["TD-short-g0.1"] = dict(kind="tube", gap=0.1, s=1.0, Lu=6.0, Ld=20.0, Re=RE_D)
arms["SD-big-g0.1"] = dict(kind="shear", gap=0.1, s=1.0, Lu=18.0, Ld=54.0, Ly=24.0, H=24.0, Re=RE_D)
arms["SD-small-g0.1"] = dict(kind="shear", gap=0.1, s=1.0, Lu=8.0, Ld=24.0, Ly=10.0, H=10.0, Re=RE_D)
# planted viscosity (x1.05 at fixed U_m, a)
arms["TN-g0.1-s1.0"] = dict(kind="tube", gap=0.1, s=1.0, Lu=10.0, Ld=30.0, nu_factor=1.05, Re=[0, 130, 260, 650])
# p-check: order 3 on the same meshes
arms["TP3-g0.1-s1.26"] = dict(kind="tube", gap=0.1, s=1.26, order=3, Lu=10.0, Ld=30.0, Re=RE_D)
arms["TP3-g0.03-s1.26"] = dict(kind="tube", gap=0.03, s=1.26, order=3, Lu=10.0, Ld=30.0, Re=RE_D)
arms["TP3-g0.1-s1.6"] = dict(kind="tube", gap=0.1, s=1.6, order=3, Lu=10.0, Ld=30.0, Re=RE_D)
# stability: full domain, linearised Crank-Nicolson march
for g in (0.1, 0.03):
    arms[f"TS-g{g}-s1.26"] = dict(kind="tube", gap=g, s=1.26, full=True, Lu=10.0, Ld=30.0, Re=RE_D,
                                  stability=dict(at=[260, 650], dt=0.1, T=150.0, seed=1))
    arms[f"SS-g{g}-s1.26"] = dict(kind="shear", gap=g, s=1.26, full=True, Lu=12.0, Ld=36.0, Ly=16.0, H=16.0, Re=RE_D,
                                  stability=dict(at=[260, 455, 650], dt=0.1, T=150.0, seed=1))
# Stokes resistances on the plane-wall mesh design, large box, against the exact bispherical values
for g in (0.1, 0.03, 0.01):
    for s in (1.6, 1.26, 1.0, 0.8):
        arms[f"CS-g{g}-s{s}"] = dict(kind="res", gap=g, s=s, Lu=40.0, Ld=40.0, Ly=40.0, H=40.0, hfar=8.0)
# validation: Ekanayake et al. 2020 Table 2 (l/a 1.2, zero slip, non-rotating), box L* 50 and 100
for s in (1.6, 1.26, 1.0, 0.8):
    arms[f"E-s{s}"] = dict(kind="eka", gap=0.2, s=s, Lu=50.0, Ld=50.0, Ly=50.0, H=51.2, hfar=8.0, Re=[0, 0.001, 0.1])
arms["E-L100-s1.0"] = dict(kind="eka", gap=0.2, s=1.0, Lu=100.0, Ld=100.0, Ly=100.0, H=101.2, hfar=12.0, Re=[0, 0.001, 0.1])
# control: unbounded sphere (uniform flow in a wide cylinder)
for s in (1.6, 1.26, 1.0, 0.8):
    arms[f"F-s{s}"] = dict(kind="free", gap=0.0, s=s, Rf=25.0, Lu=15.0, Ld=40.0, hfar=6.0, Re=[0, 50, 100, 200])
arms["F-R40-s1.0"] = dict(kind="free", gap=0.0, s=1.0, Rf=40.0, Lu=20.0, Ld=50.0, hfar=8.0, Re=[0, 50, 100, 200])

for n in arms:
    assert len(n) <= 40, n
out = sys.argv[1]
json.dump(arms, open(f"{out}/arms_spec.json", "w"), indent=1)
open(f"{out}/arms.txt", "w").write("\n".join(arms) + "\n")
print(len(arms), "arms")
