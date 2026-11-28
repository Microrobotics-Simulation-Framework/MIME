"""Compare G0 / G0b / Bunb arms with the semi-analytic anchors. Usage: python -I anal_g0.py file.jsonl [...]"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
import series as SE

K = -mp.sqrt(2) / 3   # the two-sphere force constant fixed by the plane limit (series.py output)
cache = {}


def exact_g0(gap):
    if gap not in cache:
        zz = float(SE.brenner_normal(str(gap))[0])
        ftt, ct, cr = [float(v) for v in SE.tangential_asym(str(gap))]
        rz = float(SE.jeffery_normal_rotation(str(gap))[0])
        cache[gap] = dict(zz=zz, xx=ftt, x_ry=ct, ryry=cr, rzrz=rz)
    return cache[gap]


rows = []
for fn in sys.argv[1:]:
    for line in open(fn):
        r = json.loads(line)
        if "S" not in r:
            print("ERR", r.get("spec", {}).get("tag"), r.get("error")); continue
        S = np.array(r["S"])
        sp = r["spec"]
        g = sp.get("plant_of", sp["gap"])     # K1 plants are compared with the exact value at the unplanted gap
        ndof = sum(c["free"] for c in r["classes"])
        rss = max(c["rss_gib"] for c in r["classes"])
        if sp["geom"] == "G0" and r["s"] == 0:
            e = exact_g0(g)
            num = dict(zz=S[2, 2] / (6 * np.pi), xx=S[0, 0] / (6 * np.pi), x_ry=-S[4, 0] / (8 * np.pi), ryry=S[4, 4] / (8 * np.pi),
                       rzrz=S[5, 5] / (8 * np.pi))
            rel = {k: num[k] / e[k] - 1 for k in e}
            sym = abs(S[4, 0] - S[0, 4]) / abs(S[4, 0])
            print(f"{sp.get('tag')} G0 gap {g} P{r['order']} c{sp['cfac']} hb{sp['hbody']} Lc{sp.get('Lc')} free {ndof} rss {rss:.2f}  " +
                  "  ".join(f"{k} {num[k]:.7g} ({rel[k]:+.2e})" for k in e) + f"  asym {sym:.1e}")
        elif sp["geom"] == "G0b" and r["s"] == 0:
            ex = float(SE.twosphere_normal(str(g), str(sp.get("b", 4.0)), K)[0])
            v = S[2, 2] / (6 * np.pi)
            print(f"{sp.get('tag')} G0b gap {g} P{r['order']} c{sp['cfac']} free {ndof} rss {rss:.2f} zz {v:.8g} exact {ex:.8g} ({v / ex - 1:+.2e})")
        elif sp["geom"] == "Bunb":
            F, T = [float(v) for v in SE.brinkman_unbounded(r["s"])]
            f, t = S[0, 0] / (6 * np.pi), S[3, 3] / (8 * np.pi)
            print(f"{sp.get('tag')} Brinkman s {r['s']} P{r['order']} hb{sp['hbody']} free {ndof} rss {rss:.2f} F {f:.8g} ({f / F - 1:+.2e}) T {t:.8g} ({t / T - 1:+.2e})")
        else:
            print(sp.get("tag"), sp["geom"], g, r["order"], r["s"], ndof, np.round(np.diag(S), 6))
