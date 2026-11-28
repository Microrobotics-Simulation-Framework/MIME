"""Reduce linear arms to reference values with errors. Usage: python -I anal_lin.py out.json file.jsonl [...]
Key: (geom, pose, body, gap, s, plant_of). Level = (cfac, hbody). For each quantity: best = finest level, top order; e_p from the
p-ladder on that level (Aitken if the last two changes shrink geometrically, else the last change); e_h = change of the top
order between the two finest levels; err = max(e_p, e_h) [estimate]. Quantities: the 21 entries of S (upper triangle) and F0."""
import sys, json
import numpy as np
from collections import defaultdict

QN = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def load(files):
    recs = []
    for fn in files:
        for line in open(fn):
            r = json.loads(line)
            if "S" in r:
                recs.append(r)
    return recs


def key_of(r):
    sp = r["spec"]
    return (sp["geom"], sp.get("pose", ""), sp.get("body", "SPH"), float(sp.get("plant_of", sp["gap"])), r["s"],
            "plant" if "plant_of" in sp else "", sp.get("Lc", ""), sp.get("b", ""))


def p_err(vals):
    """vals: list of (order, value) ascending. Returns (estimate of limit, e_p)."""
    vals = sorted(vals)
    v = [b for _, b in vals]
    if len(v) == 1:
        return v[0], np.nan
    if len(v) == 2:
        return v[1], abs(v[1] - v[0])
    d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
    if d1 != 0 and 0 < d2 / d1 < 0.8:
        rho = d2 / d1
        lim = v[-1] + d2 * rho / (1 - rho)
        return lim, abs(lim - v[-1])
    return v[-1], abs(d2)


def reduce(recs):
    groups = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        sp = r["spec"]
        lvl = (sp["cfac"], sp["hbody"])
        q = {}
        S = np.array(r["S"])
        for i in range(6):
            for j in range(i, 6):
                if abs(S[i, j]) > 1e-12 * abs(S[i, i]) + 1e-300:
                    q[f"S{i}{j}"] = S[i, j]
        for i, n in enumerate(QN):
            if "F0" in r and abs(r["F0"][i]) > 1e-9 * max(1.0, max(abs(t) for t in r["F0"])):
                q["F0_" + n] = r["F0"][i]
        q["_free"] = sum(c["free"] for c in r["classes"])
        q["_rss"] = max(c["rss_gib"] for c in r["classes"])
        groups[key_of(r)][lvl][r["order"]] = q
    out = {}
    for k, lv in groups.items():
        levels = sorted(lv, key=lambda t: (-t[0], -t[1]))     # coarse to fine (cfac decreasing)
        fine = levels[-1]
        top = max(lv[fine])
        res = {"levels": [list(l) for l in levels], "top_order": top, "free_dofs_top": lv[fine][top]["_free"],
               "rss_top": lv[fine][top]["_rss"], "q": {}}
        names = [n for n in lv[fine][top] if not n.startswith("_")]
        for n in names:
            best = lv[fine][top][n]
            lim, ep = p_err([(o, lv[fine][o][n]) for o in lv[fine] if n in lv[fine][o]])
            eh = np.nan
            if len(levels) > 1:
                prev = levels[-2]
                if top in lv[prev] and n in lv[prev][top]:
                    eh = abs(best - lv[prev][top][n])
            err = np.nanmax([ep, eh]) if not (np.isnan(ep) and np.isnan(eh)) else np.nan
            ladder = {f"c{l[0]}-h{l[1]}": {str(o): lv[l][o].get(n) for o in sorted(lv[l])} for l in levels}
            res["q"][n] = dict(best=best, p_limit=lim, e_p=ep, e_h=eh, err=err, rel=err / abs(best) if best else np.nan, ladder=ladder)
        out["|".join(str(t) for t in k)] = res
    return out


if __name__ == "__main__":
    out = reduce(load(sys.argv[2:]))
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    for k, r in out.items():
        print(k, "levels", r["levels"], "P", r["top_order"], "dofs", r["free_dofs_top"], "rss %.2f" % r["rss_top"])
        for n, q in r["q"].items():
            print(f"   {n:8s} {q['best']:+.8g}  e_p {q['e_p']:.2e} e_h {q['e_h']:.2e}  rel {q['rel']:.1e}")
