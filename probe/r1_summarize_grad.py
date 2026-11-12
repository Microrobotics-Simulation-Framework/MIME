"""Summarize the gradient probe's artifacts: the gates' judged checks, the controls' detection rates, the clean runs'
checks, the cache path's share of each leaf's gradient (cache-stop against clean at the same seed)."""
import glob
import json
import os
import sys
from collections import defaultdict

root = sys.argv[1]
print("== gates")
for arm in ("gate-160", "gate-cont", "gate-320"):
    for path in glob.glob(f"{root}/*{arm}*/**/judged.json", recursive=True):
        for r in json.load(open(path)):
            fails = [k for k, c in r["checks"].items() if not c["pass"]]
            print(arm, r["dtype"], "all_pass", r["all_pass"], "r_J %.3g" % r["r_J"], "FAILED:", fails)
            for k, c in r["checks"].items():
                keep = {kk: (float("%.3g" % v) if isinstance(v, float) else v) for kk, v in c.items()
                        if kk in ("in_eps_S", "diff_over_bound", "relative_bound", "bitwise", "worst", "bound", "ad", "cd")}
                print("   ", "ok " if c["pass"] else "BAD", k, keep)
print("== detection rates")
total = defaultdict(lambda: [0, 0, 0, 0, []])
for path in sorted(glob.glob(f"{root}/**/detection-*.json", recursive=True)):
    d = json.load(open(path))
    plant = os.path.basename(path).split("-")[1]
    p = d["planted"]
    c = d.get("clean", {})
    t = total[plant]
    t[0] += p.get("caught", 0) if isinstance(p.get("caught"), int) else int(p["caught_of_runs"].split("/")[0])
    t[1] += int(p["caught_of_runs"].split("/")[1])
    t[2] += c.get("flagged", 0) if isinstance(c.get("flagged"), int) else len(c.get("flagged_seeds", []))
    t[3] += c.get("runs", 0)
    t[4] += p.get("missed_seeds", []) + [("inc", s) for s in p.get("inconclusive_seeds", [])]
for plant, (caught, runs, flagged, clean, missed) in total.items():
    print(plant, f"caught {caught}/{runs}", f"clean flagged {flagged}/{clean}", "missed/inconclusive", missed)
print("== checks over the draws (clean and planted)")
by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
worst = defaultdict(float)
for path in glob.glob(f"{root}/**/check/*/judged.json", recursive=True):
    plant = os.path.basename(os.path.dirname(path)).rsplit("-", 1)[0]
    for r in json.load(open(path)):
        for k, c in r["checks"].items():
            by[plant][f"{r['dtype']} {k}"][0] += 1
            by[plant][f"{r['dtype']} {k}"][1] += 0 if c["pass"] else 1
            if plant == "clean" and c.get("diff_over_bound") is not None:
                worst[f"{r['dtype']} {k}"] = max(worst[f"{r['dtype']} {k}"], c["diff_over_bound"])
for plant, checks in by.items():
    print(plant, {k: f"{f}/{n} failed" for k, (n, f) in sorted(checks.items()) if f})
print("clean: the largest |AD - CD| / bound per check:", {k: round(v, 3) for k, v in sorted(worst.items())})
print("== the cache path's share of each leaf's gradient (1 - g(cache-stop) / g, float64, same seed)")
for path in sorted(glob.glob(f"{root}/**/check/cache-stop-*/float64.json", recursive=True)):
    seed = path.split("cache-stop-")[1].split("/")[0]
    clean = path.replace("cache-stop-", "clean-")
    if not os.path.exists(clean):
        continue
    a, b = json.load(open(path)), json.load(open(clean))
    print(seed, {n: round(1 - a["g_vjp"][i] / b["g_vjp"][i], 3) for i, n in enumerate(b["names"])}, "z0_um",
          round(b["point"]["z0_um"], 1))
