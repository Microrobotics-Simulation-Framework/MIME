"""Detection rates of the cylinder plants of the contact-loops pull request, under the rule with the rattled searches and
the stationarity judgement, beside the rule the pull request's base holds (the poses alone)."""
import json
import os
import sys

sys.path[:0] = ["tests", "tests/oracles"]
import numpy as np
import test_contact_loops as T

arm, draws = os.environ["PROBE_ARM"], int(os.environ.get("PROBE_DRAWS", "6000" if os.environ["PROBE_ARM"].startswith("hunt") else "3000"))
rng = np.random.default_rng(int(arm.rsplit("_", 1)[1]))
PLANTS = ("short", "stale", "early")
bound = T.NEWTON_ULPS
out = {"arm": arm, "draws": draws, "new": {p: 0 for p in (*PLANTS, "clean")}, "old": {p: 0 for p in PLANTS},
       "lost": {p: 0 for p in PLANTS}, "unresolved_groups": 0, "groups": 0}
kinds = ["cylinder"] if arm.startswith("cyl") else sorted(T.SUPPORTED)
HUNT = arm.startswith("hunt")
planted = {} if arm.startswith("hunt") else {(k, p): T.planted_search(k, "float32", p) for k in kinds for p in PLANTS}
for i in range(draws):
    kind = kinds[rng.integers(len(kinds))]
    spin, about, towards = rng.uniform(0, 2 * np.pi, 3)
    tilt, share = rng.uniform(0, 1.5), rng.uniform(-0.5, 1.0)
    pose = T.placed(kind, spin, tilt, about, towards, share * T.MARGIN)
    clean = T.searched_against_0_4_0(kind, "float32", pose)
    out["unresolved_groups"] += int(np.sum(~clean.resolved))
    out["groups"] += len(clean.resolved)
    out["new"]["clean"] += int(np.max(clean.missed) > bound)
    if np.max(clean.missed) > bound:
        out.setdefault("clean_flagged", []).append({"i": i, "kind": kind, "draw": [spin, tilt, about, towards, share],
            "unstationary": clean.unstationary.tolist(), "resolved": clean.resolved.tolist(), "spread": clean.spread.tolist(),
            "alone": clean.alone.tolist(), "missed": [float(v) for v in clean.missed]})
    for p in () if HUNT else PLANTS:
        new = np.max(T.searched_against_0_4_0(kind, "float32", pose, planted=planted[kind, p]).missed) > bound
        base = T.searched_against_0_4_0(kind, "float32", pose, planted=planted[kind, p], rattles=0)
        old = np.max(np.where(base.spread <= bound, base.alone, 0.0)) > bound
        out["new"][p] += int(new)
        out["old"][p] += int(old)
        out["lost"][p] += int(old and not new)
        if old and not new:
            out.setdefault("lost_draws", []).append({"i": i, "plant": p, "kind": kind, "draw": [spin, tilt, about, towards, share]})
    if i % 100 == 0:
        print(i, out, flush=True)
os.makedirs("probe/out", exist_ok=True)
json.dump(out, open(f"probe/out/rates_{arm}.json", "w"))
print("RESULT", json.dumps(out))
