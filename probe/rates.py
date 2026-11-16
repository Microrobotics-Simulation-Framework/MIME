"""Replays of two draws on which the clean looped arm was flagged on a runner (round 3), with the reason."""
import json
import os
import sys

sys.path[:0] = ["tests", "tests/oracles"]
import jax
import jax.numpy as jnp
import numpy as np
import test_contact_loops as T

arm = os.environ["PROBE_ARM"]
DRAWS = [(4.87705747991847, 0.35266508542049807, 1.6715090783522921, 2.7785117170337896, 0.971075033193801),
         (1.8290945297494494, 0.5406940802194558, 3.7627786792285014, 5.9225141303086035, 0.7581163517014757)]
out = {"arm": arm, "cpu": T.cpu_model(), "replays": []}
for d in DRAWS:
    pose = T.placed("screw", *d[:4], d[4] * T.MARGIN)
    f = T.searched_against_0_4_0("screw", "float32", pose)
    g = int(np.argmin(f.resolved)) if not f.resolved.all() else -1
    row = {"draw": d, "resolved": f.resolved.tolist(), "unstationary": f.unstationary.tolist(),
           "flag_wrong": f.flag_wrong.tolist(), "off_bound": f.off_bound.tolist(), "alone": f.alone.tolist(),
           "spread": f.spread.tolist()}
    if g >= 0:
        with jax.enable_x64(False):
            p, size, seeds, unrolled, rattled, conditioned, looped = T.searches("screw", "float32")
            q = T.traced(p, "float32")
            at = jnp.asarray(pose, dtype="float32")
            kept = seeds(q, at)
            unit = float(np.finfo("float32").eps) * size
            xl, sl, wl = (np.asarray(v, np.float64) for v in looped(q, at, kept))
            xu, su, wu = (np.asarray(v, np.float64) for v in unrolled(q, at, kept))
            xr, sr, wr = (np.asarray(v, np.float64) for v in rattled(q, at, kept, jnp.asarray(T.kicks_for(8, "float32"))))
            res = lambda x, s: np.linalg.norm(x - s, axis=-1) / unit
            row["group"] = g
            row["looped"] = {"res": float(res(xl, sl)[g]), "w": float(wl[g] / unit), "x": xl[g].tolist()}
            row["unrolled"] = {"res": float(res(xu, su)[g]), "w": float(wu[g] / unit), "x": xu[g].tolist()}
            row["rattled_res"] = [float(v) for v in res(xr, sr)[:, g]]
            row["rattled_w"] = [float(v) for v in wr[:, g] / unit]
            row["cond_looped"] = float(np.asarray(conditioned(q, at, jnp.asarray(xl, dtype="float32")))[g])
    out["replays"].append(row)
os.makedirs("probe/out", exist_ok=True)
json.dump(out, open(f"probe/out/replay_{arm}.json", "w"))
print(json.dumps(out))
