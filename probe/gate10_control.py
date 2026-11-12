"""Gate 10 at its own scale (the fix round, the review's M1): the BE patch at a stage-1 corner (the impulse along the
wall's normal, 0.15 a) in F5's structure, its columns cold (the harness as fixed), its force history against the fine
monolithic BE reference's (run 43151596075, probe/reference_forces.json: the reference holds no patch, so the round
leaves it unchanged). Exit 1 (caught) when the largest |F / F_ref - 1| over the steps exceeds GATE10_LINE (5 %, the
predicted band's edge). PATCH_PLANT=history_dropped drops the base column's history, the shift kept. The departures
and the history are written to probe/out either way. Arguments: <corner>-g<gap>."""
import json, math, os, sys
from pathlib import Path
sys.path[:0] = ["tests", "tests/patch"]
from time_cases import CORNERS, TimeCase, run

GATE10_LINE = 0.05
key = sys.argv[1]
corner, gap = key.rsplit("-g", 1)
viscosity, dt = CORNERS[corner]
probe = TimeCase(viscosity=viscosity, dt=dt, gap=float(gap), direction=2)
case = TimeCase(viscosity=viscosity, dt=dt, gap=float(gap), direction=2, steps=max(20, math.ceil(1.5 * probe.lam ** 2)))
reference = json.load(open("probe/reference_forces.json"))[key]
assert reference["steps"] == case.steps, (reference["steps"], case.steps)
found = run(case, "be_patch")
errors = [a / b - 1.0 for a, b in zip(found["force"], reference["force"])]
warm = reference.get("be_patch_warm_columns")
worst = max(abs(e) for e in errors)
held = os.environ.get("PATCH_PLANT") or "clean"
row = dict(key=key, plant=held, lam=case.lam, steps=case.steps, worst=worst, final=errors[-1], first=errors[0],
           errors=errors, force=found["force"], reference=reference["force"],
           cold_vs_warm_columns=None if warm is None else max(abs(a / b - 1.0) for a, b in zip(found["force"], warm)))
Path("probe/out").mkdir(parents=True, exist_ok=True)
json.dump(row, open(f"probe/out/gate10_{key}_{held}_{os.environ.get('DETECTION_SEED', '0')}.json", "w"), indent=1)
print(key, held, "worst", worst, "final", errors[-1], "cold vs warm columns", row["cold_vs_warm_columns"])
sys.exit(1 if worst > GATE10_LINE else 0)
