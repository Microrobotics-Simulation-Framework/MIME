"""Summarize the cost probe's node_measure outputs, the equality arm and the batching arms."""
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"{root}/probe-camera-r1-cost-*/*.json")):
    name = os.path.basename(path)[:-5]
    if name.startswith(("mem_", "bt-")) or "eq-160" in path:
        continue
    d = json.load(open(path))
    if "peak_memory" not in d:
        continue
    pm, rev = d["peak_memory"], d.get("reverse") or {}
    rows.append((name, d["environment"]["cpu_model"], d["seconds_per_step_steady_min"] * 1e3,
                 (rev.get("seconds_per_step_steady_min") or float("nan")) * 1e3,
                 (pm["forward_peak_mib"] or 0) / 1024, (pm["reverse_peak_mib"] or 0) / 1024, pm["peak_mib"] / 1024,
                 d["trace_check"]["passed"], d["timing_on_loaded_machine"], rev.get("gradient_finite")))
print(f"{'arm':30s} {'cpu':28s} {'fwd ms/st':>9s} {'rev ms/st':>9s} {'fwdpk':>6s} {'revpk':>6s} {'peak':>6s} trace loaded finite")
for r in rows:
    print(f"{r[0]:30s} {r[1][:28]:28s} {r[2]:9.2f} {r[3]:9.2f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f} {r[7]} {r[8]} {r[9]}")
print("== memory checks (arm peak against the * bound)")
for path in sorted(glob.glob(f"{root}/probe-camera-r1-cost-*/mem_check.json")):
    d = json.load(open(path))
    print(os.path.basename(os.path.dirname(path)), "peak %.2f bound %.2f raw %.2f" % (d["peak_gib"], d["bound_gib"],
                                                                                    d["raw_estimate_gib"]),
          "EXCEEDS" if d["exceeds_bound"] else "")
print("== batching")
for path in sorted(glob.glob(f"{root}/probe-camera-r1-cost-bt-*/bt-*.json")):
    d = json.load(open(path))
    print(os.path.basename(path)[:-5], "B", d["batch"], d["kind"], d["mode"], "P", d["period"],
          "ms/step %.2f ms/member-step %.2f peak %.2f GiB xla temp %.2f" % (d["seconds_per_step_min"] * 1e3,
                                                                           d["seconds_per_member_step_min"] * 1e3,
                                                                           d["peak_gib"], d["xla_temp_gib"] or 0),
          d["cpu"][:28])
