"""The step share's summary: per case and dtype, the steady forward seconds per base step with the law and without
(the same rows made, no allocation), the law's own cost (the difference) and its share of the law's step."""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for path in sorted(out.glob("share_*_*_float*.json")):
    case, allocating, dtype = path.stem.removeprefix("share_").rsplit("_", 2)
    data = json.loads(path.read_text())
    rows[(case, allocating, dtype)] = {"s_per_step": data.get("seconds_per_step_steady_min"),
                                       "cpu": (data.get("environment") or {}).get("cpu_model"),
                                       "loaded": (data.get("environment") or {}).get("timing_on_loaded_machine")}
summary = []
for (case, allocating, dtype), row in sorted(rows.items()):
    if allocating != "true":
        continue
    base = rows.get((case, "false", dtype), {})
    entry = {"case": case, "dtype": dtype, "step_us": None, "baseline_us": None, **{k: row[k] for k in ("cpu", "loaded")}}
    if row["s_per_step"] is not None and base.get("s_per_step") is not None:
        entry["step_us"], entry["baseline_us"] = row["s_per_step"] * 1e6, base["s_per_step"] * 1e6
        entry["allocation_us"] = entry["step_us"] - entry["baseline_us"]
        entry["share_of_step"] = entry["allocation_us"] / entry["step_us"]
    summary.append(entry)
print(json.dumps(summary, indent=1))
