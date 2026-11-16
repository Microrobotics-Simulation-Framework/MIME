"""One arm of the spike (PROBE_ARM): every problem, speed, phase and dtype; the controls; the JSON to probe/out/."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402


def main() -> None:
    arm = h.ARMS[sys.argv[1] if len(sys.argv) > 1 else os.environ["PROBE_ARM"]]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    out = Path(os.environ.get("PROBE_OUT", "probe/out"))
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"arm": arm.name, "ties": arm.ties, "sticky": arm.sticky, "tolerance": h.TOLERANCE, "problems": {}}
    results["controls"] = h.planted()
    started = time.time()
    for spec in h.specs():
        if only and spec.name not in only:
            continue
        block = results["problems"].setdefault(spec.name, {})
        for dtype in arm.dtypes:
            for speed in ("fast", "slow"):
                for phase in ("bottom", "top"):
                    u = h.path(spec, speed, phase)
                    # the target as the caller gives it: float64 value, rounded once to the dtype
                    targets = (spec.w0[None, :] * np.exp(u)[:, None]).astype(dtype)
                    ran = h.run_law(spec, targets, arm, dtype)
                    entry = h.count(ran["commands"], targets, spec.scales, u, ran["face"])
                    entry["traced"] = ran["traced"]
                    entry["seconds"] = round(time.time() - started, 1)
                    block.setdefault(dtype, {}).setdefault(speed, {})[phase] = entry
                    print(arm.name, spec.name, dtype, speed, phase, {k: v for k, v in entry.items() if k.startswith("switches")},
                          entry["face_changes"], entry["traced"], flush=True)
    (out / f"{arm.name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
