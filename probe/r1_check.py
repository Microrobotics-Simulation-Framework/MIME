"""Camera spike R1's check for ``detection_rate.py``: the gradient gates (``r1_judge.py``) on one drawn operating point
(seed ``DETECTION_SEED``), float64 and float32 each in its own process, with the plant ``CAMERA_R1_PLANT`` if set.

Exit 1: a check failed (caught). Exit 0: every check passed. Exit 2: an error (inconclusive). The runs' JSON is kept under
``probe/out/check/<plant or clean>-<seed>/``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    seed = int(os.environ.get("DETECTION_SEED", "0") or 0)
    plant = os.environ.get("CAMERA_R1_PLANT", "") or "clean"
    out = HERE / "out" / "check" / f"{plant}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for dtype in ("float64", "float32"):
        path = out / f"{dtype}.json"
        done = subprocess.run([sys.executable, str(HERE / "r1_grad.py"), "--dtype", dtype, "--out", str(path),
                               "--draw", str(seed), "--per-step"], check=False, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-2000:], done.stderr[-4000:])
            return 2
        paths.append(path)
    sys.path.insert(0, str(HERE))
    import r1_judge

    loaded = [json.loads(p.read_text()) for p in paths]
    f64, f32 = loaded
    results = [r1_judge.judge(f64, f64, f32), r1_judge.judge(f32, f64, f32)]
    (out / "judged.json").write_text(json.dumps(results, indent=1))
    failed = [f"{r['dtype']} {name}" for r in results for name, c in r["checks"].items() if not c["pass"]]
    print(f"r1_check seed {seed} plant {plant}: " + ("FLAGGED " + "; ".join(failed) if failed else "all pass"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
