"""Read the arms' JSON (a folder of <arm>.json files, or probe artifact folders below it) and print the counts per arm and
dtype, with float64's TIE 0 as the baseline of legitimate switches."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(root: Path) -> dict[str, dict]:
    found = {}
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text())
        if "arm" in data and "problems" in data:
            found[data["arm"]] = data
    return found


def main() -> None:
    arms = load(Path(sys.argv[1]))
    order = [a for a in ("tie0", "tie16", "tie64", "tie128", "sticky64", "w64x") if a in arms]
    for problem in ("octomag", "cardiomag", "octo_field"):
        for thr in ("1A", "0.03A"):
            print(f"\n== {problem}, switches over {thr} (sum over phases bottom+top; fast / slow)")
            for arm in order:
                row = []
                for dtype in ("float32", "float64"):
                    cell = arms[arm]["problems"].get(problem, {}).get(dtype)
                    if not cell:
                        row.append(f"{dtype}: -")
                        continue
                    fast = [cell["fast"][p][f"switches_{thr}"] for p in ("bottom", "top")]
                    slow = [cell["slow"][p][f"switches_{thr}"] for p in ("bottom", "top")]
                    row.append(f"{dtype}: fast {fast[0]}+{fast[1]} slow {slow[0]}+{slow[1]}")
                print(f"  {arm:9s} " + " | ".join(row))
    print("\ncontrols per arm:", {a: arms[a]["controls"] for a in order})
    print("traced:", {a: sorted({str(c["traced"]) for p in arms[a]["problems"].values() for d in p.values() for s in d.values() for c in s.values()}) for a in order})


if __name__ == "__main__":
    main()
