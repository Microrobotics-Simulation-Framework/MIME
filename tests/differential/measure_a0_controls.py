"""Measure how often each planted fault is caught by the A0 comparison, over sizes from below its resolution to ten
times it:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_controls.py

For each fault in ``a0_plants.py`` it draws 20 sizes between 1.5 and 10 resolutions, with random signs (a fixed seed),
and 6 below the resolution (0.1 to 0.6 of it; the controller's latency, planted only in whole steps, has none), runs
0.3's A0 in float32 at 3 Hz with the fault planted, and records whether the comparison with the declared prediction
failed and whether the quantity the fault moves most missed. Writes ``references/a0_controls_measured.json``. A fault
above the resolution must be caught every time, and one below it should not be: a control that fails at one run in
eight (the contact oracle's, once) is not a control. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_controls_measured.json"
SEED = 20261030
ABOVE, BELOW = 20, 6


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    from a0_compare import compare, measure, missed
    from a0_plants import PLANTS

    rng = np.random.default_rng(SEED)
    record: dict[str, object] = {"seed": SEED, "key": "3hz", "float32": True, "plants": {}}
    for plant in PLANTS:
        draws: list[dict[str, object]] = []
        sizes = [float(x) for x in np.exp(rng.uniform(np.log(1.5), np.log(10.0), ABOVE))]
        if plant.part == "the controller's latency":
            sizes = [float(x) for x in rng.integers(1, 11, ABOVE)]
        else:
            sizes += [float(x) for x in rng.uniform(0.1, 0.6, BELOW)]
        for size in sizes:
            sign = float(rng.choice([-1.0, 1.0]))
            rows = compare(measure("3hz", **plant.apply(size, sign)), "3hz")
            found = missed(rows)
            draws.append({"size_in_resolutions": size, "sign": sign, "caught": bool(found),
                          "quantity_missed": plant.quantity in found, "missed": found,
                          "miss_of_the_quantity": next(r.miss for r in rows if r.quantity == plant.quantity)})
            print(f"{plant.part:28s} size {size:6.2f} sign {sign:+.0f} caught {bool(found)} "
                  f"({plant.quantity} {'missed' if plant.quantity in found else 'held'})", flush=True)
        above = [d for d in draws if d["size_in_resolutions"] >= 1.5]
        below = [d for d in draws if d["size_in_resolutions"] < 1.5]
        record["plants"][plant.part] = {  # type: ignore[index]
            "fault": plant.name, "quantity": plant.quantity, "resolution": plant.resolution,
            "caught_above_the_resolution": f"{sum(bool(d['caught']) for d in above)}/{len(above)}",
            "quantity_missed_above": f"{sum(bool(d['quantity_missed']) for d in above)}/{len(above)}",
            "caught_below_the_resolution": f"{sum(bool(d['caught']) for d in below)}/{len(below)}",
            "draws": draws}
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
