"""Measure the second declaration's controls on 0.3: each plant of ``a0_dynamic_b_plants.py`` at its declared size and
at a second size, at 3 Hz and 10 Hz, against every gate of the second declaration:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_dynamic_b_controls.py

For each plant it records every gate's miss in its tolerances at both sizes and unplanted, the response of the gate
the plant must fail (its miss's move per unit of the plant, from the second size), that gate's resolution (its
tolerance over the response, in units of the declared plant) and the plant's size in those resolutions. Writes
``references/a0_dynamic_b_controls_measured.json``. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_dynamic_b_controls_measured.json"


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    from a0_dynamic_b_plants import GATE, KICK_DURATION, SECOND_SIZE, drifted_gates, planted
    from a0_dynamic_compare import FREQUENCY, SECOND, measure, run, second, transient

    def misses(gates: Any) -> dict[str, Any]:
        """Each gate's miss in its tolerances; for a gate held exactly (tolerance zero), the miss itself."""
        return {gate.quantity: gate.miss / gate.tolerance if gate.tolerance else {"exact_miss": gate.miss}
                for gate in gates}

    def kick_miss(found: Any, key: str) -> float:
        kick = transient(found)
        return kick[GATE["J"]] / SECOND["transient"]["predicted"][key][GATE["J"]] - 1

    record: dict[str, Any] = {"by_frequency": {}, "gates": GATE, "second_size": SECOND_SIZE}
    for key in FREQUENCY:
        print(f"{key}: unplanted", flush=True)
        measured, found = measure(key)
        base = second(key, measured, found)
        base_miss = {gate.quantity: gate.miss for gate in base}
        short = run(FREQUENCY[key], duration=KICK_DURATION)
        kick_base = kick_miss(short, key)
        unplanted = {"misses_in_tolerances": misses(base),
                     "kick_over_the_first_half_second_less_the_runs": {
                         name: transient(short)[name] - transient(found)[name] for name in transient(found)}}
        by_plant: dict[str, Any] = {}
        for plant, factor in SECOND_SIZE.items():
            gate = GATE[plant]
            sizes: dict[str, Any] = {}
            moved = {}
            for size in (1.0, factor):
                print(f"{key}: {plant} x {size:g}", flush=True)
                if plant == "drift":
                    gates = drifted_gates(key, measured, found, size)
                elif plant == "J":
                    _, kicked = planted(key, "J", size)
                    tolerance = SECOND["transient"]["tolerance"]
                    miss = kick_miss(kicked, key)
                    sizes[f"{size:g}"] = {"gate_miss_in_tolerances": miss / tolerance}
                    moved[size] = miss - kick_base
                    continue
                else:
                    plant_measured, plant_found = planted(key, plant, size)
                    gates = second(key, plant_measured, plant_found)
                chosen = next(g for g in gates if g.quantity == gate)
                sizes[f"{size:g}"] = {"misses_in_tolerances": misses(gates),
                                      "gate_miss_in_tolerances": chosen.miss / chosen.tolerance if chosen.tolerance
                                      else None, "gate_miss": chosen.miss}
                moved[size] = chosen.miss - base_miss[gate]
            tolerance = SECOND["transient"]["tolerance"] if plant == "J" else next(
                g.tolerance for g in base if g.quantity == gate)
            response = moved[factor] / factor
            entry = {"gate": gate, "sizes": sizes, "response_per_declared_plant": response,
                     "tolerance": tolerance}
            if tolerance:
                entry["resolution_in_declared_plants"] = tolerance / abs(response)
                entry["plant_in_resolutions"] = abs(response) / tolerance
            else:  # held exactly: any plant the gate responds to is above its resolution
                entry["resolution_in_declared_plants"] = 0.0
                entry["plant_in_resolutions"] = None
            if plant == "Ki":
                print(f"{key}: Ki x -1", flush=True)
                plant_measured, plant_found = planted(key, "Ki", 1.0, -1.0)
                gates = second(key, plant_measured, plant_found)
                chosen = next(g for g in gates if g.quantity == gate)
                entry["sizes"]["-1"] = {"misses_in_tolerances": misses(gates),
                                        "gate_miss_in_tolerances": chosen.miss / chosen.tolerance,
                                        "gate_miss": chosen.miss}
            by_plant[plant] = entry
        record["by_frequency"][key] = {"unplanted": unplanted, "plants": by_plant}
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
