"""Measure how often each planted fault of A0 on its dynamic rungs is caught by its comparison, over sizes from below
its resolution to ten times it, and the responses its resolutions rest on:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_dynamic_controls.py [sensitivities]

``only PART ...`` sweeps those plants alone, the others' draws kept. With ``sensitivities`` it plants each fault of ``a0_dynamic_plants.py`` at relative sizes 1e-3 and 1e-2 of either
sign, runs 0.3's A0 on its dynamic rungs in float32 at 3 Hz, and records every gated quantity's relative move per unit
of the fault, and which gate resolves the fault finest (``analysed``): the response a plant's resolution rests on.
``reanalyse`` analyses the recorded moves again, with no run. Without it, it draws for each plant 20 sizes between 1.5 and 10 resolutions with random signs (a fixed seed)
and 6 below the resolution (0.1 to 0.6 of it), and records whether the comparison failed, which gates did, and whether
the quantity the plant names missed. Writes ``references/a0_dynamic_controls_measured.json`` (the sweep keeps the
sensitivities already recorded). A fault above the resolution must be caught every time. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_dynamic_controls_measured.json"
SEED = 20261031
ABOVE, BELOW = 20, 6
SIZES = (1e-3, 1e-2)


def sensitivities() -> dict[str, Any]:
    from a0_dynamic_compare import GATED, PREDICTED, TOLERANCES, measure
    from a0_dynamic_plants import KEY, PLANTS

    from a0_compare import difference

    found: dict[str, Any] = {}
    for plant in PLANTS:
        moves: dict[str, dict[str, float]] = {}
        for size in SIZES:
            for sign in (1.0, -1.0):
                factor = 1.0 + sign * size
                with plant.context(factor):
                    measured, _ = measure(KEY, **plant.settings(factor))
                moves[f"{sign * size:+g}"] = {name: difference(name, measured[name], PREDICTED[KEY][name])
                                              for name in GATED}
                print(plant.part, f"{sign * size:+g}", {name: f"{v / TOLERANCES[name]:+.3g}"
                                                       for name, v in moves[f'{sign * size:+g}'].items()}, flush=True)
        found[plant.part] = analysed(moves)
    return found


def analysed(moves: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The responses of every gate to a fault planted at each size both ways, and the gate whose resolution for it is
    finest. A gate that reads the rotor's single-precision angle (the finding, ``a0_dynamic_findings.json``) moves by
    up to one unit's move whatever the fault, whenever a fault moves where the angle lands by a unit, so its resolution
    is its tolerance and that band over its response; the others' is their tolerance over their response."""
    from a0_dynamic_compare import GATED, TOLERANCES
    from a0_dynamic_plants import BAND

    response = {name: {f"{size:g}": (moves[f"+{size:g}"][name] - moves[f"-{size:g}"][name]) / (2 * size)
                       for size in SIZES} for name in GATED}
    resolution = {name: (TOLERANCES[name] + BAND.get(name, 0.0)) / abs(response[name]["0.01"])
                  if response[name]["0.01"] else float("inf") for name in GATED}
    sharpest = min(GATED, key=lambda name: resolution[name])
    return {"moves_from_the_prediction": moves, "response_per_unit": response, "resolution_at_0.01": resolution,
            "sharpest_gate": sharpest}


def sweep(only: tuple[str, ...] = ()) -> dict[str, Any]:
    """Each plant's draws (those named in ``only``, all when it is empty; the draws are the seed's whichever run),
    held to the first declaration's tolerances (``a0_dynamic_compare.failures``). The recorded sweep was run under the
    finding's allowance, since removed as a gate; at 3 Hz, where every plant runs, it was never granted (the angle sits
    0 units apart unplanted, and any plant that moves it moves it 4 or more), so the two rules agree on every draw."""
    from a0_dynamic_compare import compare, failures, measure, units_apart
    from a0_dynamic_plants import KEY, PLANTS

    generator = np.random.default_rng(SEED)
    found: dict[str, Any] = {}
    for plant in PLANTS:
        sizes = [float(s) for s in generator.uniform(1.5, 10.0, ABOVE)] + \
                [float(s) for s in generator.uniform(0.1, 0.6, BELOW)]
        signs = [float(s) for s in generator.choice([-1.0, 1.0], len(sizes))]
        if only and plant.part not in only:
            continue
        draws = []
        for size, sign in zip(sizes, signs, strict=True):
            with plant.planted(size, sign):
                measured, run = measure(KEY, **plant.apply(size, sign))
            failed = failures(compare(measured, KEY))
            draws.append({"size_in_resolutions": size, "sign": sign, "failed_gates": failed,
                          "angle_units_apart": list(units_apart(KEY, run)),
                          "caught": bool(failed), "quantity_missed": plant.quantity in failed})
            print(plant.part, f"{sign * size:+.2f}", failed, flush=True)
        above = [d for d in draws if d["size_in_resolutions"] >= 1.5]
        below = [d for d in draws if d["size_in_resolutions"] < 1.5]
        found[plant.part] = {
            "planted": plant.name, "quantity": plant.quantity, "resolution": plant.resolution, "draws": draws,
            "caught_above_the_resolution": f"{sum(d['caught'] for d in above)}/{len(above)}",
            "quantity_missed_above": f"{sum(d['quantity_missed'] for d in above)}/{len(above)}",
            "caught_below_the_resolution": f"{sum(d['caught'] for d in below)}/{len(below)}"}
    return found


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    held = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    if sys.argv[1:] == ["sensitivities"]:
        held["sensitivities"] = sensitivities()
    elif sys.argv[1:] == ["reanalyse"]:  # the recorded moves, analysed again, with no run
        held["sensitivities"] = {part: analysed(found["moves_from_the_prediction"])
                                 for part, found in held["sensitivities"].items()}
    else:
        only = tuple(sys.argv[2:]) if sys.argv[1:2] == ["only"] else ()
        held["plants"] = {**held.get("plants", {}), **sweep(only)}
        held["seed"] = SEED
        held["rule"] = ("each draw held to the first declaration's tolerances, a0_dynamic_compare.failures: each gate "
                        "beyond its tolerance")
    OUTPUT.write_text(json.dumps(held, indent=1) + "\n")
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
