"""Measure how often each planted fault is caught by the A0 comparison, over sizes from below its resolution to ten
times it, and the sensitivities its resolutions rest on:

    JAX_PLATFORMS=cpu .venv/bin/python tests/differential/measure_a0_controls.py [sensitivities]

For each fault in ``a0_plants.py`` it draws 20 sizes between 1.5 and 10 resolutions, with random signs (a fixed seed),
and 6 below the resolution (0.1 to 0.6 of it; the controller's latency, planted only in whole steps, and the motor's
speed, which is a float32 factor and so cannot be planted below one unit in its last place, have none), runs 0.3's A0 in
float32 at 3 Hz with the fault planted, and records whether the comparison with the declared prediction failed, which
gates did, and whether the quantity the fault moves most missed. Before the sweep it measures the response of that
quantity to each fault that has a resolution to find (the dipole's moment, the gradient force alone, the motor's
speed), at sizes both sides of zero, and records the response. Writes ``references/a0_controls_measured.json``; with the
argument ``sensitivities`` it measures only the responses and prints them. A fault above the resolution must be caught
every time, and one below it should not be: a control that fails at one run in eight (the contact oracle's, once) is
not a control. Not collected by pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
OUTPUT = HERE / "references" / "a0_controls_measured.json"
SEED = 20261030
ABOVE, BELOW = 20, 6


def _miss(rows: Any, name: str) -> float:
    return next(row.miss for row in rows if row.quantity == name)


def sensitivities() -> dict[str, Any]:
    """The response of the quantity each fault moves most, measured by planting the fault at sizes both sides of zero:
    the quantity's miss per unit of the fault, and so the fault that moves it by its tolerance."""
    from a0_compare import MOTOR_OUTPUT, MOTOR_OUTPUT_TOLERANCE, TOLERANCES, compare, measure, missed
    from a0_plants import _gradient_scaled, _motor_scaled
    from chain_cases import MAGNET_MOMENT

    lag, speed = "phase_lag_field_minus_moment_rad", "swim_speed_axial_mean_velocity_m_s"
    tolerance_of = {**TOLERANCES, MOTOR_OUTPUT: MOTOR_OUTPUT_TOLERANCE}
    found: dict[str, Any] = {}

    def planted(options: dict[str, Any], patch: Any, quantity: str) -> dict[str, Any]:
        with patch:
            rows = compare(measure("3hz", **options), "3hz")
        return {"miss": _miss(rows, quantity), "miss_in_tolerances": _miss(rows, quantity) / tolerance_of[quantity],
                "gates_missed": missed(rows)}

    def block(quantity: str, sizes: list[float], build: Any, linear: bool = True) -> dict[str, Any]:
        draws = {f"{sign * size:+g}": planted(*build(sign * size)) for size in sizes for sign in (1.0, -1.0)}
        found_block: dict[str, Any] = {"quantity": quantity, "tolerance": tolerance_of[quantity], "planted": draws}
        if linear:
            pairs = [(size, (draws[f"{size:+g}"]["miss"] - draws[f"{-size:+g}"]["miss"]) / (2 * size))
                     for size in sizes]
            found_block["response_per_unit"] = {f"{size:g}": response for size, response in pairs}
            found_block["resolution_by_size"] = {f"{size:g}": tolerance_of[quantity] / abs(response)
                                                 for size, response in pairs}
        return found_block

    from contextlib import nullcontext

    found["moment"] = block(lag, [1e-3], lambda x: ({"moment": MAGNET_MOMENT * (1.0 + x)}, nullcontext(), lag))
    found["moment"]["assumed_response_per_unit"] = "the lag's prediction, 5.4e-3 rad (lag ~ 1 / moment)"
    found["gradient_force"] = block(speed, [1e-3, 3e-3, 1e-2], lambda x: ({}, _gradient_scaled(1.0 + x), speed))
    # not linear: the motor's angle is a single-precision sum, whose step is rounded to a whole number of units in the
    # last place, so a speed change moves it in jumps of that unit or not at all
    found["motor_speed"] = block(MOTOR_OUTPUT, [1e-6, 1e-4, 3e-4, 1e-3, 3e-3],
                                 lambda x: ({}, _motor_scaled(1.0 + x), MOTOR_OUTPUT), linear=False)
    return found


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    from a0_compare import compare, measure, missed
    from a0_plants import PLANTS

    found = sensitivities()
    print(json.dumps(found, indent=1), flush=True)
    if sys.argv[1:] == ["sensitivities"]:
        return
    rng = np.random.default_rng(SEED)
    record: dict[str, object] = {"seed": SEED, "key": "3hz", "float32": True, "sensitivities": found, "plants": {}}
    for plant in PLANTS:
        draws: list[dict[str, object]] = []
        sizes = [float(x) for x in np.exp(rng.uniform(np.log(1.5), np.log(10.0), ABOVE))]
        if plant.part == "the controller's latency":
            sizes = [float(x) for x in rng.integers(1, 11, ABOVE)]
        elif plant.sub_resolution:
            sizes += [float(x) for x in rng.uniform(0.1, 0.6, BELOW)]
        for size in sizes:
            sign = float(rng.choice([-1.0, 1.0]))
            with plant.planted(size, sign):
                rows = compare(measure("3hz", **plant.apply(size, sign)), "3hz")
            found_by = missed(rows)
            draws.append({"size_in_resolutions": size, "sign": sign, "caught": bool(found_by),
                          "quantity_missed": plant.quantity in found_by, "missed": found_by,
                          "miss_of_the_quantity": _miss(rows, plant.quantity)})
            print(f"{plant.part:28s} size {size:6.2f} sign {sign:+.0f} caught {bool(found_by)} "
                  f"({plant.quantity} {'missed' if plant.quantity in found_by else 'held'})", flush=True)
        # the latency's whole steps are all "above": one step is the least that can be planted
        above = [d for d in draws
                 if d["size_in_resolutions"] >= 1.5 or not plant.sub_resolution]  # type: ignore[operator]
        below = [d for d in draws
                 if d["size_in_resolutions"] < 1.5 and plant.sub_resolution]  # type: ignore[operator]
        record["plants"][plant.part] = {  # type: ignore[index]
            "fault": plant.name, "quantity": plant.quantity, "resolution": plant.resolution,
            "caught_above_the_resolution": f"{sum(bool(d['caught']) for d in above)}/{len(above)}",
            "quantity_missed_above": f"{sum(bool(d['quantity_missed']) for d in above)}/{len(above)}",
            "caught_below_the_resolution": f"{sum(bool(d['caught']) for d in below)}/{len(below)}",
            "quantity_missed_below": f"{sum(bool(d['quantity_missed']) for d in below)}/{len(below)}",
            "draws": draws}
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
