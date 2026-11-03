"""The examples' spacing tables, measured: for each example (``examples/sphere_settling.py``,
``examples/screw_in_tube.py``) the time one run takes and the number it gives at a ladder of ``spacing``, against the
converged value, which ``docs/choosing_spacing.md`` prints and ``docs/data/spacing_tables.json`` holds.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/example_tables.py runs <out.json> [0.83,0.7,...]
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/example_tables.py steps <out.json>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/default_spacing_table.py miss <misses.json>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/example_tables.py coarse <coarse.json>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/default_spacing_table.py default-miss \
        <default_miss.json> <misses.json>
    uv run --no-sync python scripts/example_tables.py merge <runs.json> <steps.json> <misses.json> <coarse.json> \
        <default_miss.json> <out.json>
    uv run --no-sync python scripts/example_tables.py render <out.json>        # the tables the page prints

``runs`` times each example whole (the screw at the spacings in mm that follow, a ladder of six if none) (``build()`` then ``run()``, the first call in the process compiling, as a script's
first run does) in the precision a script runs in by default (float32), at each spacing, and reads the number the run
gives: the sphere's settled speed over Stokes's law, the screw's speed over its last turn per turn of the field.
``default_spacing_table.py miss`` (float64, so the number is the discretisation's and not float32's rounding) gives
the sphere's drag against Stokes's law, and the screw's speed per turn from its 6 x 6 resistance against the
finest spacing measured, which is no converged value (no closed form exists for the screw, and the successive spacings
still change the speed). The screw's resistance does not depend on the liquid or on the field, so the same rows serve
the example and the walk's. ``steps`` runs both at other steps. ``coarse`` takes the screw's resistance in float64, in its tube, at the coarse
spacings a first look can land on and at the default's (``Drag.MRS()`` with no spacing), which the ladder above leaves out.
``merge`` joins the five (the default's misses, ``default_spacing_table.py default-miss``, the fifth). Every row is drag
0.4.0's (phase 7m-b), the blob and the quadrature's refinement at the family's defaults; the point-force runs that
phase 7l-c took on CI's runner are not retaken, and the page no longer carries them. Not collected by pytest."""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
#: The spacings run: the sphere's as fractions of its radius, the screw's in mm.
SPHERE_RADIUS_UM = 10.0
SPHERE_FRACTIONS = (0.4, 0.3, 0.25, 0.2, 0.133, 0.1)
SCREW_SPACINGS_MM = (0.83, 0.7, 0.6, 0.5, 0.4, 0.3)
#: The steps run at the starting spacing: the sphere's in ms (a run of 3 ms is a whole number of each), the screw's.
SPHERE_STEPS_MS = (0.05, 0.1, 0.2, 0.5, 1.0)
SCREW_STEPS_MS = (10.0, 20.0, 40.0, 100.0, 250.0)
SCREW_DT_SPACING_MM = 0.7
#: The coarse spacings of the screw's resistance (mm), and ``None`` for the default's: ordered by samples, not spacing.
COARSE_SPACINGS_MM: tuple[float | None, ...] = (1.2, 1.0, 0.9, 0.83, None, 0.7)


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def machine() -> dict[str, Any]:
    """What the numbers were measured on: the CPU's model and thread count, the platform and JAX's device."""
    import os

    import jax

    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() \
        else platform.processor()
    return {"cpu": model, "threads": os.cpu_count(), "platform": platform.platform(), "devices": [str(d) for d in jax.devices()],
            "x64": bool(jax.config.read("jax_enable_x64")), "python": platform.python_version()}


def load_average() -> float:
    """The machine's one-minute load average, noted beside each time: the machine is shared, and a time measured under
    a load far above its cores' count is an overestimate."""
    import os

    return os.getloadavg()[0]


def runs(screw_spacings: tuple[float, ...] = SCREW_SPACINGS_MM) -> dict[str, Any]:
    from mime.units import m_per_s, mm_per_s

    sphere = load("example_sphere", ROOT / "examples" / "sphere_settling.py")
    screw = load("example_screw", ROOT / "examples" / "screw_in_tube.py")
    out: dict[str, Any] = {"sphere": [], "screw": [], "machine": machine()}
    for fraction in SPHERE_FRACTIONS:
        started = time.perf_counter()
        result, ratio = sphere.run(SPHERE_RADIUS_UM, fraction)
        seconds = time.perf_counter() - started
        out["sphere"].append({"spacing_per_radius": fraction, "seconds": seconds, "speed_over_stokes": ratio,
                              "reynolds": result.watches[0].reynolds, "load_average": load_average()})
        print("sphere", out["sphere"][-1], flush=True)
    for spacing in screw_spacings:
        started = time.perf_counter()
        result = screw.run(spacing)
        seconds = time.perf_counter() - started
        speed = float(result["swim_speed:umr"].mean_speed.to(m_per_s))
        out["screw"].append({"spacing_mm": spacing, "seconds": seconds, "speed_m_per_s": speed,
                             "speed_per_turn_m": speed / screw.FREQUENCY_HZ, "reynolds": result.watches[0].reynolds,
                             "load_average": load_average()})
        print("screw", out["screw"][-1], flush=True)
    return out


def steps() -> dict[str, Any]:
    """The same runs at other steps, at the starting spacing of the sphere and at ``SCREW_DT_SPACING_MM`` for the
    screw: the number each gives, and what it takes."""
    from mime.units import m_per_s, s

    sphere = load("example_sphere", ROOT / "examples" / "sphere_settling.py")
    screw = load("example_screw", ROOT / "examples" / "screw_in_tube.py")
    out: dict[str, Any] = {"sphere": [], "screw": [], "machine": machine(), "screw_spacing_mm": SCREW_DT_SPACING_MM}
    for dt_ms in SPHERE_STEPS_MS:
        started = time.perf_counter()
        result = sphere.sphere_settling(SPHERE_RADIUS_UM, 0.25, dt_ms).build().run(sphere.duration(), history=True)
        seconds = time.perf_counter() - started
        ratio = float(result["swim_speed:ball"].mean_speed.to(m_per_s)) / sphere.stokes_speed(SPHERE_RADIUS_UM)
        out["sphere"].append({"dt_ms": dt_ms, "steps": result.steps, "seconds": seconds, "speed_over_stokes": ratio,
                              "load_average": load_average()})
        print("sphere", out["sphere"][-1], flush=True)
    for dt_ms in SCREW_STEPS_MS:
        started = time.perf_counter()
        result = screw.screw_in_tube(SCREW_DT_SPACING_MM, dt_ms=dt_ms).build().run(s(screw.DURATION_S), history=True)
        seconds = time.perf_counter() - started
        speed = float(result["swim_speed:umr"].mean_speed.to(m_per_s))
        out["screw"].append({"dt_ms": dt_ms, "steps": result.steps, "seconds": seconds, "speed_m_per_s": speed,
                             "speed_per_turn_m": speed / screw.FREQUENCY_HZ,
                             "load_average": load_average()})
        print("screw", out["screw"][-1], flush=True)
    return out


def coarse() -> dict[str, Any]:
    """The screw's speed per turn from its 6 x 6 resistance (float64, in the tube) at the coarse spacings and at the
    default's, with the samples each gives: what ``scripts/default_spacing_table.py miss`` measures at a ladder, at the
    spacings that script's ladder leaves out. Run under ``JAX_ENABLE_X64=1``."""
    tables = load("default_spacing_table_for_the_page", ROOT / "scripts" / "default_spacing_table.py")
    rows = []
    for spacing_mm in COARSE_SPACINGS_MM:
        built = tables.screw_task(None if spacing_mm is None else spacing_mm * 1e-3, tube=True).build()
        r, count = tables.resistance(built, "umr")
        rows.append({"spacing_m": tables.taken_spacing(built, "umr") if spacing_mm is None else spacing_mm * 1e-3,
                     "default": spacing_mm is None, "samples": count, "speed_per_turn_m": tables.speed_per_turn(r)})
        print("coarse", rows[-1], flush=True)
    return {"rows": rows, "machine": machine()}


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def render(merged: dict[str, Any]) -> dict[str, str]:
    """The five tables of ``docs/choosing_spacing.md`` from the merged measurements, by name: the page prints them
    verbatim, and ``tests/docs/test_spacing_tables.py`` holds the page to this function's output. Every row is drag
    0.4.0's, the blob and the quadrature's refinement at the family's defaults."""
    sphere_misses = {round(row["spacing_per_radius"], 3): row for row in merged["misses"]["rows"]["sphere"]}
    screw_misses = {round(row["spacing_m"] * 1e3, 3): row for row in merged["misses"]["rows"]["screw_tube"]}
    free_misses = {round(row["spacing_m"] * 1e3, 3): row for row in merged["misses"]["rows"]["screw_free"]}
    sphere = []
    for row in merged["runs"]["sphere"]:
        held = sphere_misses[round(row["spacing_per_radius"], 3)]
        sphere.append([f"{row['spacing_per_radius']:g}", str(held["samples"]), f"{row['seconds']:.1f}",
                       f"{100.0 * (row['speed_over_stokes'] - 1.0):+.2f}%",
                       f"{100.0 * held['translation_vs_stokes']:+.2f}%", f"{100.0 * held['rotation_vs_stokes']:+.2f}%"])
    sphere_steps = [[f"{row['dt_ms']:g}", str(row["steps"]), f"{row['seconds']:.1f}",
                     f"{100.0 * (row['speed_over_stokes'] - 1.0):+.2f}%"] for row in merged["steps"]["sphere"]]
    finest = merged["misses"]["rows"]["screw_tube"][-1]

    def near(table: dict[float, Any], spacing_mm: float) -> Any:
        return next((r for key, r in table.items() if abs(key - spacing_mm) < 0.015), None)

    screw = []
    for row in merged["runs"]["screw"]:
        held, free = near(screw_misses, row["spacing_mm"]), near(free_misses, row["spacing_mm"])
        assert held is not None and free is not None, row
        screw.append([f"{row['spacing_mm']:g}", str(held["samples"]), f"{row['seconds']:.1f}",
                      f"{row['speed_per_turn_m'] * 1e6:.1f}", f"{held['speed_per_turn_m'] * 1e6:.1f}",
                      f"{100.0 * held['vs_finest']:+.1f}%", f"{100.0 * free['vs_reference']:+.1f}%"])
    default = next(row for row in merged["coarse"]["rows"] if row["default"])
    default_free = merged["default_miss"]["rows"]["screw_free"]
    at = next(i for i, row in enumerate(screw) if float(row[0]) < default["spacing_m"] * 1e3)
    screw.insert(at, [f"{default['spacing_m'] * 1e3:.3f} (the default)", str(default["samples"]), "not run",
                      "not run", f"{default['speed_per_turn_m'] * 1e6:.1f}",
                      f"{100.0 * (default['speed_per_turn_m'] / finest['speed_per_turn_m'] - 1.0):+.1f}%",
                      f"{100.0 * default_free['vs_reference']:+.1f}%"])
    free_finest = merged["misses"]["rows"]["screw_free"][-1]
    screw.append([f"{finest['spacing_m'] * 1e3:g}", str(finest["samples"]), "not run", "not run",
                  f"{finest['speed_per_turn_m'] * 1e6:.1f}", "the reference",
                  f"{100.0 * free_finest['vs_reference']:+.1f}% (at {free_finest['spacing_m'] * 1e3:g} mm)"])
    coarse_rows = sorted(merged["coarse"]["rows"], key=lambda row: row["samples"])
    ladder = [[f"{row['spacing_m'] * 1e3:.3f}".rstrip("0").rstrip(".") + (" (the default)" if row["default"] else ""),
               str(row["samples"]), f"{row['speed_per_turn_m'] * 1e6:.1f}",
               f"{100.0 * (row['speed_per_turn_m'] / finest['speed_per_turn_m'] - 1.0):+.1f}%"]
              for row in coarse_rows]
    screw_steps = [[f"{row['dt_ms']:g}", str(row["steps"]), f"{row['seconds']:.1f}",
                    f"{row['speed_per_turn_m'] * 1e6:.1f}"] for row in merged["steps"]["screw"]]
    return {
        "sphere_spacing": table(["spacing / radius", "force points", "run time, laptop (s)",
                                 "settled speed against Stokes's law", "drag against Stokes's law (float64)",
                                 "rotational drag against Stokes's law (float64)"], sphere),
        "sphere_dt": table(["dt (ms)", "steps", "run time, laptop (s)", "settled speed against Stokes's law"],
                           sphere_steps),
        "screw_spacing": table(["spacing (mm)", "force points", "run time, laptop (s)",
                                "speed per turn, from the run (um)", "speed per turn, from the resistance (um, float64)",
                                "against the finest, in the tube", "in free space, against the reference"], screw),
        "screw_coarse": table(["spacing (mm)", "force points", "speed per turn, from the resistance (um, float64)",
                               "against the finest"], ladder),
        "screw_dt": table(["dt (ms)", "steps", "run time, laptop (s)", "speed per turn (um)"], screw_steps)}


def facts(merged: dict[str, Any]) -> dict[str, str]:
    """The numbers the page's prose states, each as the phrase the page prints verbatim (``tests/docs/
    test_spacing_tables.py`` holds the page to them as it holds the tables, whitespace aside): so that no number in the
    prose is typed by hand and none can drift from the data."""
    sphere, screw = merged["runs"]["sphere"], merged["runs"]["screw"]
    misses = merged["misses"]["rows"]
    held = {round(r["spacing_per_radius"], 3): r for r in misses["sphere"]}
    resistance = {round(r["spacing_m"] * 1e3, 3): r for r in misses["screw_tube"]}
    free = {round(r["spacing_m"] * 1e3, 3): r for r in misses["screw_free"]}
    seconds = {row["spacing_per_radius"]: row["seconds"] for row in sphere}

    def percent(value: float, digits: int = 2) -> str:
        return f"{100.0 * value:+.{digits}f}%"

    out: dict[str, str] = {}
    fine, finer = held[0.2], held[0.1]
    out["cost_sphere"] = (f"from {fine['samples']} to {finer['samples']} force points "
                          f"({finer['samples'] / fine['samples']:.1f} times) the sphere's run goes from "
                          f"{seconds[0.2]:.1f} s to {seconds[0.1]:.1f} s on the laptop "
                          f"({seconds[0.1] / seconds[0.2]:.1f} times)")
    lap = {round(r["spacing_mm"], 3): r["seconds"] for r in screw}
    out["cost_screw"] = (f"the screw's, from {resistance[0.7]['samples']} to {resistance[0.3]['samples']} force points "
                         f"({resistance[0.3]['samples'] / resistance[0.7]['samples']:.1f} times, from 0.7 mm to 0.3 mm), "
                         f"goes from {lap[0.7]:.1f} s to {lap[0.3]:.1f} s on the laptop ({lap[0.3] / lap[0.7]:.1f} times)")
    top, bottom = held[0.4], held[0.1]
    out["sphere_error_falls"] = (f"from 0.4 to 0.1 of the radius, a factor of {0.4 / 0.1:.0f} in spacing, the drag's error "
                                 f"falls from {percent(top['translation_vs_stokes'])} to "
                                 f"{percent(bottom['translation_vs_stokes'])}, a factor of "
                                 f"{top['translation_vs_stokes'] / bottom['translation_vs_stokes']:.1f}")
    out["sphere_to_a_per_cent"] = (f"The drag is within {100.0 * held[0.25]['translation_vs_stokes']:.2f}% at a quarter of "
                                   f"the radius ({held[0.25]['samples']} force points) and "
                                   f"{100.0 * held[0.2]['translation_vs_stokes']:.2f}% at a fifth ({held[0.2]['samples']})")
    out["sphere_rotation"] = (f"{percent(held[0.25]['rotation_vs_stokes'])} against the translation's "
                              f"{percent(held[0.25]['translation_vs_stokes'])} at the starting size")
    reynolds = next(row["reynolds"] for row in sphere if row["spacing_per_radius"] == 0.25)
    out["sphere_reynolds"] = f"The example's sphere settles at a Reynolds number of {reynolds:.2g}"
    out["sphere_reynolds_50"] = (f"The Reynolds number goes as the cube of the radius (the settling speed as its square), so "
                                 f"a sphere of fifty micrometres radius, five times as large, settles at about "
                                 f"{125.0 * reynolds:.1f}")
    steps_ = {round(r["dt_ms"], 3): r for r in merged["steps"]["sphere"]}
    example = load("example_sphere_for_the_facts", ROOT / "examples" / "sphere_settling.py")
    relaxation = 2.0 * example.STEEL * (SPHERE_RADIUS_UM * 1e-6) ** 2 / (9.0 * example.VISCOSITY)
    out["sphere_dt"] = (f"The settled speed is {percent(steps_[0.05]['speed_over_stokes'] - 1.0)} of Stokes's law at 0.05 ms and "
                        f"{percent(steps_[0.2]['speed_over_stokes'] - 1.0)} at 0.2 ms, then {percent(steps_[0.5]['speed_over_stokes'] - 1.0)} "
                        f"at 0.5 ms and {percent(steps_[1.0]['speed_over_stokes'] - 1.0)} at 1 ms, as the step passes the "
                        f"sphere's relaxation time to its settling speed, 2 rho a^2 / (9 mu) = {relaxation * 1e3:.2f} ms for "
                        f"this sphere")
    out["screw_start"] = (f"at the example's starting size the speed is {percent(resistance[0.7]['vs_finest'], 1)} from the "
                          f"finest in the tube, and the same screw in free space {percent(free[0.7]['vs_reference'], 1)} "
                          f"from the shipped reference")
    out["screw_finest"] = (f"in the tube the speed at 0.3 mm ({resistance[0.3]['samples']} force points) lies "
                           f"{percent(resistance[0.3]['speed_per_turn_m'] / resistance[0.16]['speed_per_turn_m'] - 1.0, 1)}"
                           f" from the speed at 0.16 mm ({resistance[0.16]['samples']})")
    default_free = merged["default_miss"]["rows"]["screw_free"]
    out["screw_default"] = (f"the default ({default_free['samples']} force points) is "
                            f"{percent(default_free['vs_reference'], 1)} from the shipped reference in free space")
    loads = [r["load_average"] for kind in ("runs", "steps") for name in ("sphere", "screw") for r in merged[kind][name]]
    out["load"] = f"its one-minute load average was between {min(loads):.0f} and {max(loads):.0f}"
    screw_steps = {round(r["dt_ms"], 3): r for r in merged["steps"]["screw"]}
    out["screw_dt"] = (f"The screw's speed per turn is {screw_steps[10.0]['speed_per_turn_m'] * 1e6:.1f} um at 10 ms, "
                       f"{screw_steps[100.0]['speed_per_turn_m'] * 1e6:.1f} um at 100 ms and "
                       f"{screw_steps[250.0]['speed_per_turn_m'] * 1e6:.1f} um at 250 ms")
    return out


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "steps":
        Path(argv[1]).write_text(json.dumps(steps(), indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) in (2, 3) and argv[0] == "runs":
        spacings = tuple(float(v) for v in argv[2].split(",")) if len(argv) == 3 else SCREW_SPACINGS_MM
        Path(argv[1]).write_text(json.dumps(runs(spacings), indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) == 2 and argv[0] == "coarse":
        Path(argv[1]).write_text(json.dumps(coarse(), indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) == 7 and argv[0] == "merge":
        runs_, steps_, misses, coarse_, default_miss = (json.loads(Path(p).read_text(encoding="utf-8"))
                                                       for p in argv[1:6])
        merged = {"runs": runs_, "steps": steps_, "misses": misses, "coarse": coarse_, "default_miss": default_miss}
        Path(argv[6]).write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) == 2 and argv[0] == "render":
        merged = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        for name, text in {**render(merged), **facts(merged)}.items():
            print(f"{name}\n\n{text}\n")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
