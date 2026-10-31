"""The examples' spacing tables, measured: for each example (``examples/sphere_settling.py``,
``examples/screw_in_tube.py``) the time one run takes and the number it gives at a ladder of ``spacing``, against the
converged value, which ``docs/choosing_spacing.md`` prints and ``docs/data/spacing_tables.json`` holds.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/example_tables.py runs <out.json> [0.83,0.7,...]
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/example_tables.py steps <out.json>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --no-sync python scripts/default_spacing_table.py miss <misses.json>
    uv run --no-sync python scripts/example_tables.py merge <runs.json> <steps.json> <misses.json> <out.json> [<ci.json>]
    uv run --no-sync python scripts/example_tables.py render <out.json>        # the tables the page prints

``runs`` times each example whole (the screw at the spacings in mm that follow, a ladder of six if none) (``build()`` then ``run()``, the first call in the process compiling, as a script's
first run does) in the precision a script runs in by default (float32), at each spacing, and reads the number the run
gives: the sphere's settled speed over Stokes's law, the screw's speed over its last turn per turn of the field.
``default_spacing_table.py miss`` (float64, so the number is the discretisation's and not float32's rounding) gives
the sphere's drag against Stokes's law, and the screw's speed per turn from its 6 x 6 resistance against the
finest spacing measured, which is no converged value (no closed form exists for the screw, and the successive spacings
still change the speed). The screw's resistance does not depend on the liquid or on the field, so the same rows serve
the example and the walk's. ``steps`` runs both at other steps. ``merge`` joins the three, and the same runs on CI's runner (``ci.json``: its machine and the rows the runs printed)
when they were taken. Not collected by pytest."""

from __future__ import annotations

import importlib.util
import json
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


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def render(merged: dict[str, Any]) -> dict[str, str]:
    """The four tables of ``docs/choosing_spacing.md`` from the merged measurements, by name: the page prints them
    verbatim, and ``tests/docs/test_spacing_tables.py`` holds the page to this function's output."""
    ci = merged.get("ci", {})

    def on_ci(kind: str, key: str, value: float, section: str = "runs") -> str:
        """The run time on CI's runner for the row with ``key`` = ``value``, or a dash where CI did not run it."""
        found = next((r for r in ci.get(section, {}).get(kind, []) if abs(r[key] - value) < 1e-9), None)
        return f"{found['seconds']:.1f}" if found is not None else "-"

    sphere_misses = {round(row["spacing_per_radius"], 3): row for row in merged["misses"]["rows"]["sphere"]}
    screw_misses = {round(row["spacing_m"] * 1e3, 3): row for row in merged["misses"]["rows"]["screw_tube"]}
    sphere = []
    for row in merged["runs"]["sphere"]:
        held = sphere_misses[round(row["spacing_per_radius"], 3)]
        sphere.append([f"{row['spacing_per_radius']:g}", str(held["samples"]), f"{row['seconds']:.1f}",
                       on_ci("sphere", "spacing_per_radius", row["spacing_per_radius"]),
                       f"{100.0 * (row['speed_over_stokes'] - 1.0):+.2f}%",
                       f"{100.0 * held['translation_vs_stokes']:+.2f}%"])
    sphere_steps = [[f"{row['dt_ms']:g}", str(row["steps"]), f"{row['seconds']:.1f}",
                     on_ci("sphere", "dt_ms", row["dt_ms"], "steps"),
                     f"{100.0 * (row['speed_over_stokes'] - 1.0):+.2f}%"] for row in merged["steps"]["sphere"]]
    finest = merged["misses"]["rows"]["screw_tube"][-1]
    screw = []
    for row in merged["runs"]["screw"]:
        held = next((r for key, r in screw_misses.items() if abs(key - row["spacing_mm"]) < 0.015), None)
        assert held is not None, row
        screw.append([f"{row['spacing_mm']:g}", str(held["samples"]), f"{row['seconds']:.1f}",
                      on_ci("screw", "spacing_mm", row["spacing_mm"]),
                      f"{row['speed_per_turn_m'] * 1e6:.1f}", f"{held['speed_per_turn_m'] * 1e6:.1f}",
                      f"{100.0 * held['vs_finest']:+.0f}%"])
    screw.append([f"{finest['spacing_m'] * 1e3:g}", str(finest["samples"]), "not run", "not run", "not run",
                  f"{finest['speed_per_turn_m'] * 1e6:.1f}", "the reference"])
    screw_steps = [[f"{row['dt_ms']:g}", str(row["steps"]), f"{row['seconds']:.1f}",
                    on_ci("screw", "dt_ms", row["dt_ms"], "steps"),
                    f"{row['speed_per_turn_m'] * 1e6:.1f}"] for row in merged["steps"]["screw"]]
    return {
        "sphere_spacing": table(["spacing / radius", "samples", "run time, laptop (s)", "run time, CI (s)", "settled speed against Stokes's law",
                                 "drag against Stokes's law (float64)"], sphere),
        "sphere_dt": table(["dt (ms)", "steps", "run time, laptop (s)", "run time, CI (s)", "settled speed against Stokes's law"], sphere_steps),
        "screw_spacing": table(["spacing (mm)", "samples", "run time, laptop (s)", "run time, CI (s)", "speed per turn, from the run (um)",
                                "speed per turn, from the resistance (um, float64)", "against the finest"], screw),
        "screw_dt": table(["dt (ms)", "steps", "run time, laptop (s)", "run time, CI (s)", "speed per turn (um)"], screw_steps)}


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "steps":
        Path(argv[1]).write_text(json.dumps(steps(), indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) in (2, 3) and argv[0] == "runs":
        spacings = tuple(float(v) for v in argv[2].split(",")) if len(argv) == 3 else SCREW_SPACINGS_MM
        Path(argv[1]).write_text(json.dumps(runs(spacings), indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) in (5, 6) and argv[0] == "merge":
        runs_, steps_, misses = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[1:4])
        merged = {"runs": runs_, "steps": steps_, "misses": misses}
        if len(argv) == 6:
            merged["ci"] = json.loads(Path(argv[5]).read_text(encoding="utf-8"))
        Path(argv[4]).write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
        return 0
    if len(argv) == 2 and argv[0] == "render":
        for name, text in render(json.loads(Path(argv[1]).read_text(encoding="utf-8"))).items():
            print(f"{name}\n\n{text}\n")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
