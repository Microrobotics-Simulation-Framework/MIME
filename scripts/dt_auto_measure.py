"""``dt="auto"``'s measurement (F1b's gate 3 (d)): a report with no pass line (the line is the maintainer's). Over a
case set, each built with ``dt="auto"``: the scheme and step it chose, the CFL number it predicted from the drive,
and over the run the largest CFL number measured on the faces, their ratio, whether the per-step stop fired, and
whether the flow went non-finite; the steps a period of a pulsatile drive.

The cases: the straight tube steady (D/h 16 and 32); Womersley at alpha 6 (D/h 32), from the drive's peak and from
the worst phase (the drive starting at zero, where the start from rest overshoots most); the tube tilted along the lattice
direction (1, 0, 3) (D/h 16); a viscous fluid where IMEX Euler is chosen (D/h 16); and, on the step alone, a plug flow
along the body diagonal at RK3's line along that direction and at the decided 2.0. Run one dtype at a time, alone:

    JAX_PLATFORMS=cpu python scripts/dt_auto_measure.py --dtype float64 > dt_auto_float64.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import jax
import numpy as np


def _experiment(*, cells: int, g: float, amplitude: float = 0.0, frequency: float = 0.0, phase: float = 0.0,
                axis=(0.0, 0.0, 1.0), viscosity: float = 1.0, length: float = 0.25) -> Any:
    from mime.dynamics import FVM
    from mime.experiment import Experiment
    from mime.interactions import BodyForce
    from mime.scene import PeriodicBox, Scene, Tube, Z
    from mime.things import Medium, Structure
    from mime.units import Hz, kg_per_m3, m_per_s2, mm, mPa_s

    exp = Experiment("dt_auto", scene=Scene.new(up=Z, units="m"), dt="auto")
    exp.add_structure(Structure("tube", geometry=Tube(axis=axis, inner_radius=mm(0.5), wall=mm(0.2), length=mm(10.0))))
    exp.add_medium(Medium("water", fills="tube.lumen",
                          fields={"flow": FVM(domain=PeriodicBox(length=mm(length)), cells_across=cells)}))
    exp.declare("water.viscosity", mPa_s(viscosity), reason="the case's fluid")
    exp.declare("water.density", kg_per_m3(1000.0), reason="the case's fluid")
    pulsatile = {"amplitude": m_per_s2(amplitude), "frequency": Hz(frequency), "phase": phase} if amplitude else {}
    norm = math.sqrt(sum(c * c for c in axis))
    exp.attach(BodyForce(along=tuple(c / norm for c in axis), steady=m_per_s2(g), **pulsatile), medium="water")
    return exp


def _predicted(built: Any) -> float | None:
    for line in built.report.checked.get("water.flow", ()):
        if line.startswith("the predicted CFL"):
            return float(line.split("=")[1].split()[0])
    return None


def run_case(name: str, steps: int, **kwargs: Any) -> dict[str, Any]:
    from mime.nodes.flow._waveform import waveform

    built = _experiment(**kwargs).build()
    node = built.graph.get_node("water.flow")
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = built.report.dt
    law = next(n for n in built.graph.node_names if n.startswith("body_force"))
    q = built.graph.get_node(law).constants()
    w = waveform(q["along"], np.asarray(q["steady"]), q["amplitude"], q["frequency"], q["phase"])
    dtype = jax.numpy.asarray(node.initial_state()["velocity_z"]).dtype
    inputs = {"waveform": jax.numpy.asarray(w, dtype=dtype),
              "body_force": jax.numpy.zeros(3, dtype=dtype)}
    held = "waveform" if type(node).__name__ == "ExplicitFlowNode" else "body_force"

    def body(state: Any, _: Any) -> tuple[Any, Any]:
        given = dict(inputs)
        if held == "body_force":
            t = (state["steps"].astype(dtype) + 1.0) * dt
            given["body_force"] = jax.numpy.asarray(w[:3]) * (w[3] + w[4] * jax.numpy.cos(2 * math.pi * w[5] * t + w[6]))
        ended = node.advance(p, state, given, dt)
        speed = jax.numpy.max(jax.numpy.stack([jax.numpy.max(jax.numpy.abs(ended[f"velocity_{c}"])) for c in "xyz"]))
        return ended, speed

    ended, speeds = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(dict(node.initial_state()))
    speeds = np.asarray(speeds, np.float64)
    h = float(node.constants()["spacing"])
    measured = float(np.nanmax(speeds)) * dt / h if np.all(np.isfinite(speeds)) else float("inf")
    predicted = _predicted(built)
    period = 1.0 / kwargs["frequency"] if kwargs.get("frequency") else None
    return {"case": name, "scheme": type(node).__name__, "dt": dt, "steps": steps, "predicted_cfl": predicted,
            "measured_cfl": measured, "ratio": None if not predicted else predicted / measured if measured else None,
            "stopped_at": int(np.asarray(ended["stopped_at"])), "finite": bool(np.all(np.isfinite(speeds))),
            "steps_a_period": None if period is None else period / dt, "report": list(built.report.checked.get(
                "water.flow", ()))}


def plug_case(direction: tuple[float, float, float], cfl: float, viscous_number: float, steps: int = 600) -> dict:
    """RK3's step alone on a 16^3 periodic box, a uniform flow along ``direction`` at face CFL ``cfl`` with 1e-7 noise:
    the perturbation's growth a step."""
    from mime.nodes.flow.fvm import MACGrid, _rk3

    grid = MACGrid((16, 16, 16), 1.0)
    dt = viscous_number
    d = np.asarray(direction, float) / np.max(np.abs(direction))
    rng = np.random.default_rng(0)
    u = grid.project(tuple(jax.numpy.asarray(cfl * d[c] / dt + 1e-7 * rng.standard_normal(grid.cells))
                           for c in range(3)))
    zero = grid.zeros(u[0].dtype)

    def body(carry: Any, _: Any) -> tuple[Any, Any]:
        v, pr = carry
        v, pr, _ = _rk3.step(grid, v, pr, 1.0, zero, lambda k, t: zero, True, 0.0, dt)
        return (v, pr), jax.numpy.max(jax.numpy.abs(v[0] - jax.numpy.mean(v[0])))

    _, deviation = jax.jit(lambda c: jax.lax.scan(body, c, None, length=steps))((u, jax.numpy.zeros(grid.cells)))
    deviation = np.asarray(deviation, np.float64)
    finite = deviation[np.isfinite(deviation)]
    return {"case": f"plug {direction} face cfl {cfl:.3g} r {viscous_number}", "finite": bool(np.all(np.isfinite(deviation))),
            "growth": float(finite[-1] / max(finite[0], 1e-300)) if finite.size else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    nu, radius = 1e-6, 0.5e-3
    omega6 = (6.0 / radius) ** 2 * nu
    rows = [
        run_case("straight, steady, D/h 16", 3000, cells=16, g=20.0),
        run_case("straight, steady, D/h 32", 6000, cells=32, g=20.0),
        run_case("Womersley alpha 6, D/h 32", 1500, cells=32, g=0.0, amplitude=20.0, frequency=omega6 / (2 * math.pi)),
        run_case("Womersley alpha 6 from the worst phase, D/h 32", 1500, cells=32, g=0.0, amplitude=20.0,
                 frequency=omega6 / (2 * math.pi), phase=math.pi / 2),
        run_case("tilted (1, 0, 3), steady, D/h 16", 3000, cells=16, g=20.0, axis=(1.0, 0.0, 3.0),
                 length=0.25 * math.sqrt(10.0)),
        run_case("viscous fluid (IMEX Euler), D/h 16", 300, cells=16, g=0.5, viscosity=100.0),
    ]
    line = 1.7 / 3.0
    rows += [plug_case((1.0, 1.0, 1.0), line, 0.15), plug_case((1.0, 1.0, 1.0), line, 0.01),
             plug_case((1.0, 1.0, 1.0), 2.0, 0.15), plug_case((1.0, 0.0, 0.0), 2.0, 0.15)]
    print(json.dumps({"dtype": arguments.dtype, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
