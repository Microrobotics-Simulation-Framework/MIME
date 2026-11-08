"""F1b's budget (F0's B10, Q-c): a report with no pass line. RK3 on the empty periodic tube at three sizes, the ledger
off and on: the time per cell per step, the device bytes per cell (the compiled step's memory analysis where the
backend gives one) and the compile time; the prescribed flow's cost per reading at points; and ``dt="auto"``'s cost at
build (the step rule's prediction, its Womersley peak included).

Run it alone, on a quiet machine (its numbers are relative: a laptop's CPU is not a GPU), one dtype at a time:

    JAX_PLATFORMS=cpu python scripts/fvm_rk3_budget.py --dtype float64 > rk3_budget_float64.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import jax


def _node(dh: int, *, ledger: bool) -> Any:
    from mime.core import ledger as books
    from mime.nodes.flow.fvm import ExplicitFlowNode, periodic_tube_layout

    layout = periodic_tube_layout(0.5, 4.0 / dh, cells_across=dh)
    hosted = (
        books.LedgerDelivery("body_force", "drive:water.body_force", books.UNMETERED, books.HOSTED).encoded(),
        books.LedgerDelivery("viscous", "viscosity:water", books.DISSIPATIVE, books.HOSTED).encoded(),
        books.LedgerDelivery("penalty", "wall:vessel", books.DISSIPATIVE, books.HOSTED).encoded(),
    ) if ledger else ()
    return ExplicitFlowNode("flow", 0.1 * layout.spacing ** 2, cells_x=layout.cells[0], cells_y=layout.cells[1],
                            cells_z=layout.cells[2], spacing=layout.spacing, viscosity=1.0, density=1.0,
                            tube_centre=layout.centre, tube_axis=layout.axis, tube_radius=layout.radius,
                            ledger=hosted)


def measure(dh: int, steps: int, *, ledger: bool) -> dict[str, Any]:
    from mime.nodes.flow._waveform import waveform

    node = _node(dh, ledger=ledger)
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = node.delta_t
    drive = {"waveform": waveform((0.0, 0.0, 1.0), jax.numpy.asarray(8.0), 4.0, 1.0, 0.0)}

    def body(state: Any, _: Any) -> tuple[Any, Any]:
        return node.advance(p, state, drive, dt), None

    run = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps)[0])
    start = dict(node.initial_state())
    began = time.perf_counter()
    compiled = run.lower(start).compile()
    compile_seconds = time.perf_counter() - began
    memory = compiled.memory_analysis()
    held = None if memory is None else int(getattr(memory, "temp_size_in_bytes", 0)
                                           + getattr(memory, "argument_size_in_bytes", 0)
                                           + getattr(memory, "output_size_in_bytes", 0))
    jax.block_until_ready(compiled(start))
    began = time.perf_counter()
    jax.block_until_ready(compiled(start))
    seconds = time.perf_counter() - began
    c = node.constants()
    cells = c["cells_x"] * c["cells_y"] * c["cells_z"]
    return {"dh": dh, "cells": cells, "steps": steps, "ledger": ledger, "compile_s": compile_seconds,
            "run_s": seconds, "s_per_cell_step": seconds / (cells * steps),
            "bytes_per_cell": None if held is None else held / cells}


def prescribed(points: int, repeats: int = 20) -> dict[str, Any]:
    from mime.nodes.flow._waveform import waveform
    from mime.nodes.flow._womersley import PrescribedFlowNode

    radius, nu = 0.5e-3, 1e-6
    omega = (6.0 / radius) ** 2 * nu
    node = PrescribedFlowNode("flow", 1e-3, viscosity=1e-3, density=1e3, tube_centre=(0.0, 0.0, 0.0),
                              tube_axis=(0.0, 0.0, 1.0), tube_radius=radius)
    p = node._step_constants(None)  # noqa: SLF001
    w = waveform((0.0, 0.0, 1.0), jax.numpy.asarray(1.0), 1.0, omega / (2 * math.pi), 0.0)
    state = node.advance(p, dict(node.initial_state()), {"waveform": w}, node.delta_t)
    at = jax.numpy.stack([jax.numpy.linspace(0.0, radius, points), jax.numpy.zeros(points),
                          jax.numpy.zeros(points)], axis=-1)
    read = jax.jit(lambda s, x: node.velocity_at(p, s, x))
    jax.block_until_ready(read(state, at))
    began = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(read(state, at))
    return {"points": points, "s_per_reading": (time.perf_counter() - began) / repeats}


def auto_build() -> dict[str, Any]:
    from mime.dynamics import FVM
    from mime.experiment import Experiment
    from mime.interactions import BodyForce
    from mime.scene import PeriodicBox, Scene, Tube, Z
    from mime.things import Medium, Structure
    from mime.units import Hz, kg_per_m3, m_per_s2, mm, mPa_s, ms

    def experiment(dt: Any) -> Any:
        exp = Experiment("tube", scene=Scene.new(up=Z, units="m"), dt=dt)
        exp.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                          length=mm(10.0))))
        exp.add_medium(Medium("water", fills="tube.lumen",
                              fields={"flow": FVM(domain=PeriodicBox(length=mm(0.25)), cells_across=16)}))
        exp.declare("water.viscosity", mPa_s(1.0), reason="water near 20 C")
        exp.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(2.0), amplitude=m_per_s2(1.0), frequency=Hz(50.0)),
                   medium="water")
        return exp

    timings = {}
    for label, dt in (("given", ms(0.5)), ("auto", "auto")):
        experiment(dt).build()
        began = time.perf_counter()
        experiment(dt).build()
        timings[label] = time.perf_counter() - began
    return {"build_s_given_dt": timings["given"], "build_s_auto": timings["auto"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--steps", type=int, default=50)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    rows = [measure(dh, arguments.steps, ledger=ledger) for dh in (8, 16, 32) for ledger in (False, True)]
    readings = [prescribed(n) for n in (100, 10_000)]
    print(json.dumps({"dtype": arguments.dtype, "devices": [str(d) for d in jax.devices()], "rk3": rows,
                      "prescribed": readings, "dt_auto": auto_build()}, indent=1))


if __name__ == "__main__":
    main()
