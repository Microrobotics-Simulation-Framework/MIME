"""The FVM's budget (F0's B10, Q-c): a report with no pass line. For the empty periodic tube at three sizes, IMEX Euler
with the Stokes approximation and with convection on: the time per cell per step, the CG iterations per step, the
device bytes per cell (the compiled step's memory analysis where the backend gives one), the compile time, and the
step's time with the energy ledger off and on.

Run it alone, on a quiet machine (its numbers are relative: a laptop's CPU is not a GPU), one dtype at a time:

    JAX_PLATFORMS=cpu python scripts/fvm_budget.py --dtype float64 > budget_float64.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax


def _node(dh: int, *, ledger: bool, convection: bool) -> Any:
    from mime.core import ledger as books
    from mime.nodes.flow.fvm import ImplicitFlowNode, periodic_tube_layout

    layout = periodic_tube_layout(0.5, 4.0 / dh, cells_across=dh)
    hosted = (
        books.LedgerDelivery("body_force", "drive:water.body_force", books.UNMETERED, books.HOSTED).encoded(),
        books.LedgerDelivery("viscous", "viscosity:water", books.DISSIPATIVE, books.HOSTED).encoded(),
        books.LedgerDelivery("penalty", "wall:vessel", books.DISSIPATIVE, books.HOSTED).encoded(),
    ) if ledger else ()
    return ImplicitFlowNode("flow", 0.002, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                            spacing=layout.spacing, viscosity=1.0, density=1.0, tube_centre=layout.centre,
                            tube_axis=layout.axis, tube_radius=layout.radius, convection=convection,
                            ledger=hosted)


def measure(dh: int, steps: int, *, ledger: bool, convection: bool) -> dict[str, Any]:
    node = _node(dh, ledger=ledger, convection=convection)
    p = node._step_constants(None)  # noqa: SLF001 - the step's own constants
    dt = node.delta_t

    drive = {"body_force": jax.numpy.asarray((0.0, 0.0, 8.0))}

    def body(state: Any, _: Any) -> tuple[Any, Any]:
        ended = node.advance(p, state, drive, dt)
        return ended, ended["solve_report"][0]

    run = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))
    start = dict(node.initial_state())
    began = time.perf_counter()
    lowered = run.lower(start)
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - began
    memory = compiled.memory_analysis()
    held = None if memory is None else int(getattr(memory, "temp_size_in_bytes", 0)
                                           + getattr(memory, "argument_size_in_bytes", 0)
                                           + getattr(memory, "output_size_in_bytes", 0))
    ended, iterations = compiled(start)
    jax.block_until_ready(ended)
    began = time.perf_counter()
    ended, iterations = compiled(start)
    jax.block_until_ready(ended)
    seconds = time.perf_counter() - began
    c = node.constants()
    cells = c["cells_x"] * c["cells_y"] * c["cells_z"]
    return {"dh": dh, "cells": cells, "steps": steps, "ledger": ledger, "convection": convection,
            "compile_s": compile_seconds, "run_s": seconds, "s_per_cell_step": seconds / (cells * steps),
            "iterations_per_step": float(iterations.mean()), "bytes_per_cell": None if held is None else held / cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--steps", type=int, default=50)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    rows = [measure(dh, arguments.steps, ledger=ledger, convection=convection)
            for dh in (8, 16, 32) for convection in (False, True) for ledger in (False, True)]
    print(json.dumps({"dtype": arguments.dtype, "devices": [str(d) for d in jax.devices()], "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
