"""R-15's gates on decision (A) in product code (the Inflow's opening prescribed), F2b PR 4a: the open-ports spike's
FB7 case on the open node, stepped by IMEX Euler at 0.9 of its own CFL line along the tube, with the backflow term in
its lagged implicit form (the head's), its explicit form (a test-only patch: the term moved out of the operator into the
drive at the step's start velocity) or removed; and a genuine reversal's damping. Predictions:
tests/flow/predictions_prescribed_inlet.json.

The case (the spike's question 5, in SI): a tube of D = 1 mm along z, 4 D long (the open node's own box, the build's
margin), a cosine stenosis 1 D long halving the radius centred 0.9 D before the outlet (the spike's); water; the
Inflow ``Q_m (1 + 1.5 sin wt)`` at alpha 5.5 (the whole flow reversing to -0.5 Q_m); an RCR outlet (1e8, 1e-11, 1e9);
D/h 16; dt = T / ceil(T / (0.9 (0.4) h / (1.25 s U_m))), ``s`` 25 (``LARGEST``). One JSON line a chunk of steps, then a summary; a wall budget ends a run short of a runner's limit.

``python probe/fb7node.py --reynolds 300 --periods 2.25 --form implicit --output probe/out/fb7.jsonl``"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mime.nodes.flow.fvm import OpenFlowNode, periodic_tube_layout  # noqa: E402

D, NU, RHO = 1.0e-3, 1.0e-6, 1.0e3
ALPHA = 5.5
CELLS = 16
#: The largest speed over the mean speed the step is sized from: the first probe's runs at the spike's 7.45 and 8.1
#: stopped at the CFL check by 0.14 and 0.23 of a period with the stenosis's jet at 10.5 and 11.3 U_m and rising
#: (run 43252920973; the throat's mean at the inflow's peak is 10 U_m), so 25 for both points.
LARGEST = {300.0: 25.0, 100.0: 25.0}
LINE = 0.4
BEFORE = 0.9


@dataclass(frozen=True)
class LocalStenosis:
    """A cosine narrowing ``R (1 - s (1 + cos(2 pi (z - z_c) / l)) / 2)`` over ``|z - z_c| < l / 2``, the full radius
    elsewhere, with the first-order signed distance (positive in the solid)."""

    centre_x: float
    centre_y: float
    radius: float
    at: float
    length: float
    narrowing: float = 0.5
    region: str = "wall"

    def signed_distance(self, points: Any) -> Any:
        r = jnp.hypot(points[..., 0] - self.centre_x, points[..., 1] - self.centre_y)
        angle = 2.0 * jnp.pi * (points[..., 2] - self.at) / self.length
        inside = jnp.abs(points[..., 2] - self.at) < 0.5 * self.length
        local = self.radius * (1.0 - jnp.where(inside, self.narrowing * (1.0 + jnp.cos(angle)) / 2.0, 0.0))
        slope = jnp.where(inside, self.radius * self.narrowing * jnp.pi / self.length * jnp.sin(angle), 0.0)
        return (r - local) / jnp.sqrt(1.0 + slope ** 2)


class StenosedNode(OpenFlowNode):
    def solids(self, p: Any, grid: Any) -> Any:
        c = self.constants()
        length = grid.cells[2] * grid.spacing
        return [LocalStenosis(float(c["tube_centre"][0]), float(c["tube_centre"][1]), float(c["tube_radius"]),
                              length - BEFORE * D, D)]


def node_for(reynolds: float, form: str) -> tuple[OpenFlowNode, float, int, float]:
    r = D / 2.0
    mean = reynolds * NU / r
    flux = mean * math.pi * r ** 2
    omega = ALPHA ** 2 * NU / r ** 2
    period = 2.0 * math.pi / omega
    layout = periodic_tube_layout(r, 4.0 * D, cells_across=CELLS, offset_cells=(0.31, -0.17))
    h = layout.spacing
    per_period = math.ceil(period / (0.9 * LINE * h / (1.25 * LARGEST[float(reynolds)] * mean)))
    dt = period / per_period
    node = StenosedNode("water.flow", dt, cells_x=layout.cells[0], cells_y=layout.cells[1], cells_z=layout.cells[2],
                        spacing=h, viscosity=NU * RHO, density=RHO, tube_centre=layout.centre,
                        tube_axis=(0.0, 0.0, 1.0), tube_radius=r, openings=("tube.start", "tube.end"),
                        opening_laws=("inflow", "windkessel"),
                        opening_parameters=[[flux, 1.5 * flux, 1.0 / period, -math.pi / 2.0], [1e8, 1e-11, 1e9, 0.0]],
                        backflow_openings=() if form == "off" else (1,), inlet_reach=4.0 * D)
    return node, mean, per_period, dt


def explicit() -> None:
    """The explicit form: the term ``-c(u^n) w^n`` a force in the drive at the step's start velocity, the operator
    without it (the base's form before the lagged implicit one)."""
    node_class = OpenFlowNode
    coefficient, stepped, forcing = node_class.backflow_coefficient, node_class.stepped, node_class.forcing
    held: dict[str, Any] = {}

    def stepped_held(self, p, state, boundary_inputs, dt):
        held["velocity"] = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        held["p"] = p
        return stepped(self, p, state, boundary_inputs, dt)

    def forcing_with(self, fields, boundary_inputs, scheme, dtype):
        drive = forcing(self, fields, boundary_inputs, scheme, dtype)
        velocity = held["velocity"]
        c = coefficient(self, self.open_grid(held["p"]), fields, velocity)
        if c is None:
            return drive
        extra = -c * velocity[2]

        def forced(stage, at):
            d = drive(stage, at)
            return (d[0], d[1], d[2] + extra)

        return forced

    node_class.stepped = stepped_held  # type: ignore[method-assign]
    node_class.forcing = forcing_with  # type: ignore[method-assign]
    node_class.backflow_coefficient = lambda self, grid, fields, velocity: None  # type: ignore[method-assign]


def readings(node: OpenFlowNode, p: Any, mean: float):
    """Per step (in the scan): the outlet's largest-to-date inward flux through its patch (fluid-weighted), the
    watch, the term's work, the largest speed, the inlet's pressure, and the prescribed profile's convective readings
    (the half cells' power over the net kinetic flux in; the scheme's kinetic flux through the plane against the face
    formula)."""
    grid = node.open_grid(p)
    fields = node.fields(p, grid)
    fluid_top = 1.0 - fields.chi[2][..., -1]

    def read(state: Any) -> Any:
        u = tuple(state[f"velocity_{x}"] for x in "xyz")
        top = u[2][..., -1]
        inward = grid.spacing ** 2 * jnp.sum(fluid_top * jnp.maximum(-top, 0.0))
        conv = grid.convection(u)
        full = grid.inner(conv, u)
        face = 0.5 * grid.cell_volume * jnp.sum(conv[2][..., 0] * u[2][..., 0])
        kinetic, _ = grid.opening_kinetic_fluxes(u)
        formula = grid.spacing ** 2 * jnp.sum(0.5 * u[2][..., 0] ** 3)
        speed = jnp.max(jnp.stack([jnp.max(jnp.abs(a)) for a in u]))
        return jnp.stack([inward, state["opening_reversed"][1], state["backflow_work"], speed / mean,
                          state["opening_pressure"][0], state["opening_pressure"][1], -state["opening_flux"][0],
                          jnp.abs(face) / jnp.maximum(jnp.abs(full), 1e-300),
                          jnp.abs(-kinetic[0] - formula) / jnp.maximum(jnp.abs(formula), 1e-300),
                          state["inlet_development"], jnp.max(jnp.abs(u[2][..., 1:5])) / mean,
                          state["backflow_flagged"], state["cfl"]])

    return read


NAMES = ("outlet_inward_flux", "watch", "backflow_work", "speed_over_mean", "inlet_pressure", "outlet_pressure",
         "inflow", "residual", "books_vs_formula", "inlet_development", "inlet_w_over_mean", "backflow_flagged", "cfl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--periods", type=float, required=True)
    parser.add_argument("--form", choices=("off", "implicit", "explicit"), required=True)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--budget", type=float, default=19_500.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.form == "explicit":
        explicit()
    start = time.time()
    node, mean, per_period, dt = node_for(args.reynolds, args.form)
    p = node._step_constants(None)  # noqa: SLF001
    read = readings(node, p, mean)

    def body(s: Any, _: Any) -> Any:
        ended = node.advance(p, s, {}, dt)
        return ended, read(ended)

    stepper = jax.jit(lambda s: jax.lax.scan(body, s, None, length=args.chunk))
    steps = args.chunk * math.ceil(args.periods * per_period / args.chunk)
    state = dict(node.initial_state())
    out = open(args.output, "w")
    rows: list[np.ndarray] = []
    reached, stopped, ended = 0, None, "completed"
    while reached < steps:
        if time.time() - start > args.budget:
            ended = "budget"
            break
        state, chunk_rows = stepper(state)
        chunk_rows = np.asarray(chunk_rows, dtype=np.float64)
        stop = int(state["stopped_at"])
        finite = bool(np.all(np.isfinite(chunk_rows[:, :7])))
        if not finite or stop >= 0:
            good = chunk_rows[np.all(np.isfinite(chunk_rows[:, :7]), axis=1)]
            rows.append(good)
            reached = (stop if stop >= 0 else reached + len(good))
            stopped = "cfl" if stop >= 0 else "not finite"
            ended = stopped
            break
        rows.append(chunk_rows)
        reached += args.chunk
        last = {name: float(chunk_rows[-1, k]) for k, name in enumerate(NAMES)}
        out.write(json.dumps({"step": reached, "period": reached / per_period, **last,
                              "seconds": time.time() - start}) + "\n")
        out.flush()
    held = np.concatenate(rows) if rows else np.zeros((0, len(NAMES)))
    second = held[int(per_period):] if len(held) > per_period else held[: 0]
    summary = {"summary": True, "form": args.form, "dtype": "float64" if jax.config.jax_enable_x64 else "float32",
               "reynolds_radius": args.reynolds, "alpha": ALPHA, "cells_across": CELLS, "dt": dt,
               "steps_per_period": per_period, "steps_asked": steps, "steps_reached": reached,
               "periods_reached": reached / per_period, "ended": ended,
               "box": [int(node.constants()[f"cells_{x}"]) for x in "xyz"],
               "tolerance": float(node.constants()["tolerance"]),
               "largest_speed_over_mean": float(np.max(held[:, 3])) if len(held) else None,
               "largest_inward_flux_second_period": float(np.max(second[:, 0])) if len(second) else None,
               "largest_inward_flux": float(np.max(held[:, 0])) if len(held) else None,
               "watch_largest": float(np.max(held[:, 1])) if len(held) else None,
               "backflow_work_sum": float(np.sum(held[:, 2])) if len(held) else None,
               "backflow_flagged_steps": int(np.sum(held[:, 11])) if len(held) else None,
               "residual_median": float(np.median(held[50:, 7])) if len(held) > 50 else None,
               "residual_largest_after_start": float(np.max(held[50:, 7])) if len(held) > 50 else None,
               "books_vs_formula_median": float(np.median(held[50:, 8])) if len(held) > 50 else None,
               "inlet_development_largest": float(np.max(held[:, 9])) if len(held) else None,
               "inlet_w_over_mean_largest": float(np.max(held[:, 10])) if len(held) else None,
               "cfl_largest": float(np.max(held[:, 12])) if len(held) else None,
               "seconds": time.time() - start}
    np.save(args.output.replace(".jsonl", "_series.npy"), held[:: max(1, len(held) // 4000)])
    out.write(json.dumps(summary) + "\n")
    out.close()
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
