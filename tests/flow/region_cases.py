"""The region builder's two-outlet gate (F2b, PR 4a; scope item 0) on drawn tube trees, through the experiment, shared
by the CI tests and the detection-rate probes.

**The rig** (:func:`experiment`): a Y (``mime.scene.TubeTree``) of unequal daughters, drawn (:func:`draw`), turned as a
whole from z by up to 20 degrees about a drawn axis and moved by part of a cell (a vessel aligned with no grid axis and
on no symmetry plane), filled by water, its region the whole vessel (``VesselRegion(extension=...)``), stepped by the
steady Stokes limit; an ``Inflow`` at the root and one ``SharedDistal`` across the two ends (``surface=(end_0,
end_1)``), metered.

**The oracle** (:func:`network_split`): the 0-D network of the two branches in parallel from the junction, each its
daughter's Poiseuille resistance from the junction to its cut and on along its extension, bend and run to the face
(the region's own report, ``RegionOpening.series_length``), in series with the law's own resistance; the shared distal
resistance drops out of the split (both outlets see one distal pressure). The laws' resistances are drawn at 20 to 50
times the larger branch's and their ratio at 2 to 4, so the branches' Poiseuille estimate (the band's offset, the
junction) moves the split by under a tenth of a branch's share.

**The plants** (:data:`PLANTS`, switched by ``REGION_PLANT``): ``dropped``, the law's off-diagonals dropped (each
``R_d`` folded into its own row, its books consistent with it); ``swapped``, the two ends' patches exchanged after the
build, a mix-up between an opening's name and its patch (the twin rig's mirror could not see one)."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.dynamics import FVM, Scheme
from mime.experiment import Experiment
from mime.interactions import Inflow, SharedDistal
from mime.nodes.flow.fvm import _ports
from mime.observations import EnergyLedger
from mime.scene import Scene, TubeTree, VesselRegion, Z
from mime.testing import ledger_closes
from mime.things import Medium, Structure
from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_s, ms

SEED = 70_000
MU, RHO = 1e-3, 1e3
FLUX = 1e-9
PLANTS = ("dropped", "swapped")


@dataclass(frozen=True)
class YDraw:
    """One drawn Y: the parent's and daughters' radii (m), the daughters' angles from the parent (degrees) and lengths
    (m), the turn of the whole tree (its angle from z and the azimuth of its axis, degrees), its offset (cells), the
    cells across the narrowest lumen, the extension (diameters), and the law's two resistances and its distal one
    (Pa s/m^3)."""

    seed: int
    radii: tuple[float, float, float]
    angles: tuple[float, float]
    lengths: tuple[float, float]
    plane: float
    tilt: float
    tilt_axis: float
    offset: tuple[float, float, float]
    cells_across: int
    extension: float
    first: float
    second: float
    distal: float


def _branch(radius: float, length: float) -> float:
    return 8.0 * MU * length / (math.pi * radius ** 4)


def draw(seed: int) -> YDraw:
    rng = np.random.default_rng(SEED + seed)
    r0 = 0.5e-3
    radii = (r0, r0 * float(rng.uniform(0.7, 0.9)), r0 * float(rng.uniform(0.6, 0.85)))
    angles = (float(rng.uniform(20.0, 40.0)), float(rng.uniform(20.0, 40.0)))
    lengths = (float(rng.uniform(2.0e-3, 3.0e-3)), float(rng.uniform(1.6e-3, 2.6e-3)))
    larger = max(_branch(radii[1], lengths[0] + 4e-3), _branch(radii[2], lengths[1] + 4e-3))
    first = larger * float(rng.uniform(20.0, 50.0))
    second = first * float(rng.uniform(2.0, 4.0))
    if rng.uniform() < 0.5:
        first, second = second, first
    return YDraw(seed, radii, angles, lengths, float(rng.uniform(0.0, 360.0)), float(rng.uniform(0.0, 20.0)),
                 float(rng.uniform(0.0, 360.0)), tuple(float(v) for v in rng.uniform(-0.5, 0.5, 3)),
                 int(rng.choice([6, 8])), float(rng.uniform(1.0, 2.0)), first, second,
                 first * float(rng.uniform(1.0, 3.0)))


def _turn(angle: float, about: np.ndarray) -> np.ndarray:
    k = about / np.linalg.norm(about)
    cross = np.asarray([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def tree_of(d: YDraw) -> TubeTree:
    """The drawn Y: the parent 2 mm along z to the junction at the origin, the daughters in the plane at azimuth
    ``plane``, on either side, then the whole turned and moved by part of a cell."""
    plane = math.radians(d.plane)
    across = np.asarray([math.cos(plane), math.sin(plane), 0.0])
    ends = []
    for side, angle, length in ((1.0, d.angles[0], d.lengths[0]), (-1.0, d.angles[1], d.lengths[1])):
        a = math.radians(angle)
        ends.append(length * (math.cos(a) * np.asarray([0.0, 0.0, 1.0]) + side * math.sin(a) * across))
    points = np.asarray([[0.0, 0.0, -2.0e-3], [0.0, 0.0, 0.0], ends[0], ends[1]])
    axis = math.radians(d.tilt_axis)
    turn = _turn(math.radians(d.tilt), np.asarray([math.cos(axis), math.sin(axis), 0.0]))
    spacing = 2.0 * min(d.radii) / d.cells_across
    radii = np.asarray([[d.radii[0]] * 2, [d.radii[1]] * 2, [d.radii[2]] * 2])
    return TubeTree.si(points, ((0, 1), (1, 2), (1, 3)), radii).turned(turn, np.asarray(d.offset) * spacing)


def experiment(d: YDraw, *, distal_scale: float = 1.0) -> Experiment:
    exp = Experiment("region_y", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    exp.add_structure(Structure("vessel", geometry=tree_of(d)))
    flow = FVM(domain=VesselRegion(extension=d.extension), cells_across=d.cells_across, stokes=True,
               scheme=Scheme.Steady())
    exp.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(MU), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(RHO), reason="water near 20 C")
    exp.attach(Inflow(steady=m3_per_s(FLUX)), medium="water", surface="vessel.root")
    exp.attach(SharedDistal(first=Pa_s_per_m3(d.first), second=Pa_s_per_m3(d.second),
                            distal=Pa_s_per_m3(d.distal * distal_scale)),
               medium="water", surface=("vessel.end_0", "vessel.end_1"))
    exp.observe(EnergyLedger())
    return exp


def network_split(d: YDraw) -> float:
    """The 0-D network's share of the inflow leaving through ``end_0``: each branch from the junction to its face
    (the daughter's length to its cut and the region's series length beyond it) in series with its law's resistance."""
    branches = []
    for (radius, length), series in zip(((d.radii[1], d.lengths[0]), (d.radii[2], d.lengths[1])),
                                        series_lengths(d), strict=True):
        branches.append(_branch(radius, length + series))
    zero, one = branches[0] + d.first, branches[1] + d.second
    return one / (zero + one)


def series_lengths(d: YDraw) -> tuple[float, float]:
    """Each daughter's length from its cut to its face, as the region builder lays it."""
    from mime.dynamics.flow._fvm import vessel_of
    from mime.registry import Region
    from mime.scene._region import build_region

    tree = tree_of(d)
    parameters = tree.parameters()
    region = Region("vessel.lumen", "tube_tree", {"points": tuple(np.asarray(parameters["points"]).ravel()),
                                                 "radii": tuple(np.asarray(parameters["radii"]).ravel()),
                                                 "segments": tuple(float(i) for pair in tree.segments for i in pair),
                                                 "fillet": float(tree.fillet)},
                    ("vessel.root", "vessel.end_0", "vessel.end_1"))
    built = build_region(vessel_of(region), around=None, reach=None, extension=d.extension, bend=2.0)
    by_name = {o.name: o.series_length for o in built.openings}
    return by_name["end_0"], by_name["end_1"]


@contextmanager
def planted(plant: str | None) -> Iterator[None]:
    """``dropped``: the shared distal law's block made diagonal (each ``R_d`` in its own row), its books with it."""
    if plant != "dropped":
        yield
        return
    law = _ports.SharedDistal
    affine, books = law.affine, law.books

    def diagonal(self: Any, parameters: Any, state: Any, time: Any, dt: Any) -> Any:
        import jax.numpy as jnp

        distal = parameters[2]
        return jnp.zeros(2, dtype=parameters.dtype), jnp.diag(jnp.stack([parameters[0] + distal, parameters[1] + distal]))

    def own(self: Any, parameters: Any, state: Any, ended: Any, flux: Any, pressure: Any, dt: Any) -> Any:
        zero = flux[0] * 0.0
        loss = dt * ((parameters[0] + parameters[2]) * flux[0] ** 2 + (parameters[1] + parameters[2]) * flux[1] ** 2)
        return _ports.LawBooks(zero, zero, loss, zero)

    law.affine, law.books = diagonal, own  # type: ignore[method-assign]
    try:
        yield
    finally:
        law.affine, law.books = affine, books  # type: ignore[method-assign]


def split_reading(d: YDraw, plant: str | None = None, *, distal_scale: float = 1.0) -> dict[str, float | bool]:
    """The run's share of the inflow through ``end_0`` against the network's, whether its books closed, and the
    port system's flux at the root."""
    with planted(plant):
        built = experiment(d, distal_scale=distal_scale).build()
        node = built.graph.get_node("water.flow")
        if plant == "swapped":
            centres = list(node.params["opening_centres"])
            centres[1], centres[2] = centres[2], centres[1]
            node.params["opening_centres"] = tuple(centres)
        result = built.run(ms(2.0))
    flux = np.asarray(result.final["water.flow"]["opening_flux"], dtype=np.float64)
    split = float(flux[1] / -flux[0])
    network = network_split(d)
    return {"split": split, "network": network, "error": split - network, "closes": bool(ledger_closes(result).passed),
            "root": float(-flux[0] / FLUX),
            "backflow_flagged": float(np.asarray(result.final["water.flow"]["backflow_flagged"])) > 0.0}


#: FB8's vessel: a straight vessel along z of diameter 1 mm, its root 2 D below where it narrows, narrowing over 0.5 D
#: to its throat and widening back over 0.25 D, its end 0.5 D after the throat.
D = 1.0e-3


@dataclass(frozen=True)
class StenosisDraw:
    """One drawn stenosed vessel for the outlet watch: the throat's diameter as a share of the vessel's, the Reynolds
    number by the vessel's diameter and mean speed, the offset (cells)."""

    seed: int
    throat: float
    reynolds: float
    offset: tuple[float, float, float]


def stenosis_draw(seed: int) -> StenosisDraw:
    rng = np.random.default_rng(SEED + 500 + seed)
    return StenosisDraw(seed, float(rng.uniform(0.55, 0.65)), float(rng.uniform(80.0, 120.0)),
                        tuple(float(v) for v in rng.uniform(-0.5, 0.5, 3)))


def stenosis_tree(d: StenosisDraw, cells_across: int = 8) -> TubeTree:
    r, throat = D / 2.0, d.throat * D / 2.0
    points = np.asarray([[0.0, 0.0, -2.0 * D], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5 * D], [0.0, 0.0, 0.75 * D],
                         [0.0, 0.0, 1.0 * D]]) + np.asarray(d.offset) * D / cells_across
    radii = np.asarray([[r, r], [r, throat], [throat, r], [r, r]])
    return TubeTree.si(points, ((0, 1), (1, 2), (2, 3), (3, 4)), radii)


#: The arms of FB8: the plane 0.5 D after the throat (near: the control, a plane in the recirculation), moved on by an
#: extension of 8 D (far), and a reversing inflow at low Womersley number through the far plane (reversing, Stokes).
WATCH_ARMS = ("near", "far", "reversing")


def watch_experiment(d: StenosisDraw, arm: str, cells_across: int = 8) -> tuple[Experiment, int, float]:
    """The experiment of an arm, its step count and its step: IMEX Euler, convection on at the draw's Re for ``near``
    and ``far`` (the step at 0.9 of the CFL line along the diagonal at twice the mean speed through the throat, run for
    1.2 times the region's length over the mean speed); for ``reversing``, Stokes, an inflow ``Q_m (1 + 1.5 sin wt)``
    at alpha 0.5, 40 steps a period for two periods."""
    from mime.units import Hz, s

    nu = 1e-6
    area = math.pi * (D / 2.0) ** 2
    extension = 0.0 if arm == "near" else 8.0
    length = (3.0 + extension) * D
    h = D / cells_across
    if arm == "reversing":
        omega = 0.25 * nu / (D / 2.0) ** 2
        period = 2.0 * math.pi / omega
        dt, steps, mean = period / 40.0, 80, 0.5 * nu / D * area
        inflow = Inflow(steady=m3_per_s(mean), amplitude=m3_per_s(1.5 * mean), frequency=Hz(1.0 / period))
        flow = FVM(domain=VesselRegion(extension=extension), cells_across=cells_across, stokes=True,
                   scheme=Scheme.IMEXEuler())
    else:
        speed = d.reynolds * nu / D
        peak = 2.0 * speed / d.throat ** 2
        dt = 0.9 * (0.4 / 3.0) * h / peak
        steps = math.ceil(1.2 * length / speed / dt)
        inflow = Inflow(steady=m3_per_s(speed * area))
        flow = FVM(domain=VesselRegion(extension=extension), cells_across=cells_across, scheme=Scheme.IMEXEuler())
    exp = Experiment(f"watch_{arm}", scene=Scene.new(up=Z, units="m"), dt=s(dt))
    exp.add_structure(Structure("vessel", geometry=stenosis_tree(d, cells_across)))
    exp.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", Pa_s(nu * RHO), reason="water near 20 C")
    exp.declare("water.density", kg_per_m3(RHO), reason="water near 20 C")
    exp.attach(inflow, medium="water", surface="vessel.root")
    from mime.interactions import Windkessel
    from mime.units import m3_per_Pa

    resistance = 8.0 * nu * RHO * length / (math.pi * (D / 2.0) ** 4)
    exp.attach(Windkessel(proximal=Pa_s_per_m3(resistance), compliance=m3_per_Pa(1e-16), distal=Pa_s_per_m3(resistance)),
               medium="water", surface="vessel.end_0")
    exp.observe(EnergyLedger())
    return exp, steps, dt


def watch_reading(d: StenosisDraw, arm: str) -> dict[str, float | bool]:
    """The watch's reading over the run's second half, read at the ends of chunks of 20 steps (every step for the
    reversing arm): the largest reversed share at the outlet, the samples it flagged, the samples with the net flux
    outward; the backflow term's work over the samples as a share of the outlet law's (``|p q|`` at the same samples);
    whether the books closed and the term never gave energy."""
    from mime.control import Chunk
    from mime.units import s

    exp, steps, dt = watch_experiment(d, arm)
    every = 1 if arm == "reversing" else 20
    samples: list[dict[str, float]] = []

    def kept(chunk: Chunk) -> None:
        held = chunk.state["water.flow"]
        if chunk.steps > steps // 2:
            samples.append({"reversed": float(np.asarray(held["opening_reversed"])[1]),
                            "flagged": float(np.asarray(held["outlet_reversed_flagged"])),
                            "flux": float(np.asarray(held["opening_flux"])[1]),
                            "port": abs(float(np.asarray(held["opening_pressure"])[1] * np.asarray(held["opening_flux"])[1])),
                            "backflow": abs(float(np.asarray(held["backflow_work"]))) / dt,
                            "gave": float(np.asarray(held["backflow_flagged"]))})

    built = exp.build()
    result = built.run(s(steps * dt), hook=kept, chunk=s(every * dt))
    port = sum(x["port"] for x in samples)
    return {"largest": max(x["reversed"] for x in samples), "flagged": int(sum(x["flagged"] for x in samples)),
            "outward": int(sum(x["flux"] > 0.0 for x in samples)), "samples": len(samples),
            "backflow_share": sum(x["backflow"] for x in samples) / max(port, 1e-300),
            "backflow_flagged": int(sum(x["gave"] for x in samples)), "closes": bool(ledger_closes(result).passed),
            "steps": steps}


#: The gate's line on ``|split - network|``: a tenth of the larger branch's share of its outlet's resistance.
LINE = 0.01


def gate_fails(reading: dict[str, float | bool]) -> bool:
    return abs(float(reading["error"])) > LINE or not bool(reading["closes"]) or bool(reading["backflow_flagged"])


def main() -> int:
    """The probe's oracle (``detection_rate.py --script``): one draw at ``DETECTION_SEED``, its plant from
    ``REGION_PLANT``; exits 1 when the gate fails."""
    import json

    seed = int(os.environ.get("DETECTION_SEED", "0"))
    if os.environ.get("WATCH"):
        # FB8: the far plane (clean) or, planted, the plane moved to 0.5 D after the throat; the watch firing fails.
        arm = os.environ.get("WATCH_PLANT") or os.environ["WATCH"]
        reading = watch_reading(stenosis_draw(seed), arm)
        print(json.dumps({"seed": seed, "arm": arm, **reading}))
        return 1 if int(reading["flagged"]) > 0 or int(reading["backflow_flagged"]) > 0 or not reading["closes"] else 0
    if os.environ.get("BACKFLOW_PLANT") == "flipped":
        from mime.nodes.flow.fvm import _open_node

        _open_node.BACKFLOW_BETA = -1.0
    plant = os.environ.get("REGION_PLANT") or None
    reading = split_reading(draw(seed), plant)
    print(json.dumps({"seed": seed, "plant": plant, **reading}))
    return 1 if gate_fails(reading) else 0


if __name__ == "__main__":
    raise SystemExit(main())


def estimate_y(seed: int = 0) -> Experiment:
    """The memory estimate's case of the two-outlet arms (``mem_estimate.py --case``): a draw at 8 cells across."""
    d = draw(seed)
    return experiment(YDraw(**{**d.__dict__, "cells_across": 8}))


def estimate_watch(arm: str = "far") -> Experiment:
    """The memory estimate's case of an outlet watch arm."""
    return watch_experiment(stenosis_draw(0), arm)[0]
