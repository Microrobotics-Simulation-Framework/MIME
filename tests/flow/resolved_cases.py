"""Shared set-ups for F4's resolved bodies (``Drag.Penalty``): a sphere held in the flow node's steady step on a tube's
axis and its resistance columns (flux-free, with the drive's displaced share), R3's case at Re 76 through the
experiment, and a neutral sphere in a free box stepped with its rigid body by hand. Each plant is switched by an
environment variable (``MIME_PLANT_<NAME>``), for ``.github/probe/detection_rate.py``."""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mime.core import run_dtype
from mime.dynamics import FVM, RigidInertial, Scheme
from mime.experiment import Experiment
from mime.interactions import BodyForce, Drag, Inflow, Windkessel
from mime.nodes.flow.fvm import band_named, periodic_tube_layout
from mime.nodes.flow.fvm import _resolved
from mime.nodes.flow.fvm._resolved import window_cells
from mime.nodes.flow.fvm._resolved_node import ResolvedFlowNode, ResolvedSpec
from mime.scene import PeriodicBox, Scene, Sphere, Tube, TubeTree, VesselRegion, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Pa_s_per_m3, kg, kg_m2, kg_per_m3, m, m3_per_Pa, m_per_s, m_per_s2, mL_per_s, mPa_s, \
    rad_per_s, s

WATER_MU, WATER_RHO = 1e-3, 1e3
BAND = band_named("smooth_2h_0.3h")


def plant(name: str) -> str:
    """The plant switched on for this run (``MIME_PLANT_<NAME>``'s value), or an empty string."""
    return os.environ.get(f"MIME_PLANT_{name}", "")


def seed() -> int:
    return int(os.environ.get("DETECTION_SEED", "0"))


def haberman_sayre(ratio: float) -> float:
    """Haberman and Sayre's (1958) wall factor of a sphere translating along a tube's axis (``tests/tube_cases.py``'s
    form)."""
    return (1 - 0.75857 * ratio ** 5) / (1 - 2.1050 * ratio + 2.0865 * ratio ** 3 - 1.7068 * ratio ** 5
                                          + 0.72603 * ratio ** 6)


def sphere_spec(radius: float, spacing: float, law: str = "drag:ball") -> ResolvedSpec:
    return ResolvedSpec("ball", law, "sphere", {"centre": (0.0, 0.0, 0.0), "radius": radius}, (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), window_cells(radius, spacing, BAND),
                        4.0 / 3.0 * math.pi * radius ** 3)


class HeldSphere:
    """A sphere of ``radius`` held on the axis of a periodic tube of ``radius / ratio``, ``cells`` across its radius,
    two diameters long, in the flow node's steady step (Stokes): its resistance's columns flux-free (the unit drive's
    field superposed so the axial flux is zero, its displaced share on the body), ``offset`` its centre off the cell's
    corner (in cells)."""

    def __init__(self, ratio: float, cells: float, offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 radius: float = 0.5e-3, lateral: float = 0.0) -> None:
        self.radius, self.ratio = radius, ratio
        tube = radius / ratio
        self.layout = periodic_tube_layout(tube, 4.0 * tube, spacing=radius / cells)
        h = self.layout.spacing
        self.node = ResolvedFlowNode(
            "flow", 1.0, cells_x=self.layout.cells[0], cells_y=self.layout.cells[1], cells_z=self.layout.cells[2],
            spacing=h, viscosity=WATER_MU, density=WATER_RHO, tube_centre=self.layout.centre,
            tube_axis=self.layout.axis, tube_radius=self.layout.radius, scheme="steady", convection=False,
            bodies=[sphere_spec(radius, h)])
        self.centre = np.asarray((self.layout.centre[0] + lateral + offset[0] * h, self.layout.centre[1] + offset[1] * h,
                                  0.5 * self.layout.cells[2] * h + offset[2] * h))
        self.step = jax.jit(lambda pose, twist, drive: self.node.update(
            dict(self.node.initial_state()), {"body_0_pose": pose, "body_0_twist": twist,
                                              "body_force": jnp.stack([0.0 * drive, 0.0 * drive, drive])}, 1.0))

    def pose(self, centre: Any = None) -> Any:
        held = self.centre if centre is None else centre
        return jnp.concatenate([jnp.asarray(held, dtype=run_dtype()), jnp.asarray((1.0, 0.0, 0.0, 0.0),
                                                                                 dtype=run_dtype())])

    def flux(self, state: Any) -> Any:
        h = self.layout.spacing
        return jnp.sum(state["velocity_z"]) * h ** 3 / (self.layout.cells[2] * h)

    def column(self, mode: int, centre: Any = None) -> Any:
        """The resistance's column for the unit rigid mode ``mode`` (0 to 5: translations, then rotations about the
        centre), flux-free, the share included: ``-(load)``."""
        dtype = run_dtype()
        twist = jnp.zeros(6, dtype=dtype).at[mode].set(1.0)
        moved = self.step(self.pose(centre), twist, jnp.asarray(0.0, dtype=dtype))
        driven = self.step(self.pose(centre), jnp.zeros(6, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
        alpha = self.flux(moved) / self.flux(driven)
        load = moved["body_0_load"] - alpha * driven["body_0_load"]
        share = moved["body_0_share"] - alpha * driven["body_0_share"]
        return -(load + share)

    def at_rest(self, drive: Any) -> tuple[Any, Any]:
        """The held body's steady load and its displaced share at rest under the drive ``drive`` (m/s^2, a vector)."""
        dtype = run_dtype()
        state = jax.jit(lambda pose, force: self.node.update(
            dict(self.node.initial_state()), {"body_0_pose": pose, "body_0_twist": jnp.zeros(6, dtype=dtype),
                                              "body_force": force}, 1.0))(self.pose(), jnp.asarray(drive, dtype=dtype))
        return state["body_0_load"], state["body_0_share"]

    def k(self) -> float:
        """The axial drag over Stokes's, ``R_zz / (6 pi mu a)``."""
        return float(self.column(2)[2]) / (6.0 * math.pi * WATER_MU * self.radius)


def effective_ratio(k: float) -> float:
    """The radius ratio at which Haberman and Sayre's factor is ``k``."""
    low, high = 1e-4, 0.9
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if haberman_sayre(middle) < k else (low, middle)
    return 0.5 * (low + high)


def shift_cells(k: float, ratio: float, cells: float) -> float:
    """The radius by which the sphere reads smaller, in cells, from its drag: ``F = 6 pi mu a' K(a'/R)`` solved for
    ``a'`` (the effective radius in the drag's prefactor and in the wall factor alike)."""
    target = k * ratio
    low, high = 1e-4, ratio * 1.5
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if middle * haberman_sayre(middle) < target else (low, middle)
    return (ratio - 0.5 * (low + high)) / ratio * cells


@contextlib.contextmanager
def patched(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    """``owner.name`` replaced for a block: a plant (the node's step is traced afresh in each new node)."""
    held = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, held)


def deeper_band(scale: float) -> Callable[..., Any]:
    """The body's lift with its band's penalty depth times ``scale`` (the band plant)."""
    original = _resolved.body_faces

    def faces(grid: Any, component: int, body: Any, position: Any, turn: Any, band: Any, interior: float) -> Any:
        import dataclasses
        return original(grid, component, body, position, turn, dataclasses.replace(band, depth=band.depth * scale),
                        interior)
    return faces


def r3_experiment(ratio: float, *, dt: float = 1.5e-4, feature_cells: int = 3, hull: Any = None,
                  one_way: bool | None = None, spacing: Any = None, scheme: Any = None,
                  length: Any = None, opened: bool = False, tree: bool = False) -> Experiment:
    """The coupling spike's R3 case at Re 76: a sphere of radius 0.1875 D (3 cells a radius at D/h 16) free on a
    periodic tube's axis (D = 1 mm, 4 D long) in water, the fluid driven by the uniform force that gives the empty
    tube a mean speed of 76 nu / D; the body's density ``ratio`` times the fluid's; from rest. ``tree``: the same
    straight lumen as a tube tree's (two segments reaching a diameter past each end of the period)."""
    diameter = 1e-3
    nu = WATER_MU / WATER_RHO
    mean = 76.0 * nu / diameter
    drive = 8.0 * nu * mean / (diameter / 2.0) ** 2
    radius = 0.1875 * diameter
    exp = Experiment("resolved_r3", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    if tree:
        exp.add_structure(Structure("tube", geometry=TubeTree(
            points=m((0.0, 0.0, -diameter, 0.0, 0.0, 2.0 * diameter, 0.0, 0.0, 5.0 * diameter)),
            segments=((0, 1), (1, 2)), radii=m((diameter / 2.0,) * 4))))
    else:
        exp.add_structure(Structure("tube", geometry=Tube(axis=Z, inner_radius=m(diameter / 2.0), wall=m(2e-4),
                                                          length=m(0.1))))
    domain = VesselRegion(margin=4) if opened else PeriodicBox(length=m(4.0 * diameter) if length is None else length)
    flow = FVM(domain=domain, cells_across=16,
               scheme=Scheme.IMEXEuler() if scheme is None else scheme,
               **({"spacing": spacing} if spacing is not None else {}))
    exp.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    exp.declare("water.viscosity", mPa_s(WATER_MU * 1e3), reason="water")
    exp.declare("water.density", kg_per_m3(WATER_RHO), reason="water")
    mass = ratio * WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    exp.add_body(Body("ball", dynamics=RigidInertial(),
                      parts={"shell": Part(Sphere(radius=m(radius)) if hull is None else hull)},
                      initial=State(position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("ball.mass", kg(mass), reason="the density ratio the gate holds")
    exp.declare("ball.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="a uniform sphere")
    if opened:
        exp.attach(Inflow(steady=mL_per_s(1e-3)), medium="water", surface="tube.start")
        exp.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
                   medium="water", surface="tube.end")
    else:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(drive)), medium="water")
    exp.attach(Drag.Penalty(feature_cells=feature_cells, **({"one_way": one_way} if one_way is not None else {})),
               body="ball", medium="water")
    return exp


def r3_history(ratio: float, steps: int = 200, dt: float = 1.5e-4) -> tuple[np.ndarray, float]:
    """R3's case run ``steps`` steps: the body's axial velocity after each, and the flow's largest speed at the end;
    a run that goes non-finite ends there (its velocities after it NaN)."""
    built = r3_experiment(ratio, dt=dt).build()
    held: list[float] = []

    def hook(chunk: Any) -> None:
        held.append(float(np.asarray(chunk.state["ball"]["twist"])[2]))

    try:
        result = built.run(s(steps * dt), hook=hook, chunk=s(dt))
        top = max(float(np.max(np.abs(np.asarray(result.final["water.flow"][f"velocity_{c}"])))) for c in "xyz")
    except Exception:  # noqa: BLE001 - a run that diverged is refused as not finite: the gate's control reads it
        top = float("nan")
    velocities = np.full(steps, np.nan)
    velocities[:len(held)] = held
    return velocities, top


def stable(velocities: np.ndarray, top: float, run: int = 4) -> tuple[bool, int]:
    """Whether a velocity history is stable: finite, under twice the flow's largest speed, and with no ``run`` steps
    in a row whose step-to-step changes alternate in sign (a lagged loop that rings or diverges flips the change every
    step; the moving body's ripple, a dip as it crosses a cell, flips it once and back); and its longest run."""
    if not np.all(np.isfinite(velocities)) or not math.isfinite(top):
        return False, -1
    change = np.diff(velocities)
    flips = (change[1:] * change[:-1]) < 0.0
    longest, current = 0, 0
    for flipped in flips:
        current = current + 1 if flipped else 0
        longest = max(longest, current)
    return bool(np.max(np.abs(velocities)) < 2.0 * top and longest < run - 1), longest


def rigid_interior(ratio: float, original: Callable[..., Any]) -> Callable[..., Any]:
    """The rigid-interior shortcut (the plant), R3's: the interior taken to move with the body, so its momentum change
    cancels the body's displaced mass and the body's equation is ``(m_s - m_f) dV/dt = F``, the penalty's load alone:
    delivered to a body of mass ``m_s`` as ``m_s / (m_s - m_f) F``, ``ratio`` the density ratio."""
    def stepped(self: Any, p: Any, state: Any, inputs: Any, dt: float) -> Any:
        held, parts = original(self, p, state, inputs, dt)
        return {**held, "body_0_load": ratio / (ratio - 1.0) * held["body_0_penalty"]}, parts
    return stepped


class FreeBox:
    """A sphere of ``radius_cells`` in a free periodic box of ``cells`` a side (the tube outside it), driven by a
    uniform body force ``g`` along z, its rigid body stepped by hand in the graph's order (the flow, then the body),
    the body taking the flow's load and, unless ``share`` is off, its displaced share."""

    def __init__(self, cells: int = 24, radius_cells: float = 3.0, ratio: float = 1.0, g: float = 1.0,
                 share: bool = True) -> None:
        h = 1e-5
        side = cells * h
        self.dt = 0.1 * h * h * WATER_RHO / WATER_MU
        self.radius = radius_cells * h
        spec = sphere_spec(self.radius, h)
        self.node = ResolvedFlowNode("flow", self.dt, cells_x=cells, cells_y=cells, cells_z=cells, spacing=h,
                                     viscosity=WATER_MU, density=WATER_RHO, tube_centre=(side / 2, side / 2, 0.0),
                                     tube_axis=(0.0, 0.0, 1.0), tube_radius=10.0 * side, scheme="imex_euler",
                                     convection=False, tolerance=1e-10 if run_dtype() == jnp.float64 else 1e-5,
                                     bodies=[spec])
        self.mass = ratio * WATER_RHO * spec.volume
        self.g, self.share = g, share
        self.position = jnp.asarray((side / 2 + 0.3 * h, side / 2 + 0.2 * h, side / 2 + 0.1 * h), dtype=run_dtype())
        self.velocity = jnp.zeros(3, dtype=run_dtype())
        self.state = dict(self.node.initial_state())
        dtype = run_dtype()
        self._step = jax.jit(lambda state, pose, twist: self.node.update(
            state, {"body_0_pose": pose, "body_0_twist": twist,
                    "body_force": jnp.asarray((0.0, 0.0, g), dtype=dtype)}, self.dt))

    def run(self, steps: int) -> list[float]:
        """``steps`` steps; the body's axial velocity after each."""
        held: list[float] = []
        dtype = run_dtype()
        for _ in range(steps):
            pose = jnp.concatenate([self.position, jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype)])
            twist = jnp.concatenate([self.velocity, jnp.zeros(3, dtype=dtype)])
            self.state = self._step(self.state, pose, twist)
            load = self.state["body_0_load"][:3] + (self.state["body_0_share"][:3] if self.share else 0.0)
            self.velocity = self.velocity + self.dt * load / self.mass
            self.position = self.position + self.dt * self.velocity
            held.append(float(self.velocity[2]))
        return held


__all__ = ["BAND", "WATER_MU", "WATER_RHO", "FreeBox", "HeldSphere", "deeper_band", "effective_ratio",
           "haberman_sayre", "patched", "plant", "r3_experiment", "r3_history", "rigid_interior", "seed",
           "shift_cells", "stable", "sphere_spec"]
