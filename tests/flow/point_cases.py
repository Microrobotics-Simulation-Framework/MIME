"""Shared set-ups for F3's point bodies: the exchange law's draws, a free box stepped through the flow node and the point
body's node with the mapping kind's own transfers, and experiments of a point sphere in a driven tube. Each plant is
switched by an environment variable (``MIME_PLANT_<NAME>``), each draw seeded by ``DETECTION_SEED``, for
``.github/probe/detection_rate.py``."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from mime.core import run_dtype
from mime.dynamics import FVM, RigidInertial
from mime.dynamics.flow._fvm import Scheme
from mime.experiment import Experiment
from mime.interactions import Drag, Inflow, Windkessel
from mime.interactions.flow import BodyForce
from mime.mappings import gaussian_mac
from mime.nodes.flow.fvm import ImplicitFlowNode
from mime.nodes.hydro._point_drag import AVERAGES, COUPLE, FAXEN_FORCES, FORCES, LAPLACIANS, ROTATIONS, PointDragNode
from mime.scene import Capsule, PeriodicBox, Scene, Sphere, Tube, TubeTree, VesselRegion, Z
from mime.things import Body, Medium, Part, State, Structure
from mime.units import Pa_s, Pa_s_per_m3, kg, kg_m2, kg_per_m3, m3_per_Pa, m_per_s, m_per_s2, mL_per_s, mm, rad_per_s, s, \
    um

WATER_MU, WATER_RHO = 1e-3, 1e3
TUBE_RADIUS = 0.5e-3
#: A tube tree narrowing to 0.8 of its radius over half its 1 mm period and back (``tests/flow/
#: test_fvm_tree_declaration.py``'s), its parent reaching past both ends.
NARROWING = dict(points=mm((0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.5)),
                 segments=((0, 1), (1, 2), (2, 3), (3, 4)), radii=mm((0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5)))


def plant(name: str) -> str:
    """The plant switched on for this run (``MIME_PLANT_<NAME>``'s value), or an empty string."""
    return os.environ.get(f"MIME_PLANT_{name}", "")


def seed() -> int:
    return int(os.environ.get("DETECTION_SEED", "0"))


def body_step(mass: float, resistance: Any, load: Any, velocity: Any, dt: float, rows: slice = slice(0, 3)) -> Any:
    """The rigid body's step of one block (translation by default; ``rows=slice(3, 6)`` the rotation, ``mass`` then
    the moment of inertia), as ``RigidInertialNode.velocities`` takes it: ``V + (M/dt + R)^-1 (load - R V)``."""
    block = resistance[rows, rows]
    return velocity + jnp.linalg.solve(mass / dt * jnp.eye(3) + block, load[rows] - block @ velocity)


class FreeBox:
    """A point body in a free periodic box (the tube outside it, so no wall), stepped by hand in the graph's order
    (the drag's node, the flow's, the body's) through the real nodes and the mapping kind's own transfers; a uniform
    counter-force holds the box's net force at zero (the box's mean is no unbounded fluid's). ``pinned`` holds the
    body's point where it starts (its velocity still stepped), so that its own field is read where it was made."""

    def __init__(self, cells: int, radius_cells: float, dt_r: float, *, mass_ratio: float = 1.0,
                 force: float = 0.0, torque: float = 0.0, faxen: bool = True, maxey_riley: bool = True,
                 width: float = 1.0, self_history: str = "exact", one_way: bool = False, pinned: bool = False,
                 rate_reading: Callable[..., Any] | None = None) -> None:
        h = 1e-5
        self.h, self.cells = h, cells
        self.dt = dt_r * h * h * WATER_RHO / WATER_MU
        side = cells * h
        self.flow = ImplicitFlowNode("flow", self.dt, cells_x=cells, cells_y=cells, cells_z=cells, spacing=h,
                                     viscosity=WATER_MU, density=WATER_RHO, tube_centre=(side / 2, side / 2, 0.0),
                                     tube_axis=(0.0, 0.0, 1.0), tube_radius=10.0 * side, scheme="imex_euler",
                                     convection=False, tolerance=1e-10 if run_dtype() == jnp.float64 else 1e-5,
                                     exchange=True)
        self.radius = radius_cells * h
        start = (side / 2 + 0.37 * h, side / 2 + 0.21 * h, side / 2 + 0.11 * h)
        self.drag = PointDragNode("drag", self.dt, radius=self.radius, viscosity=WATER_MU, density=WATER_RHO,
                                  spacing=h, width_cells=width, origin=(0.0, 0.0, 0.0), initial_points=start,
                                  faxen=faxen, maxey_riley=maxey_riley, self_history=self_history, one_way=one_way)
        self.mass = mass_ratio * WATER_RHO * 4.0 / 3.0 * math.pi * self.radius ** 3
        self.moment = 0.4 * self.mass * self.radius ** 2
        self.force, self.torque = force, torque
        self.position = jnp.asarray(start, dtype=run_dtype())
        self.velocity = jnp.zeros(3, dtype=run_dtype())
        self.spin = jnp.zeros(3, dtype=run_dtype())
        self.flow_state = self.flow.initial_state()
        self.drag_state = self.drag.initial_state()
        self.kernels = {(c, mode, direction): gaussian_mac(nx=cells, ny=cells, nz=cells, spacing=h, width_cells=width,
                                                           component=c, mode=mode, direction=direction, count=1)
                        for c in range(3) for mode in ("average", "laplacian", "curl")
                        for direction in ("gather", "scatter")}
        self.rate_reading = rate_reading
        self.pinned = pinned
        self.gathered: dict[str, Any] = {}

    def step(self) -> None:
        dtype = run_dtype()
        points = jnp.asarray(self.drag_state["points"], dtype=dtype)
        gathered = {}
        for c, (average, laplacian) in enumerate(zip(AVERAGES, LAPLACIANS, strict=True)):
            field = self.flow_state[f"velocity_{'xyz'[c]}"]
            gathered[average] = self.kernels[(c, "average", "gather")].apply(field, geom=points)
            gathered[laplacian] = self.kernels[(c, "laplacian", "gather")].apply(field, geom=points)
            gathered[ROTATIONS[c]] = self.kernels[(c, "curl", "gather")].apply(field, geom=points)
        pose = jnp.concatenate([self.position, jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)])
        twist = jnp.concatenate([self.velocity, self.spin])
        self.drag_state = self.drag.update(dict(self.drag_state), {"pose": pose, "twist": twist, **gathered},
                                           self.dt)
        self.gathered = gathered
        if self.rate_reading is not None:
            self.drag_state = self.rate_reading(self, gathered)
        points = jnp.asarray(self.drag_state["points"], dtype=dtype)
        exchange = {}
        total = jnp.zeros(3, dtype=dtype)
        for c, axis in enumerate("xyz"):
            force = jnp.asarray(self.drag_state[FORCES[c]], dtype=dtype)
            faxen = jnp.asarray(self.drag_state[FAXEN_FORCES[c]], dtype=dtype)
            couple = jnp.asarray(self.drag_state[COUPLE], dtype=dtype)
            exchange[f"exchange_{axis}"] = (self.kernels[(c, "average", "scatter")].apply(force, geom=points)
                                            + self.kernels[(c, "laplacian", "scatter")].apply(faxen, geom=points)
                                            + self.kernels[(c, "curl", "scatter")].apply(couple, geom=points))
            total = total.at[c].set(force[0])
        volume = (self.cells * self.h) ** 3
        counter = -total / (WATER_RHO * volume)
        self.flow_state = self.flow.update(dict(self.flow_state), {"body_force": counter, **exchange}, self.dt)
        load = (jnp.asarray(self.drag_state["background_load"], dtype=dtype)
                + jnp.asarray(self.drag_state["added_load"], dtype=dtype)).at[2].add(self.force)
        resistance = (jnp.asarray(self.drag_state["resistance"], dtype=dtype)
                      + jnp.asarray(self.drag_state["added_resistance"], dtype=dtype))
        self.velocity = body_step(self.mass, resistance, load, self.velocity, self.dt)
        self.spin = body_step(self.moment, resistance, load.at[5].add(self.torque), self.spin, self.dt, slice(3, 6))
        if not self.pinned:
            self.position = self.position + self.dt * self.velocity


def point_experiment(*, radius: float = 40e-6, offset: float = 0.0, dt: float = 2e-4, cells_across: int = 16,
                     drive: float = 1.0, mass_ratio: float = 1.0, one_way: bool = False, steady: bool = False,
                     kernel_width: float | None = None, scheme: str = "", flow: bool = True,
                     hull: str = "sphere", viscosity: float = WATER_MU, self_history: str | None = None,
                     opened: bool = False, tree: bool = False, coupling: str | None = None) -> Experiment:
    """A point sphere of ``radius`` (m) ``offset`` (m) off a straight tube's axis, in water whose flow is the FVM
    (Stokes, IMEX Euler or its steady limit, or ``scheme="rk3"``) driven along the axis by a uniform ``drive``
    (m/s^2); ``flow=False`` leaves the medium at rest with no field, ``hull="capsule"`` gives the body a capsule,
    ``opened=True`` opens the vessel at its ends (an inflow at its start, a Windkessel at its end) in place of the
    periodic box and its drive; ``tree=True`` makes the vessel a tube tree narrowing over its period; ``coupling`` is the
    drag's (one pass by default, ``"lagged"`` the declared route)."""
    exp = Experiment("point_body", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    geometry = TubeTree(**NARROWING) if tree else Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(TUBE_RADIUS * 1e3),
                                                       wall=mm(0.1), length=mm(4.0))
    exp.add_structure(Structure("tube", geometry=geometry))
    chosen = Scheme.RK3() if scheme == "rk3" else Scheme.Steady() if steady else Scheme.IMEXEuler()
    domain = VesselRegion(margin=4) if opened else PeriodicBox(length=mm(1.0))
    fields = {"flow": FVM(domain=domain, stokes=True, cells_across=cells_across, scheme=chosen)} if flow else {}
    exp.add_medium(Medium("water", fills="tube.lumen", fields=fields))
    exp.declare("water.viscosity", Pa_s(viscosity), reason="water")
    exp.declare("water.density", kg_per_m3(WATER_RHO), reason="water")
    shape = Sphere(radius=um(radius * 1e6)) if hull == "sphere" else Capsule(radius=um(radius * 1e6),
                                                                             height=um(4.0 * radius * 1e6), axis=(0.0, 0.0, 1.0))
    exp.add_body(Body("ball", dynamics=RigidInertial(), parts={"shell": Part(shape)},
                      initial=State(position=mm((offset * 1e3, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    mass = mass_ratio * WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    exp.declare("ball.mass", kg(mass), reason="a sphere of the drawn density")
    moment = 0.4 * mass * radius ** 2
    exp.declare("ball.inertia", kg_m2((moment, moment, moment)), reason="a uniform sphere")
    exp.attach(Drag.Stokes(one_way=one_way, kernel_width=kernel_width, self_history=self_history, coupling=coupling),
               body="ball", medium="water")
    if flow and opened:
        exp.attach(Inflow(steady=mL_per_s(1e-3)), medium="water", surface="tube.start")
        exp.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
                   medium="water", surface="tube.end")
    elif flow:
        exp.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(drive)), medium="water")
    return exp


def with_second_body(exp: Experiment, at: tuple[float, float, float], *, radius: float = 40e-6,
                     one_way: bool = False, coupling: str | None = None) -> Experiment:
    """``exp`` with a second neutral point sphere, ``bob``, of ``radius`` (m) starting ``at`` (m) in the same water."""
    exp.add_body(Body("bob", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=um(radius * 1e6)))},
                      initial=State(position=mm(tuple(c * 1e3 for c in at)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    mass = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    exp.declare("bob.mass", kg(mass), reason="a neutral sphere")
    exp.declare("bob.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="a uniform sphere")
    exp.attach(Drag.Stokes(one_way=one_way, coupling=coupling), body="bob", medium="water")
    return exp


def axis_profile(state: dict[str, Any], centre: tuple[float, float, float], spacing: float) -> tuple[Any, Any]:
    """The axial velocity on the row of z faces through the tube's centre along x, and each face's distance from the
    axis (m): what Poiseuille's profile is fitted to."""
    w = np.asarray(state["velocity_z"])
    i_centre = int(round(centre[0] / spacing - 0.5))
    j_centre = int(round(centre[1] / spacing - 0.5))
    row = w[:, j_centre, 0]
    x = (np.arange(w.shape[0]) + 0.5) * spacing - centre[0]
    del i_centre
    return x, row


__all__ = ["TUBE_RADIUS", "WATER_MU", "WATER_RHO", "FreeBox", "axis_profile", "body_step", "plant", "point_experiment",
           "with_second_body",
           "seed"]
