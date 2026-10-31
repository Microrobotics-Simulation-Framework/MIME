"""The rigs the results tests run: a bead falling through water under gravity with nothing to resist it, so its speed
along the fall grows by the same amount every step and any one-step change of a window moves the speed's mean by half
a step's gain, at every draw; and a synthetic run, built from arrays with no graph, for the rules that need none."""

from __future__ import annotations

import numpy as np

from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.registry import OutputKind, Reduction
from mime.results import FlagLayout, FlagTrack, Layout, ObservationLayout, OutputLayout, RunResult, Samples
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import LENGTH, VELOCITY, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mPa_s, rad_per_s, s

GRAVITY = 9.81
WATER = 1000.0


def falling_bead(dt: float, after: float, density: float, volume: float, v0: float, viscosity: float | None = None):
    """A bead of ``density`` (kg/m^3) and ``volume`` (m^3) released at ``v0`` (m/s) downwards in still water, with no
    drag: its speed downwards grows by ``(1 - WATER / density) g dt`` each step. With ``viscosity`` (Pa s) the bead is
    a sphere with Stokes drag in the water (``Drag.BEM``, linear in its velocity), so its speed approaches a terminal
    speed as an exponential with the time constant ``m / (6 pi viscosity radius)``: a speed that is not linear in
    time. Returns the experiment and the swim speed's handle, observed downwards from ``after`` (s)."""
    experiment = Experiment("falling_bead", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(GRAVITY), dt=s(dt))
    water = experiment.add_medium(Medium("water"))
    experiment.declare("water.density", kg_per_m3(WATER), reason="the rig's")
    experiment.declare("water.viscosity", mPa_s(1.0 if viscosity is None else viscosity * 1e3),
                       reason="the rig's: nothing reads it" if viscosity is None else "the rig's")
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    parts = {} if viscosity is None else {"shell": Part(Sphere(radius=m(radius)))}
    bead = experiment.add_body(Body("bead", dynamics=RigidInertial(), parts=parts, initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, -v0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    mass = density * volume
    experiment.declare("bead.mass", kg(mass), reason="the rig's")
    experiment.declare("bead.inertia", kg_m2((0.4 * mass * radius**2,) * 3), reason="a solid ball's")
    experiment.declare("bead.buoyant_volume", m3(volume), reason="the rig's")
    experiment.declare("bead.centre_of_buoyancy", m((0.0, 0.0, 0.0)), reason="a uniform ball's")
    experiment.attach(Gravity(), body=bead, medium=water)
    if viscosity is not None:
        experiment.attach(Drag.BEM(spacing=m(0.8 * radius), blob=0.5, wall_kind="free_space"), body=bead,
                          medium=water)
    swim = experiment.observe(SwimSpeed(bead, along=(0.0, 0.0, -1.0), after=s(after)))
    return experiment, swim


def stokes_time_constant(density: float, radius: float, viscosity: float) -> float:
    """The time constant (s) of a sphere's speed approaching its terminal speed under Stokes drag."""
    return density * (4.0 / 3.0) * np.pi * radius**3 / (6.0 * np.pi * viscosity * radius)


def viscosity_for(density: float, volume: float, time_constant: float) -> float:
    """The viscosity (Pa s) at which a sphere of ``density`` and ``volume`` has the speed time constant
    ``time_constant``."""
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    return stokes_time_constant(density, radius, 1.0) / time_constant


def gain_per_step(dt: float, density: float) -> float:
    """How much the bead's speed grows in a step (m/s)."""
    return (1.0 - WATER / density) * GRAVITY * dt


#: A synthetic run's layout: one observation with a per-step speed, a vector position and a mean at the end, and one
#: flag on another entry.
SPEED = OutputLayout("probe:p", "speed", "probe:p", "speed", OutputKind.PER_STEP, VELOCITY, (), "a speed")
WHERE = OutputLayout("probe:p", "where", "probe:p", "where", OutputKind.PER_STEP, LENGTH, (3,), "a position")
MEAN = OutputLayout("probe:p", "mean", "probe:p", "mean", OutputKind.AT_END, VELOCITY, (), "its mean",
                    reduces="speed", rule=Reduction.MEAN, after=None)
FLAG = FlagLayout("law:p", "law 0.1.0", "law:p", "fired", "what the law's flag means")


def synthetic_run(steps: int, dt: float, *, raised: np.ndarray | None = None, recorded: bool = True,
                  start_step: int = 0, every: int = 1) -> RunResult:
    """A run of ``steps`` steps from ``start_step``, its speed the step count, its position three times it, the flag
    raised where ``raised`` says (one per step), with no graph."""
    counts = np.arange(start_step + 1, start_step + steps + 1, dtype=np.int64)
    kept = counts[counts % every == 0]
    speed = counts.astype(np.float32)
    flags = np.zeros(steps, dtype=bool) if raised is None else np.asarray(raised, dtype=bool)
    layout = Layout("synthetic", dt, (ObservationLayout("probe:p", "probe 0.1.0", (SPEED, WHERE, MEAN)),), (FLAG,))
    final = {"probe:p": {"speed": speed[-1:].reshape(()), "where": np.full(3, speed[-1]), "mean": np.asarray(
        speed.mean(), dtype=np.float32)}, "law:p": {"fired": np.asarray(float(flags[-1]), dtype=np.float32)}}
    series = {SPEED.path: Samples(kept, kept.astype(np.float32)),
              WHERE.path: Samples(kept, np.stack([kept, 2 * kept, 3 * kept], axis=1).astype(np.float32))}
    track = FlagTrack(flags if recorded else None, bool(flags[-1]))
    return RunResult(steps, final, layout=layout, start_step=start_step, recorded=series if recorded else {},
                     tracks={FLAG.path: track}, every=every if recorded else None)
