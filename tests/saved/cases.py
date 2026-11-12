"""The round-trip cases the registry walk requires: together they use every registered entry, MIME's toys and the
lab fixture's, and every kind of thing. An entry registered without a case here fails the walk.

A case whose build is refused (an entry that exists to be refused) round-trips its document instead: written, read
back in a fresh process, and refused there for the same reasons.
"""

import dataclasses
import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from mrs_cases import (free_space, mrs, mrs_free_space, mrs_graded, mrs_kept, mrs_near_field, mrs_near_field_gmres,
                       mrs_plane, mrs_solids, mrs_tube)
from rft_cases import rft_free_space, rft_kept, rft_tube
from chain_cases import dynamic_rpm, gantry_magnet, kinematic_rpm, massive_rpm, swing_magnet, tracked_rpm
from hull_cases import hull_contact
from robot_asset import bound_robot, bound_robot_with_parts
from script_a import ALL_RECORDS, EPSILON, fl9, script_a

from mime.build import Attachment, ThingDeclaration
from mime.data import asset
from mime.drives import Ramp
from mime.fields import KEPT as KEPT_FIELDS
from mime.fields import UniformRotating
from mime.observations import PhaseLag
from mime.dynamics.body import RigidInertialExplicit, RigidInertialMoments
from mime.interactions import KEPT, Drag
from mime.experiment import Experiment
from mime.interactions import Lubrication
from mime.interactions.hydrodynamic import PointForceDrag, UncoupledDrag
from mime.scene import Capsule, Sphere
from mime.testing.toys import (
    ToyAnchoredBody,
    ToyBody,
    ToyRegionProbe,
    ToySpring,
    ToyTwist,
    coupled_toy,
    full_toy,
    toy_experiment,
)
from mime.scene import X
from mime.things import Apparatus, Body, Medium, State, Structure
from mime.units import Hz, N, kg, kg_m2, m, m_per_s, mm, ms, rad_per_s, s


KEPT_TORQUE, _ = KEPT["magnetic.torque", "0.1.0"]
KEPT_FORCE, _ = KEPT["magnetic.force", "0.1.0"]
KEPT_CONTACT, _ = KEPT["contact", "0.4.0"]


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[], Experiment]
    refused: tuple[str, ...] = ()
    #: Whether it uses the lab fixture's entries, which every CI lane installs and a local run may not.
    lab: bool = False


def periodic_tube_flow(steady: bool = False, rk3: bool = False) -> Experiment:
    """The FVM on its periodic box around a straight tube, eight cells across the lumen, driven by a pulsatile body
    force on the fluid; IMEX Euler with convection on, or its steady limit under Stokes."""
    from mime.dynamics import FVM, Scheme
    from mime.interactions import BodyForce
    from mime.scene import PeriodicBox, Scene, Tube, Z
    from mime.units import Pa_s, kg_per_m3, m_per_s2

    name = "steady_tube_flow" if steady else "rk3_tube_flow" if rk3 else "periodic_tube_flow"
    # dt 0.25 ms: the drive's periodic state is predicted at 0.09 m/s, a CFL of 0.18 on the case's grid, inside IMEX
    # Euler's 0.4 (at 1 ms the build refuses it, flow.cfl-predicted).
    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(0.25), margin=4), cells_across=8, stokes=steady,
               scheme=Scheme.Steady() if steady else Scheme.RK3() if rk3 else Scheme.IMEXEuler())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0), amplitude=m_per_s2(0.5), frequency=Hz(2.0)),
                      medium="water")
    return experiment


def point_body_flow(self_history: str | None = None) -> Experiment:
    """A point sphere coupled two-way to the FVM (drag 0.5.0's stokes rung) in the periodic tube's steady drive, off the
    axis, slightly heavier than the water: IMEX Euler under Stokes, the exchange law's gathers and scatters on
    MADDENING's geometry edges; its self-history the exact route, or ``self_history`` written."""
    from mime.dynamics import FVM, RigidInertial, Scheme
    from mime.interactions import BodyForce, Drag
    from mime.scene import PeriodicBox, Scene, Sphere, Tube, Z
    from mime.things import Body, Part, State
    from mime.units import Pa_s, kg, kg_m2, kg_per_m3, m_per_s, m_per_s2, rad_per_s, um

    experiment = Experiment("point_body_flow", scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(10.0))))
    flow = FVM(domain=PeriodicBox(length=mm(0.25), margin=4), cells_across=8, stokes=True, scheme=Scheme.IMEXEuler())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    radius = 50e-6
    mass = 1.2 * 1000.0 * 4.0 / 3.0 * 3.141592653589793 * radius ** 3
    experiment.add_body(Body("bead", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=um(50.0)))},
                             initial=State(position=mm((0.15, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("bead.mass", kg(mass), reason="a bead 1.2 times water's density")
    experiment.declare("bead.inertia", kg_m2((0.4 * mass * radius ** 2,) * 3), reason="a uniform sphere")
    experiment.attach(Drag.Stokes() if self_history is None else Drag.Stokes(self_history=self_history), body="bead",
                      medium="water")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    return experiment


def tree_flow() -> Experiment:
    """The FVM on a tube tree's lumen in its periodic box (``TubeTree``, filleted): a vessel narrowing to 0.8 of its
    radius over half its period and back, its parent reaching a radius past each end of the period, steady Stokes under
    a body force on the fluid; the throat twelve cells across, so the wall is the union's exact distance at its
    junctions."""
    from mime.dynamics import FVM, Scheme
    from mime.interactions import BodyForce
    from mime.scene import PeriodicBox, Scene, TubeTree, Z
    from mime.units import Pa_s, kg_per_m3, m_per_s2

    experiment = Experiment("tree_flow", scene=Scene.new(up=Z, units="m"), dt=ms(5.0))
    tree = TubeTree(points=mm((0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.75, 0.0, 0.0, 1.5)),
                    segments=((0, 1), (1, 2), (2, 3), (3, 4)), radii=mm((0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5)))
    experiment.add_structure(Structure("vessel", geometry=tree))
    flow = FVM(domain=PeriodicBox(length=mm(1.0), margin=4), cells_across=12, stokes=True, scheme=Scheme.Steady())
    experiment.add_medium(Medium("water", fills="vessel.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(0.01)), medium="water")
    return experiment


def prescribed_tube_flow() -> Experiment:
    """The prescribed flow (Womersley) in the same tube under the same drive: the cheap rung, no grid."""
    from mime.dynamics import Womersley
    from mime.interactions import BodyForce
    from mime.scene import Scene, Tube, Z
    from mime.units import Pa_s, kg_per_m3, m_per_s2

    experiment = Experiment("prescribed_tube_flow", scene=Scene.new(up=Z, units="m"), dt=ms(1.0))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(10.0))))
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": Womersley()}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0), amplitude=m_per_s2(0.5), frequency=Hz(2.0)),
                      medium="water")
    return experiment


def open_tube_flow() -> Experiment:
    """The FVM on the open vessel: the same tube, one millimetre long, cut at its own ends, an inflow at its start
    (steady and pulsatile) and an RCR Windkessel at its end; IMEX Euler with convection on."""
    from mime.dynamics import FVM, Scheme
    from mime.interactions import Inflow, Windkessel
    from mime.scene import Scene, Tube, VesselRegion, Z
    from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, mL_per_s

    experiment = Experiment("open_tube_flow", scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(1.0))))
    flow = FVM(domain=VesselRegion(margin=4), cells_across=8, scheme=Scheme.IMEXEuler())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(Inflow(steady=mL_per_s(1e-3), amplitude=mL_per_s(3e-4), frequency=Hz(2.0)), medium="water",
                      surface="tube.start")
    experiment.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
                      medium="water", surface="tube.end")
    return experiment


def open_tube_loop() -> Experiment:
    """The open vessel closed on itself: a shared distal resistance across its two ends (``surface=`` its lumen's
    openings), a body force driving the loop; steady Stokes."""
    from mime.dynamics import FVM, Scheme
    from mime.interactions import BodyForce, SharedDistal
    from mime.scene import Scene, Tube, VesselRegion, Z
    from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m_per_s2

    experiment = Experiment("open_tube_loop", scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(1.0))))
    flow = FVM(domain=VesselRegion(margin=4), cells_across=8, stokes=True, scheme=Scheme.Steady())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    experiment.attach(SharedDistal(first=Pa_s_per_m3(4e7), second=Pa_s_per_m3(1.2e8), distal=Pa_s_per_m3(2e8)),
                      medium="water", surface="tube.lumen")
    return experiment


def closed_tube() -> Experiment:
    """The open vessel with a Windkessel at its start and a closed valve at its end, a body force pushing its fluid
    against the valve; IMEX Euler, Stokes."""
    from mime.dynamics import FVM, Scheme
    from mime.interactions import BodyForce, Valve, Windkessel
    from mime.scene import Scene, Tube, VesselRegion, Z
    from mime.units import Pa_s, Pa_s_per_m3, kg_per_m3, m3_per_Pa, m_per_s2

    experiment = Experiment("closed_tube", scene=Scene.new(up=Z, units="m"), dt=ms(0.25))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+Z, inner_radius=mm(0.5), wall=mm(0.2),
                                                             length=mm(1.0))))
    flow = FVM(domain=VesselRegion(margin=4), cells_across=8, stokes=True, scheme=Scheme.IMEXEuler())
    experiment.add_medium(Medium("water", fills="tube.lumen", fields={"flow": flow}))
    experiment.declare("water.viscosity", Pa_s(1e-3), reason="water near 20 C")
    experiment.declare("water.density", kg_per_m3(1000.0), reason="water near 20 C")
    experiment.attach(BodyForce(along=(0.0, 0.0, 1.0), steady=m_per_s2(1.0)), medium="water")
    experiment.attach(Windkessel(proximal=Pa_s_per_m3(1e8), compliance=m3_per_Pa(1e-13), distal=Pa_s_per_m3(1e9)),
                      medium="water", surface="tube.start")
    experiment.attach(Valve(), medium="water", surface="tube.end")
    return experiment


def walled_toy() -> Experiment:
    """The coupled toy beside a passive structure, which builds no node and saves as a thing."""
    experiment = coupled_toy()
    experiment.add_structure(Structure("wall"))
    return experiment


def anchored_toy() -> Experiment:
    experiment = toy_experiment("anchored_toy")
    experiment.add_body(Body("anchor", dynamics=ToyAnchoredBody(), initial=State(position=m((0.1, 0.0, -0.2)))))
    experiment.attach(ToySpring(k=N(2.3) / m(1.0)), body="anchor")
    return experiment


def twisted_toy() -> Experiment:
    experiment = coupled_toy()
    experiment.attach(ToyTwist(), body="probe")
    return experiment


def nudged_toy() -> Experiment:
    """The lab fixture's entries beside MIME's toys, its interaction and its observation: its lock names its own
    distribution."""
    lab = importlib.import_module("mime_lab_fixture")
    experiment = coupled_toy()
    experiment.attach(lab.Nudge(force=N((0.0, 0.01, -0.04))), body="probe")
    experiment.observe(lab.Height(body="probe"))
    return experiment


def lab_bead() -> Experiment:
    """The lab fixture's geometry kind as a body's hull under the drag's MRS rung, which samples it by the geometry
    protocol as it samples MIME's kinds, moving and turning from the start."""
    lab = importlib.import_module("mime_lab_fixture")
    return free_space("lab_bead", {"bead": lab.Bead(radius=mm(0.8))}, mrs(mm(0.45)), mass=kg(1e-7),
                      inertia=kg_m2((1e-14, 2e-14, 3e-14)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                      spin=rad_per_s((3.0, -1.0, 2.0)))


def script_a_body() -> Experiment:
    """Script A's body side, with no drag, the body moving along the tube at 1 mm/s so that every leaf of every node
    acts within the walk's first steps: 4b's case, which the body side's golden outputs come from. Its torque is the
    one kept registered, 0.1.0, whose golden output came from it; in a uniform field 0.2.0 computes the same, and its
    golden output comes from script A's case. Its body's dynamics likewise: ``rigid_inertial`` 0.3.0, kept registered,
    whose golden output came from it; 0.4.0 steps this isotropic body as 0.3.0 did, and its golden output comes from
    script A's case. And its field: ``uniform_rotating`` 0.1.0, kept registered, whose golden output came from it;
    0.2.0 computes the same field for its constant frequency, and its golden output comes from script A's case."""
    return on_the_kept_field(on_the_kept_rigid_inertial(on_the_kept_magnetic_laws(
        script_a("script_a_body", velocity=m_per_s((1e-3, 0.0, 0.0)), drag=False))))


def on_the_kept_field(experiment: Experiment) -> Experiment:
    """``experiment`` with each uniform rotating field at ``uniform_rotating`` 0.1.0, kept registered, and nothing else
    changed."""
    held = cast(Any, experiment)
    kept, _ = KEPT_FIELDS["uniform_rotating", "0.1.0"]
    declarations = []
    for d in held._declarations:
        if isinstance(d, ThingDeclaration) and isinstance(d.thing, Apparatus):
            fields = {name: kept(**{f.name: getattr(v, f.name) for f in dataclasses.fields(v)})
                      if isinstance(v, UniformRotating) else v for name, v in d.thing.fields.items()}
            d = ThingDeclaration(dataclasses.replace(d.thing, fields=fields))
        declarations.append(d)
    return cast(Experiment, held._declared(held._settings, declarations))


def script_a_ramped() -> Experiment:
    """Script A with its field's frequency ramped from 10 Hz after a step's hold, at 200 Hz/s to 10.2 Hz, where it holds
    from the fourth step (``drive.ramp``), and its phase lag observed (``phase_lag``), so that the ramp's hold, its rate
    and its end act within the walk's first steps: the drive's and the observation's golden outputs come from it."""
    experiment = script_a("script_a_ramped", velocity=m_per_s((1e-3, 0.0, 0.0)), frequency=Ramp(
        Hz(10.0), to=Hz(10.2), rate=Hz(200.0) / s(1.0), hold=ms(0.5)))
    experiment.observe(PhaseLag("umr", field="field.magnetic", about=X))
    return experiment


def on_the_kept_rigid_inertial(experiment: Experiment) -> Experiment:
    """``experiment`` with each body on ``rigid_inertial`` 0.3.0, kept registered, and nothing else changed."""
    held = cast(Any, experiment)
    declarations = [ThingDeclaration(dataclasses.replace(d.thing, dynamics=RigidInertialMoments()))
                    if isinstance(d, ThingDeclaration) and isinstance(d.thing, Body) else d
                    for d in held._declarations]
    return cast(Experiment, held._declared(held._settings, declarations))


def on_the_kept_magnetic_laws(experiment: Experiment) -> Experiment:
    """``experiment`` with each magnetic law it attaches at its kept version, 0.1.0, bound as before, and nothing else
    changed (script A's builder, a preregistered file, is not edited for it)."""
    held = cast(Any, experiment)
    declarations = [Attachment(KEPT[(d.law.name, "0.1.0")][0](), d.bindings)
                    if isinstance(d, Attachment) and d.law.name.startswith("magnetic.") else d
                    for d in held._declarations]
    return cast(Experiment, held._declared(held._settings, declarations))


def kinematic_rpm_kept() -> Experiment:
    """A0's apparatus, kinematic, on the magnetic laws kept registered, 0.1.0 (the field read at the body's centre of
    mass, MIME-ANO-137): what a folder saved before 0.2.0 names, and the case their golden outputs and the point
    dipole's came from. The body's magnet is at its centre of mass, where 0.2.0 computes the same."""
    return kinematic_rpm(magnetic=(KEPT_TORQUE, KEPT_FORCE))


def swing_magnet_kept() -> Experiment:
    """A one-joint dynamic swing carrying a magnet above a body, on the magnetic laws kept registered, 0.1.0: each
    delivers its reaction to the swing, as a folder saved before 0.2.0 names it."""
    return swing_magnet("swing_magnet_kept", magnetic=(KEPT_TORQUE, KEPT_FORCE))


def hull_contact_kept() -> Experiment:
    """FL-9 held by its hull's points from the first step on the contact kept registered, 0.4.0 (its loops unrolled):
    what a folder saved before 0.5.0 names, and the case its golden output came from."""
    return hull_contact(contact=KEPT_CONTACT)


def script_a_with_drag() -> Experiment:
    """Script A, with the drag surrogate and its lubrication (drag 0.4.0, kept registered, which keeps the blend), the
    body moving as in the body side's case: drag 0.4.0's golden output comes from it."""
    return script_a(velocity=m_per_s((1e-3, 0.0, 0.0)))


def script_a_unblended() -> Experiment:
    """Script A with drag 0.5.0's surrogate rung, which has no blend (retired in 0.5.0, ``lubrication.retired``), the
    body moving as in the body side's case: drag 0.5.0's golden output, and its surrogate rung's round trip."""
    return script_a("script_a_unblended", velocity=m_per_s((1e-3, 0.0, 0.0)),
                    drag=Drag.Surrogate(weights=asset("dejongh_mlp_v2")))


def script_a_uncoupled_drag() -> Experiment:
    """Script A with the drag kept registered, 0.4.0 (``UncoupledDrag``), under its own name: what a folder saved before
    the near field names. Script A itself is on 0.4.0 since 0.5.0 retired the blend, so 0.4.0's golden output comes from
    script A's case, the same program."""
    return script_a("script_a_uncoupled", velocity=m_per_s((1e-3, 0.0, 0.0)),
                    drag=UncoupledDrag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(
                        epsilon=EPSILON)))


def script_a_metered() -> Experiment:
    """Script A with the energy ledger attached: its body metered, each law in its own input on it."""
    from mime.observations import EnergyLedger

    experiment = script_a("script_a_metered", velocity=m_per_s((1e-3, 0.0, 0.0)))
    experiment.observe(EnergyLedger())
    return experiment


def script_a_kept_drag() -> Experiment:
    """Script A with the drag kept registered, 0.3.0 (its surrogate rung is 0.4.0's): what a folder saved before the
    separated quadrature names."""
    return script_a("script_a_kept", velocity=m_per_s((1e-3, 0.0, 0.0)),
                    drag=PointForceDrag.Surrogate(weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(
                        epsilon=EPSILON)))


def mrs_point_forces() -> Experiment:
    """The MRS rung's free-space case on drag 0.3.0, kept registered: a point force at each sample, as before the
    separated quadrature, on the ball, the pill and the screw, so that its golden output is 0.3.0's own."""
    return mrs_free_space(law=PointForceDrag.MRS(spacing=mm(1.5), blob=0.5, wall_kind="free_space"),
                          name="mrs_point_forces")


def script_a_records() -> Experiment:
    """Script A's record arm (water and N45 from OpenMaterica's example records, the body from its parts), the body
    moving as in the body side's case: the folder carries the record files, and the fresh process reads them there."""
    return script_a("script_a_records", velocity=m_per_s((1e-3, 0.0, 0.0)), records=ALL_RECORDS)


def probed_shapes() -> Experiment:
    """A toy body on a spring passing through three solid structures, a ball, a capsule and FL-9's helix, each region
    filled by a medium a probe reads at the body: every shape a solid kind names, evaluated in a step."""
    experiment = toy_experiment("probed_shapes")
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=State(position=mm((-0.4, 0.6, 0.3)),
                                                                        velocity=m_per_s((0.02, -0.01, 0.005)))))
    experiment.attach(ToySpring(), body="probe")
    solids = (("ball", Sphere(radius=mm(1.0)), ("solid",)),
              ("pill", Capsule(radius=mm(0.5), height=mm(0.4), axis=(0.0, 0.6, 0.8)), ("solid",)),
              ("screw", fl9(), ("screw", "span")))
    for name, geometry, regions in solids:
        experiment.add_structure(Structure(name, geometry=geometry))
        for region in regions:
            experiment.add_medium(Medium(f"{name}_{region}", fills=f"{name}.{region}"))
            experiment.observe(ToyRegionProbe(body="probe", medium=f"{name}_{region}"))
    return experiment


def coil_array(**settings: Any) -> Experiment:
    """A millimetre body in the Navion_2's field, the calibrated multipole placed off the world's origin and turned,
    with a background field (``settings`` written over these), its currents a constant drive, and dev1's magnetic pair
    (0.2.0) on the body, its magnet off its centre of mass: so that the walk moves the array's pose and background and
    the pair reads it at the magnet's centroid. No gravity, no medium."""
    from mime.data import asset
    from mime.dynamics import RigidInertial
    from mime.fields import CalibratedMultipole
    from mime.interactions import Magnetic
    from mime.scene import Scene, Z
    from mime.units import A, A_m2, T, m_per_s2

    experiment = Experiment("coil_array", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    placed = {"at": m((0.01, -0.02, 0.005)), "orientation": (0.96, 0.0, 0.0, 0.28), "background": T((2e-5, 0.0, -4e-5)),
              **settings}
    experiment.add_apparatus(Apparatus("coils", fields={"magnetic": CalibratedMultipole(
        calibration=asset("navion_2"), **placed)}, commands={"currents": A((5.0, -3.0, 2.0))}))
    experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((3.0, 2.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((1e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("umr.mass", kg(4e-6)), ("umr.inertia", kg_m2((1.6e-10, 1.6e-10, 1.6e-10))),
                        ("umr.magnetic_moment", A_m2((1e-3, 0.0, 5e-4))), ("umr.magnet_centre", mm((0.2, 0.0, 0.0)))):
        experiment.declare(path, value, reason="a millimetre body's magnet, its order")
    experiment.attach(Magnetic.Torque(), body="umr", field="coils.magnetic")
    experiment.attach(Magnetic.Force(), body="umr", field="coils.magnetic")
    return experiment


def rigid_explicit() -> Experiment:
    """A ball of three different moments, moving and spinning at 200 rad/s in water, on the kept ``rigid_inertial``
    0.2.0 (the gyroscopic term explicit, MIME-ANO-129) with the drag's MRS rung: what a folder saved before the term
    was made implicit names, and the golden of that version, which gains energy where 0.3.0's does not."""
    return free_space("rigid_explicit", {"ball": Sphere(radius=mm(1.0))}, mrs(mm(1.5)), mass=kg(1e-6),
                      inertia=kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                      spin=rad_per_s((150.0, 100.0, -100.0)), dynamics=RigidInertialExplicit())


def camera_scene(name: str = "camera_scene", mount: Any = None, gantry: bool = False) -> Experiment:
    """A steel ball moving along a tube seen by the in-loop camera, which sees the whole scene: the tube drawn, the ball
    read from its dynamics, a frame every two steps. With ``gantry``, the camera rides the carriage of the lab fixture's
    gantry, a drive moving it (a camera that may move, so the static structure is rendered in every frame)."""
    import math

    from mime.dynamics import RigidInertial
    from mime.observations import Camera, Detector, Mounted, World
    from mime.scene import Scene, Tube, X, Z
    from mime.things import Part
    from mime.units import kg_per_m3, um

    experiment = Experiment(name, scene=Scene.new(up=Z, units="m"), dt=ms(1))
    experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60), wall=um(10),
                                                            length=um(480))))
    experiment.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((-50.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((2e-3, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(20)))}))
    experiment.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    looking = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
    if gantry:
        from mime.things import Apparatus

        experiment.add_apparatus(Apparatus(
            "stage", chain={"gantry": importlib.import_module("mime_lab_fixture").Gantry(
                travel=X, at=m((0.0, -1e-3, 0.0)))},
            commands={"gantry.position": Ramp(m(0.0), to=m(1.0), rate=m(0.004) / s(1.0))}))
        mount = Mounted(frame="stage.gantry.carriage", orientation=looking)
    experiment.observe(Camera(
        detector=Detector(width=32, height=24, fx=150.0, fy=150.0, cx=15.5, cy=11.5, k1=-0.05),
        mount=mount or World(position=m((0.0, -1e-3, 0.0)), orientation=looking), frame_period=ms(2),
        sub_frames=2, absorbers=(("tube", 3000.0),)))
    return experiment


CASES = (
    Case("script_a_body", script_a_body),
    Case("script_a", script_a_with_drag),
    # Script A's body on drag 0.5.0's surrogate rung, unblended: drag 0.5.0's golden output (its first case) and the
    # rung's one round trip; after script A, so that the entries script A uses keep their golden outputs' case.
    Case("script_a_unblended", script_a_unblended),
    # Script A's surrogate on drag 0.4.0 under the name a folder saved before the near field carries: what such a
    # folder names (script A itself is on 0.4.0 since 0.5.0 retired the blend, so 0.4.0's golden output is its).
    Case("script_a_uncoupled", script_a_uncoupled_drag),
    # Script A's surrogate on drag 0.3.0, kept registered: what a folder saved before the separated quadrature names.
    Case("script_a_kept", script_a_kept_drag),
    Case("script_a_records", script_a_records),
    # FL-9 held by its hull's points from the first step, on the contact kept registered, 0.4.0: its golden outputs
    # come from it.
    Case("hull_contact", hull_contact_kept),
    # The same on the contact 0.5.0, its loops XLA loops: 0.5.0's golden outputs come from it.
    Case("hull_contact_looped", lambda: hull_contact("hull_contact_looped")),
    Case("coupled_toy", coupled_toy),
    # A medium's flow on the FVM, its periodic box driven by a pulsatile body force: IMEX Euler, and its steady limit.
    Case("periodic_tube_flow", periodic_tube_flow),
    Case("steady_tube_flow", lambda: periodic_tube_flow(steady=True)),
    # The same tube stepped by RK3, and its prescribed flow: the second scheme and the second flow field kind.
    Case("rk3_tube_flow", lambda: periodic_tube_flow(rk3=True)),
    Case("prescribed_tube_flow", prescribed_tube_flow),
    # The FVM on a tube tree's lumen, its wall the union's exact distance where the band reads it.
    Case("tree_flow", tree_flow),
    Case("open_tube_flow", open_tube_flow),
    # A law over the open vessel's two openings, and a closed valve at one (F2b's PR 3).
    Case("open_tube_loop", open_tube_loop),
    Case("closed_tube", closed_tube),
    Case("full_toy", full_toy),
    Case("walled_toy", walled_toy),
    Case("anchored_toy", anchored_toy),
    Case("twisted_toy", twisted_toy, refused=("vocabulary.reserved",)),
    Case("nudged_toy", nudged_toy, lab=True),
    # A body bound to a prim of a referenced robot asset, its magnet's direction read from the asset: the folder
    # carries the asset, and the fresh process reads it from there.
    Case("bound_robot", bound_robot),
    # The same robot with its parts declared: an ellipsoidal shell and the magnet inside it, placed and turned by the
    # asset's own transforms.
    Case("bound_robot_with_parts", bound_robot_with_parts),
    # A toy body probing the regions of the solid kinds, so that a step evaluates each of their shapes.
    Case("probed_shapes", probed_shapes),
    # A0's apparatus, kinematic: the AR4 arm from its URDF, a motor on its flange, a magnet on the rotor, the field's
    # torque and force on a body. The folder carries the URDF, and the fresh process reads it from there.
    Case("kinematic_rpm", kinematic_rpm_kept),
    # The lab fixture's gantry, a chain segment with no URDF, carrying a magnet: mounts work on it unchanged.
    Case("gantry_magnet", gantry_magnet),
    # The in-loop camera: the whole scene seen from the world, and from a gantry's carriage that a drive moves.
    Case("camera_scene", camera_scene),
    Case("camera_on_gantry", lambda: camera_scene("camera_on_gantry", gantry=True), lab=True),
    # A0's apparatus closed by its controller, a callback every 2 ms, its writes 1 ms late: the arm's joint positions
    # are the graph's one external input, which the run feeds.
    Case("tracked_rpm", tracked_rpm),
    # A0's apparatus on its dynamic rungs: the AR4 held at its home pose by the torques that balance its weight, the
    # motor on its flange driven by a torque, the field's torque and force on the body each delivering its reaction to
    # the rotor, and the rotor's to the flange a step late.
    Case("dynamic_rpm", dynamic_rpm),
    # A one-joint dynamic swing carrying a magnet, on the magnetic laws kept registered, 0.1.0, whose reactions no other
    # case builds.
    Case("swing_magnet_kept", swing_magnet_kept),
    # A0's dynamic apparatus with its masses mounted: the magnet's density and the motor's top speed declared, a camera
    # on link 3 and a stator part on the motor's base by their declared masses, each welded on its link, the rotor's
    # mass and mean inertia across its axis on the flange.
    Case("massive_rpm", lambda: massive_rpm(extra=True)),
    # The drag's MRS rung in an unbounded medium, on each kind it serves: a ball, a pill and a screw, on drag 0.4.0,
    # kept registered, each body its own node: its golden outputs are 0.4.0's.
    Case("mrs_free_space", mrs_free_space),
    # Drag 0.5.0's near field: a ball, a pill and a smaller ball close together in one medium, one joint node; its
    # golden outputs are 0.5.0's mrs rung's and the joint node's.
    Case("mrs_near_field", mrs_near_field),
    # The same past the takeover line: the joint system by GMRES, its report and its preconditioner's held inverses.
    Case("mrs_near_field_gmres", mrs_near_field_gmres),
    # The same on drag 0.3.0, kept registered: a point force at each sample, what a folder saved before the separated
    # quadrature names.
    Case("mrs_point_forces", mrs_point_forces),
    # The rung on the kept versions of the ball, the pill and the screw, whose samples are their quadratures': what a
    # folder saved before quasi-uniform sampling names.
    Case("mrs_kept", mrs_kept),
    # The rung on the solids that sample their surface from 7m-a: a cylinder, a box and an ellipsoid.
    Case("mrs_solids", mrs_solids),
    # The same rung in a straight tube, with the tube wall kind: it reads the corrected table, the asset
    # unit_tube_R0.997, which the run fetches into the cache before the tests (CI does it once per job).
    Case("mrs_tube", mrs_tube),
    # The same rung above a plane wall along a turned normal, with the plane wall kind: its regularised image, of the
    # drag's own blob.
    Case("mrs_plane", mrs_plane),
    # Graded sampling above the plane: a tilted cylinder's rims and its side held facing the wall, the wall's image
    # integrated near it, so that the graded step's own operations (the cells' sizes, the patch) are traced.
    Case("mrs_graded", mrs_graded),
    # The kept rigid_inertial 0.2.0 on a body of three different moments: what a folder saved with the explicit
    # gyroscopic term names.
    Case("rigid_explicit", rigid_explicit),
    # Script A with its frequency ramped and its phase lag observed: the ramp's and the lag's golden outputs.
    Case("script_a_ramped", script_a_ramped),
    # The drag's rft rung in an unbounded medium, on each kind with a centreline: a helix, a capsule and a prolate
    # ellipsoid, each moving and turning; the helix's golden output comes from it.
    Case("rft_free_space", rft_free_space),
    # The same rung with the tube wall kind, on a helix small against its tube (definite with the wall): it reads the
    # corrected table, the asset unit_tube_R0.997.
    Case("rft_tube", rft_tube),
    # rft_free_space on drag 0.4.0, kept: its rft rung's golden output, the version the rung was added to first.
    Case("rft_kept", rft_kept),
    # A body in an electromagnet array's field, the calibrated multipole, placed, turned and with a background.
    Case("coil_array", coil_array),
    # A point body coupled two-way to the tube's flow through the exchange law, drag 0.5.0's stokes rung: after the
    # cases that use the sphere first, so that its golden output stays probed_shapes'.
    Case("point_body_flow", point_body_flow),
    # The lab fixture's geometry kind, a ball with its own quadrature, as a body's hull under the drag's MRS rung: last,
    # so that the rung's and free space's golden outputs stay mrs_free_space's.
    Case("lab_bead", lab_bead, lab=True),
    # Script A metered by the energy ledger: last, so that every other entry's golden output stays its earlier case's.
    Case("script_a_metered", script_a_metered),
)

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry has no case: the planted run drops the only case using
    # toy.anchored_body.
    CASES = tuple(case for case in CASES if case.name != "anchored_toy")
