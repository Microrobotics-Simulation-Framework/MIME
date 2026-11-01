"""The round-trip cases the registry walk requires: together they use every registered entry, MIME's toys and the
lab fixture's, and every kind of thing. An entry registered without a case here fails the walk.

A case whose build is refused (an entry that exists to be refused) round-trips its document instead: written, read
back in a fresh process, and refused there for the same reasons.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from mrs_cases import free_space, mrs, mrs_free_space, mrs_kept, mrs_solids, mrs_tube
from chain_cases import dynamic_rpm, gantry_magnet, kinematic_rpm, massive_rpm, tracked_rpm
from robot_asset import bound_robot, bound_robot_with_parts
from script_a import ALL_RECORDS, fl9, script_a

from mime.dynamics.body import RigidInertialExplicit
from mime.experiment import Experiment
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
from mime.things import Body, Medium, State, Structure
from mime.units import N, kg, kg_m2, m, m_per_s, mm, rad_per_s


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[], Experiment]
    refused: tuple[str, ...] = ()
    #: Whether it uses the lab fixture's entries, which every CI lane installs and a local run may not.
    lab: bool = False


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
    acts within the walk's first steps: 4b's case, which the body side's golden outputs come from."""
    return script_a("script_a_body", velocity=m_per_s((1e-3, 0.0, 0.0)), drag=False)


def script_a_with_drag() -> Experiment:
    """Script A, with the drag surrogate and its lubrication, the body moving as in the body side's case."""
    return script_a(velocity=m_per_s((1e-3, 0.0, 0.0)))


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


def rigid_explicit() -> Experiment:
    """A ball of three different moments, moving and spinning at 200 rad/s in water, on the kept ``rigid_inertial``
    0.2.0 (the gyroscopic term explicit, MIME-ANO-129) with the drag's MRS rung: what a folder saved before the term
    was made implicit names, and the golden of that version, which gains energy where 0.3.0's does not."""
    return free_space("rigid_explicit", {"ball": Sphere(radius=mm(1.0))}, mrs(mm(1.5)), mass=kg(1e-6),
                      inertia=kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), velocity=m_per_s((1e-3, -2e-3, 0.5e-3)),
                      spin=rad_per_s((150.0, 100.0, -100.0)), dynamics=RigidInertialExplicit())


CASES = (
    Case("script_a_body", script_a_body),
    Case("script_a", script_a_with_drag),
    Case("script_a_records", script_a_records),
    Case("coupled_toy", coupled_toy),
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
    Case("kinematic_rpm", kinematic_rpm),
    # The lab fixture's gantry, a chain segment with no URDF, carrying a magnet: mounts work on it unchanged.
    Case("gantry_magnet", gantry_magnet),
    # A0's apparatus closed by its controller, a callback every 2 ms, its writes 1 ms late: the arm's joint positions
    # are the graph's one external input, which the run feeds.
    Case("tracked_rpm", tracked_rpm),
    # A0's apparatus on its dynamic rungs: the AR4 held at its home pose by the torques that balance its weight, the
    # motor on its flange driven by a torque, the field's torque and force on the body each delivering its reaction to
    # the rotor, and the rotor's to the flange a step late.
    Case("dynamic_rpm", dynamic_rpm),
    # A0's dynamic apparatus with its masses mounted: the magnet's density and the motor's top speed declared, a camera
    # on link 3 and a stator part on the motor's base by their declared masses, each welded on its link, the rotor's
    # mass and mean inertia across its axis on the flange.
    Case("massive_rpm", lambda: massive_rpm(extra=True)),
    # The drag's MRS rung in an unbounded medium, on each kind it serves: a ball, a pill and a screw.
    Case("mrs_free_space", mrs_free_space),
    # The rung on the kept versions of the ball, the pill and the screw, whose samples are their quadratures': what a
    # folder saved before quasi-uniform sampling names.
    Case("mrs_kept", mrs_kept),
    # The rung on the solids that sample their surface from 7m-a: a cylinder, a box and an ellipsoid.
    Case("mrs_solids", mrs_solids),
    # The same rung in a straight tube, with the tube wall kind: it reads the corrected table, the asset
    # unit_tube_R0.997, which the run fetches into the cache before the tests (CI does it once per job).
    Case("mrs_tube", mrs_tube),
    # The kept rigid_inertial 0.2.0 on a body of three different moments: what a folder saved with the explicit
    # gyroscopic term names.
    Case("rigid_explicit", rigid_explicit),
    # The lab fixture's geometry kind, a ball with its own quadrature, as a body's hull under the drag's MRS rung: last,
    # so that the rung's and free space's golden outputs stay mrs_free_space's.
    Case("lab_bead", lab_bead, lab=True),
)

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry has no case: the planted run drops the only case using
    # toy.anchored_body.
    CASES = tuple(case for case in CASES if case.name != "anchored_toy")
