"""The round-trip cases the registry walk requires: together they use every registered entry, MIME's toys and the
lab fixture's, and every kind of thing. An entry registered without a case here fails the walk.

A case whose build is refused (an entry that exists to be refused) round-trips its document instead: written, read
back in a fresh process, and refused there for the same reasons.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from bem_cases import bem_free_space
from chain_cases import gantry_magnet, kinematic_rpm, tracked_rpm
from robot_asset import bound_robot, bound_robot_with_parts
from script_a import ALL_RECORDS, fl9, script_a

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
from mime.units import N, m, m_per_s, mm


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[], Experiment]
    refused: tuple[str, ...] = ()


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
    """The lab fixture's entry beside MIME's toys: its lock names its own distribution."""
    nudge = importlib.import_module("mime_lab_fixture").Nudge
    experiment = coupled_toy()
    experiment.attach(nudge(force=N((0.0, 0.01, -0.04))), body="probe")
    return experiment


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


CASES = (
    Case("script_a_body", script_a_body),
    Case("script_a", script_a_with_drag),
    Case("script_a_records", script_a_records),
    Case("coupled_toy", coupled_toy),
    Case("full_toy", full_toy),
    Case("walled_toy", walled_toy),
    Case("anchored_toy", anchored_toy),
    Case("twisted_toy", twisted_toy, refused=("vocabulary.reserved",)),
    Case("nudged_toy", nudged_toy),
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
    # The drag's boundary-element rung in an unbounded medium, on each kind it serves: a ball, a pill and a screw.
    Case("bem_free_space", bem_free_space),
)

if os.environ.get("MIME_CI_PLANT_FAILURE") == "1":
    # The control that the walk fails CI when an entry has no case: the planted run drops the only case using
    # toy.anchored_body.
    CASES = tuple(case for case in CASES if case.name != "anchored_toy")
