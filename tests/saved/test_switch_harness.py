"""The switch harness (``mime.testing``) over every switch MIME declares: each owner has a row, placed a quarter of a
step from its switch on script A's nodes (the body confined in its tube, the drag, the swim speed) or on the cylinder
itself, and measures within its declared order, its indicator at each placed point being that point's offset. Every
row runs in float64: the forced-branch difference needs no finite difference, and the placed points lie from 2.5e-12
m (``contact.band``'s, at delta / 10) to 2.5e-4 m from their switches, outside the contact band (about 1e-16 m) and
the axis band (3e-15 m), or inside one on purpose. The float32 bands (the contact's about 1e-7 m, the tube's 1.6 um) and
what a step does inside a band are not measured here. The dynamic arm's stops and Coulomb friction are placed on a
strongly coupled two-link arm (``multibody_cases.coupled_planar``) with no gravity or damping, one joint with the
switch, so that every evaluation of its owner over the sweeps is at the row's offset: the elbow, its only bounded joint,
turning into its stop and ending the step at the row's offset from it; and the shoulder, its only joint with friction, at
rest under a torque that needs just under its friction (0.5 N m) to hold it with the elbow free. The stops' rows also
measure the jump itself: the forced branches' velocities differ by the stop's impulse, every joint moved by
``A^-1 e_2 / (A^-1)_22`` times the stopping joint's change, not only within an order. The start of the sweeps,
``arm.stick``, is placed on the same arm, the elbow with friction and the shoulder driven past a stop that alone
brings the elbow to rest, so that the start lies away from the solution and reaches the velocities through the
shoulder's first update and what the sweeps leave after it (on an arm that sticks everywhere the start is the solution,
and no output shows it).

A row of a switch no valid state reaches (``cylinder.axis_norm``'s axis of zero length, ``drag.room``'s and
``drag.clamp_divisor``'s tube the body does not fit) measures its order at a forced branch outside the valid states:
the law's order there, not one a run meets. The arm's limits are placed on A0's apparatus, kinematic. ``contact.scale``'s row measures nothing, as its
declaration says.

Each row's ``delta`` keeps its smooth part small beside the jump in question (the kink limit's resolution,
``mime.testing``). Where a row's jump is what it measures, the smooth part is below every output's floor where the
branch difference does not move with the offset (``contact.rule``, ``cylinder.axis``'s normal, ``cylinder.axis_norm``,
``drag.across``, ``drag.end``, ``swim.window``); in ``contact.band``, ``drag.axis`` and ``swim.mean`` it grows with
the offset, and the floor would need a point inside the switch's band or below the placement's own rounding, so it is
held at 1.4e-4 of the jump or less.

The solids' shapes have rows of their own, each placed on the shape itself: the ball's centre, the capsule's segment
and end planes, the slab's mid-plane, the screw's seam, whose indicator (the gap between its two nearest roots) a
scalar root find on the distance across the axis places at each offset, and the screw's focus, placed along the ray
through a crest from its centre of curvature, where the seam ends. A declared order of 1 or more (the capsule's end
planes, and the focus in the distance) is checked as at least 0: the harness tells a jump from anything continuous,
and the orders above are the theory's (the focus's are measured in ``test_shape_suite.py``). The focus's row measures
nothing, as its declaration says: its two branches take the root, polished to rounding, with the Newton step or
without it, and differ in the derivative only.

The kink limit is measured: fitted on ``drag.clamp`` (a kink) and ``drag.end`` (a jump), then checked on two
independent planted rows, an analytic kink and an analytic jump, and on two mixed rows that state its resolution: a
jump of 0.4 of the smooth part's growth over a quarter step reads as one, a jump of 0.04 of it reads continuous. The
controls: ``drag.end`` declared a kink fails, ``drag.clamp``'s continuity passes, branches that coincide away from
the switch measure continuous, a jump in a small output beside a large one is measured, a step traced before the
forced setting opened fails, ``drag.clamp_divisor`` written as a jump fails, and a row that measures nothing (or one
that measures something) fails where its declaration says otherwise. The domain guards: at points where each guard's
condition is false, replacing every fallback (twice and half its value, or 2 and 0.5 for a zero) changes no output
and no derivative; every guard the switch scan lists is reached with its condition false, and every ``nonzero_norm``
reached and never at zero; a planted guard whose fallback reaches an output fails, and so does ``_unit``'s old
selection, whose fallback reaches the derivative."""

import dataclasses
import math
from functools import cache
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tube_asset
from mrs_cases import mrs_free_space, mrs_solids
import multibody_cases
from chain_cases import AR4_BASE, AR4_HOME, kinematic_rpm, tracked_rpm
from script_a import script_a
from hull_cases import MARGIN as HULL_MARGIN, TUBE_RADIUS, hull_node, kind_points, rotation

from switch_configurations import PACKAGE, tree_scan

import mime.nodes.hydro._surrogate_drag as surrogate_drag
from mime.build._in_graph import inlined
from mime.build._maddening import node_state, parameter_leaves
from mime.control import control_context
from mime.control._arm_tracking import _outer
from mime.core.multibody import frame_pose, mass_matrix
from mime.core.tables import Axis, Table, evaluate
from mime.core import CAPSULE, CYLINDER, SCREW, SLAB, SPHERE, quaternion, shape_names, shape_switches, switch
from mime.scene import Cylinder, Ellipsoid, Screw, Sphere
from mime.units import dimensionless, mm
from mime.docs import load_namespaces
from mime.nodes.hydro.mrs import frame_across, tube_correction, tube_reads, tube_table_from
from mime.nodes.hydro.mrs._tube import pair_block
from mime.registry import EntryKind, declared_switches, lookup
from mime.nodes.actuation import DynamicArmNode
from mime.testing import KINK_LIMIT, SwitchRow, classify, fallback_problems
from mime.testing._switches import FLOOR_ULPS
from mime.testing._shape_suite import _screw_focus_place

DT = 5e-4
#: Where the body sits along the tube's axis (+x): script A's start.
ALONG = 0.01
RADIUS, EXTENT, MARGIN = 3.175e-3, 2.0748e-3, 1e-4
#: The radius of the surface the body's centre is confined to: the tube's, less its extent and margin.
SHRUNK = RADIUS - EXTENT - MARGIN
REFERENCE, OUTER = 1.56e-3, 1.33
#: A body's outward speed where a row needs the hold to engage, m/s.
OUTWARD = 1e-3


@pytest.fixture(scope="module")
def graph():
    return script_a().build().graph


def declarations() -> dict[str, switch.Declaration]:
    load_namespaces()
    found = {d.owner: d for name in shape_names() for d in shape_switches(name)}
    for kind, name, version in (("interaction", "contact", "0.1.0"), ("interaction", "contact", "0.2.0"),
                                ("interaction", "drag", "0.3.0"),
                                ("observation", "swim_speed", "0.1.0"), ("dynamics", "rigid_inertial", "0.2.0"),
                                ("dynamics", "arm", "0.1.0"), ("controller", "arm_tracking", "0.1.0"),
                                ("wall kind", "tube", "0.1.0"), ("geometry", "cylinder", "0.1.0")):
        record = lookup(EntryKind(kind), name, version)
        assert record is not None
        found.update({d.owner: d for d in declared_switches(record.cls)})
    return found


def leaves(graph, name: str, **changed) -> dict:
    return {**parameter_leaves(graph)[name], **{key: jnp.asarray(value) for key, value in changed.items()}}


# -- the body, confined to script A's tube -------------------------------------------------------------------------

def body_at(graph, position, velocity=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), coupled=False):
    """The body's outputs after one step from ``position`` and ``velocity`` with no load, and with the drag's
    resistance at that pose where ``coupled`` (a resistance coupling translation and rotation, so that holding the
    contact changes more than the normal speed). ``velocity`` is the speed the solve leaves the body with, from which
    the starting speed is found by the step's own equation, ``V = V' + dt M^-1 R V'``."""
    node = graph.get_node("umr")
    pose = jnp.asarray((*position, *orientation))
    resistance = jnp.zeros((6, 6))
    if coupled:
        drag = graph.get_node("drag:umr")
        resistance = drag.fresh_update(node_state(graph, "drag:umr"), {"pose": pose}, DT,
                                       params=leaves(graph, "drag:umr"))["resistance"]
    after = jnp.asarray((*velocity, 0.0, 0.0, 0.0))
    mass = parameter_leaves(graph)["umr"]["mass"]
    inertia = parameter_leaves(graph)["umr"]["inertia"]
    inverse_mass = jnp.concatenate([jnp.full(3, 1.0 / mass), 1.0 / inertia])
    start = after + DT * inverse_mass * (resistance @ after)
    out = node.fresh_update({"pose": pose, "twist": start},
                            {"applied_wrench": jnp.zeros(6), "resistance": resistance, "background_load": jnp.zeros(6)},
                            DT, params=leaves(graph, "umr"))
    return {"pose": out["pose"], "twist": out["twist"]}


def below(depth: float) -> tuple[float, float, float]:
    """A point ``depth`` past the shrunk surface (negative: inside it), below the axis, where the outward normal is
    -z."""
    return (ALONG, 0.0, -(SHRUNK + depth))


def tilted(angle: float) -> tuple[float, float, float, float]:
    """The body turned by ``angle`` about z: its x axis at ``cos(angle)`` to the tube's."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def rolled_then_tilted(angle: float) -> tuple[float, float, float, float]:
    """The body turned a quarter turn about x, then by ``angle`` about z: its x axis at ``cos(angle)`` to the tube's
    and its y axis along z, so that the two axes' parts across the tube are perpendicular."""
    quarter = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    return tuple(float(c) for c in quaternion.multiply(jnp.asarray(tilted(angle)), jnp.asarray(quarter)))


#: The hold ends the step on the surface, where the rule's own switch sits within its rounding: a row that engages
#: the hold keeps the rule's branch fixed, so a rounding flip of it does not read as this switch's.
ON_THE_SURFACE = {"contact.rule": False}


def body_rows(graph, declared) -> list[SwitchRow]:
    down = (0.0, 0.0, -OUTWARD)
    return [
        SwitchRow(declared["contact.band"], lambda o: body_at(graph, below(o), down, coupled=True), 1e-10, side=-1,
                  held=ON_THE_SURFACE),
        SwitchRow(declared["contact.outward"], lambda o: body_at(graph, below(0.0), (0.0, 0.0, -o), coupled=True),
                  1e-4, held=ON_THE_SURFACE),
        SwitchRow(declared["contact.hold"], lambda o: body_at(graph, below(0.0), (0.0, 0.0, -o), coupled=True), 1e-4,
                  held=ON_THE_SURFACE),
        SwitchRow(declared["contact.closing"], lambda o: body_at(graph, below(-o), down, coupled=True), 1e-8,
                  held={**ON_THE_SURFACE, "contact.band": True}),
        SwitchRow(declared["contact.scale"], lambda o: body_at(graph, (0.0, 0.0, -o)), 1e-6),
        SwitchRow(declared["contact.project"], lambda o: body_at(graph, below(o)), 1e-6),
        SwitchRow(declared["contact.rule"], lambda o: body_at(graph, below(o - OUTWARD * DT), down), 1e-6),
        SwitchRow(declared["contact.rule_outward"],
                  lambda o: body_at(graph, below(1e-7 - o * DT), (0.0, 0.0, -o)), 1e-4),
    ]


# -- the cylinder --------------------------------------------------------------------------------------------------

def cylinder_at(point, axis=(1.0, 0.0, 0.0)):
    parameters = {"centre": jnp.zeros(3), "axis": jnp.asarray(axis), "radius": jnp.asarray(RADIUS)}
    held = jnp.asarray(point)
    return {"distance": CYLINDER.signed_distance(parameters, held), "normal": CYLINDER.outward_normal(parameters, held),
            "curvature": CYLINDER.curvature(parameters, held)}


def rod_support_at(direction):
    """The rod's support (the cylinder kind's closed form) along ``direction`` from a sample at its +x end: its point
    and its derivative in the direction, the capability's two outputs."""
    held = ROD.support({"radius": jnp.asarray(5e-4), "length": jnp.asarray(2e-3), "axis": (1.0, 0.0, 0.0)},
                       jnp.asarray(direction, dtype=jnp.float64), jnp.asarray((1e-3, 0.0, 5e-4)))
    return {"support": jnp.concatenate([held.point, held.jacobian.ravel()])}


def cylinder_rows(declared) -> list[SwitchRow]:
    """The cylinder's axis, placed on the shape; and its support's, a direction along the rod's axis turned off it by
    the row's offset (its part across the axis, the indicator), placed on the kind's closed form."""
    return [
        SwitchRow(declared["cylinder.axis"], lambda o: cylinder_at((ALONG, 0.0, -o)), 1e-6),
        SwitchRow(declared["cylinder.axis_norm"], lambda o: cylinder_at((ALONG, 3e-4, -1e-3), axis=(o, 0.0, 0.0)),
                  1e-10),
        SwitchRow(declared["cylinder.support"], lambda o: rod_support_at((1.0, 0.0, o)), 1e-8),
    ]


# -- the solids' shapes -------------------------------------------------------------------------------------------

#: A ball, a capsule along x (its segment from -1 to 1 mm), a slab across x and FL-9's screw, each about its own origin.
BALL = {"centre": jnp.zeros(3), "radius": jnp.asarray(1e-3)}
PILL = {"centre": jnp.zeros(3), "axis": jnp.asarray((1.0, 0.0, 0.0)), "radius": jnp.asarray(5e-4),
        "height": jnp.asarray(2e-3)}
PLATES = {"centre": jnp.zeros(3), "axis": jnp.asarray((1.0, 0.0, 0.0)), "half_thickness": jnp.asarray(5e-4)}
SCREW_PARAMETERS = {"centre": jnp.zeros(3), "orientation": jnp.asarray((1.0, 0.0, 0.0, 0.0)), "radius": jnp.asarray(1.56e-3),
         "length": jnp.asarray(7.47e-3), "wavenumber": jnp.asarray(2.33), "modulation": jnp.asarray(0.33), "starts": 2}
#: Where the screw's seam is probed: on its axis 1 mm from its centre, moved across it along a direction off the
#: troughs' line and off the lobes', so that one trough is nearer.
SEAM_AT, SEAM_ACROSS = np.array([1e-3, 0.0, 0.0]), np.array([0.0, 0.6, 0.8])


def shape_at(shape, parameters, point):
    held = jnp.asarray(point, dtype=jnp.float64)
    found = {"distance": shape.signed_distance(parameters, held), "normal": shape.outward_normal(parameters, held)}
    if hasattr(shape, "tangent"):
        found["tangent"] = shape.tangent(parameters, held)
    return found


def seam_gap(across: float) -> float:
    """The seam's indicator at a point ``across`` metres off the screw's axis: the gap between its two nearest roots."""
    with switch.collecting() as rows:
        SCREW.signed_distance(SCREW_PARAMETERS, jnp.asarray(SEAM_AT + across * SEAM_ACROSS))
    return min(abs(float(row.indicator)) for row in rows if row.owner == "screw.seam")


def at_the_seam(o: float):
    """The screw's outputs at the point whose seam indicator is ``o``: a scalar root find on the distance across the
    axis, outside the step, on the gap's slope there."""
    across = o / (seam_gap(1e-7) / 1e-7)
    for _ in range(3):
        across *= o / seam_gap(across)
    return shape_at(SCREW, SCREW_PARAMETERS, SEAM_AT + across * SEAM_ACROSS)


#: Where the screw's focus is probed: the ray through a crest from its centre of curvature, 1 mm along the axis from
#: the screw's centre, as the seam is.
FOCUS_DRAW = np.array([0.5 + 1e-3 / 7.47e-3, 0.2, 0.0])


#: FL-9's screw turned off every axis for the focus's row, so that no output there has a component near zero, whose
#: floor (units in the last place of its own size) a rounding difference would cross.
TURNED = np.array([0.9, 0.3, -0.2, 0.25]) / np.linalg.norm([0.9, 0.3, -0.2, 0.25])
TURNED_SCREW = {**SCREW_PARAMETERS, "orientation": jnp.asarray(TURNED)}


def at_the_focus(o: float):
    """The turned screw's outputs where its focus's indicator is ``o``: along the ray through a crest from its centre
    of curvature, towards the crest."""
    local = _screw_focus_place(SCREW_PARAMETERS, o, FOCUS_DRAW)
    return shape_at(SCREW, TURNED_SCREW, quaternion.rotation_matrix(TURNED_SCREW["orientation"]) @ local)


def solid_rows(declared) -> list[SwitchRow]:
    tilted_unit = np.array([0.6, 0.0, 0.8])
    return [
        SwitchRow(declared["sphere.centre"], lambda o: shape_at(SPHERE, BALL, o * tilted_unit), 1e-6),
        SwitchRow(declared["capsule.segment"], lambda o: shape_at(CAPSULE, PILL, (3e-4, 0.0, o)), 1e-6),
        SwitchRow(declared["capsule.end.high"], lambda o: shape_at(CAPSULE, PILL, (1e-3 - o, 2e-4, 0.0)), 1e-6),
        SwitchRow(declared["capsule.end.low"], lambda o: shape_at(CAPSULE, PILL, (-1e-3 + o, 2e-4, 0.0)), 1e-6),
        SwitchRow(declared["slab.mid"], lambda o: shape_at(SLAB, PLATES, (o, 2e-4, -1e-4)), 1e-6),
        SwitchRow(declared["screw.seam"], at_the_seam, 1e-6),
        SwitchRow(declared["screw.focus"], at_the_focus, 1.56e-3),
    ]


# -- the drag ------------------------------------------------------------------------------------------------------

def drag_at(graph, position, orientation=(1.0, 0.0, 0.0, 0.0), **changed):
    node = graph.get_node("drag:umr")
    return node.fresh_update(node_state(graph, "drag:umr"), {"pose": jnp.asarray((*position, *orientation))}, DT,
                             params=leaves(graph, "drag:umr", **changed))


def largest(tube: float) -> float:
    """The largest offset the surrogate is read at, in the body's radius, for a tube of ``tube`` body radii."""
    return (tube - OUTER) * 0.95


#: A tube the body does not fit, whose largest offset, 5e-11 body radii, lies below the clamp's divisor's floor: the
#: one place the divisor's branch reaches the resistance.
UNFIT = REFERENCE * (OUTER + 5e-11 / 0.95)


def drag_rows(graph, declared) -> list[SwitchRow]:
    tube = RADIUS / REFERENCE
    outer = OUTER * REFERENCE
    wide = 2 * outer + 5e-4

    def room(o):
        return drag_at(graph, (ALONG, 0.0, -2 * o * REFERENCE), vessel_radius=REFERENCE * (OUTER + o / 0.95))

    def feature_floor(o):
        narrow = OUTER + (1e-3 + o) / 0.05
        return drag_at(graph, (ALONG, 0.0, -2 * largest(narrow) * REFERENCE), vessel_radius=REFERENCE * narrow)

    def log_floor(o):
        return drag_at(graph, (ALONG, 0.0, -(wide - outer - outer * math.exp(-o))), vessel_radius=wide,
                       lubrication_epsilon=1e-2)

    return [
        SwitchRow(declared["drag.axis"], lambda o: drag_at(graph, (ALONG, 0.0, -o)), 1e-9),
        SwitchRow(declared["drag.across"], lambda o: drag_at(graph, (ALONG, 0.0, 0.0),
                                                             rolled_then_tilted(math.asin(math.sqrt(0.25 + o)))),
                  1e-3),
        SwitchRow(declared["drag.end"], lambda o: drag_at(graph, (ALONG, 0.0, -5e-4), tilted(math.acos(o - 1e-3))),
                  1e-3),
        SwitchRow(declared["drag.room"], room, 1e-3),
        SwitchRow(declared["drag.clamp"], lambda o: drag_at(graph, (ALONG, 0.0, -(largest(tube) + o) * REFERENCE)),
                  1e-3),
        SwitchRow(declared["drag.clamp_divisor"],
                  lambda o: drag_at(graph, (ALONG, 0.0, -(1e-10 + o) * REFERENCE), vessel_radius=UNFIT), 4e-11),
        SwitchRow(declared["drag.feature_gap_floor"], feature_floor, 1e-4),
        SwitchRow(declared["drag.gap_floor"], lambda o: drag_at(graph, (ALONG, 0.0, -(RADIUS - outer - 1e-9 - o))),
                  4e-10),
        SwitchRow(declared["drag.log_floor"], log_floor, 1e-3),
    ]


# -- the swim speed ------------------------------------------------------------------------------------------------

def swim_at(graph, *, steps=10, distance=2e-6, observed=1e-3, after=1.0):
    node = graph.get_node("swim_speed:umr")
    state = {"speed": jnp.asarray(0.0), "mean_speed": jnp.asarray(0.0), "distance": jnp.asarray(distance),
             "observed": jnp.asarray(observed), "steps": jnp.asarray(steps, dtype=jnp.int32)}
    return node.fresh_update(state, {"twist": jnp.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0))}, DT,
                             params=leaves(graph, "swim_speed:umr", after=after))


def swim_rows(graph, declared) -> list[SwitchRow]:
    return [
        SwitchRow(declared["swim.window"], lambda o: swim_at(graph, after=11 * DT - o), 1e-5),
        SwitchRow(declared["swim.mean"], lambda o: swim_at(graph, observed=DT / 2 + o), 1e-7),
    ]


# -- the chain: the arm's joint limits and the dipole, on A0's apparatus --------------------------------------

@cache
def chain_graph():
    """A0's apparatus, kinematic (``chain_cases.kinematic_rpm``), built in float64 as the harness runs."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return kinematic_rpm("harness_rpm").build().graph
    finally:
        jax.config.update("jax_enable_x64", held)


def arm_at(graph, command) -> dict:
    """The arm's outputs after one step commanded to ``command``, no servo. Every joint of the AR4 is bounded, and the
    clip's switch is one over all of them: forced, it holds every joint at its bound, so a row places every joint the
    same offset from its bound."""
    node = graph.get_node("rpm.arm")
    out = node.fresh_update(node_state(graph, "rpm.arm"), {"joint_positions": jnp.asarray(command)}, DT,
                            params=leaves(graph, "rpm.arm"))
    return {"joint_positions": out["joint_positions"], "frame_poses": out["frame_poses"]}


#: Where the magnet sits for its row: its mount's rotor frame at the origin, unturned.
AT_THE_ORIGIN = jnp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2)


def dipole_at(graph, point) -> dict:
    """The dipole's field and gradient at ``point``, the magnet at the origin, for each of its reader slots."""
    node = graph.get_node("rpm.magnetic")
    slots = [name.removeprefix("points_") for name in node.boundary_input_spec() if name.startswith("points_")]
    given = {"mount_poses": AT_THE_ORIGIN, **{f"points_{slot}": jnp.asarray((*point, 1.0, 0.0, 0.0, 0.0))
                                               for slot in slots}}
    out = node.fresh_update(node_state(graph, "rpm.magnetic"), given, DT, params=leaves(graph, "rpm.magnetic"))
    return {"flux_density": jnp.stack([out[f"flux_density_{slot}"] for slot in slots]),
            "gradient": jnp.stack([out[f"gradient_{slot}"] for slot in slots])}


#: The dynamic arm's rows' tree: two strongly coupled links (``multibody_cases.coupled_planar``).
COUPLED = multibody_cases.coupled_planar()
#: The stops' rows bound the second link's joint at +-1 rad; the friction's give the first joint 0.5 N m.
STOP_LIMIT, FRICTION = 1.0, 0.5
#: The velocity the stops' rows turn the arm at, the second joint into its stop.
STOP_VELOCITY = np.array([0.5, 1.0])


def impulse_node(tree, friction, limits=None) -> DynamicArmNode:
    """``tree``'s dynamic arm with no damping and no weight, ``friction`` per joint, and ``limits`` (lower and upper,
    per joint, infinite where none), built in float64."""
    a = tree.articulation
    described = DynamicArmNode.described(a, [0.0] * a.dofs, list(friction))
    lower, upper = limits if limits is not None else (np.full(a.dofs, -np.inf), np.full(a.dofs, np.inf))
    bounded = tuple(i for i in range(a.dofs) if np.isfinite(lower[i]) or np.isfinite(upper[i]))
    described.update(bounded=bounded, lower=tuple(float(lower[i]) for i in bounded),
                     upper=tuple(float(upper[i]) for i in bounded))
    return DynamicArmNode("arm", DT, **described, placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                          gravity=(0.0, 0.0, 0.0), initial_joint_positions=(0.0,) * a.dofs)


def impulse_at(node: DynamicArmNode, positions, velocity) -> dict:
    """The node's outputs after one step from rest at ``positions`` under the torques that give it ``velocity`` with
    no stop or friction, ``M velocity / dt`` (at rest nothing else acts), traced afresh."""
    q = jnp.asarray(positions, dtype=jnp.float64)
    torques = mass_matrix(node.articulation(node._step_constants(None), q.dtype), q) @ jnp.asarray(velocity) / DT
    state = {**node.initial_state(), "joint_positions": q, "joint_velocities": jnp.zeros_like(q)}
    out = node.fresh_update(state, {"joint_torques": torques}, DT)
    return {"joint_positions": out["joint_positions"], "joint_velocities": out["joint_velocities"],
            "frame_poses": out["frame_poses"]}


def stop_node() -> DynamicArmNode:
    return impulse_node(COUPLED, (0.0, 0.0), (np.array([-np.inf, -STOP_LIMIT]), np.array([np.inf, STOP_LIMIT])))


def at_a_stop(o: float, side: int) -> dict:
    """The second joint ``o`` inside its lower stop (``side`` -1) or its upper one (+1) at the end of the step,
    turning into it at ``STOP_VELOCITY``: the only joint with a stop, so every evaluation of its switch is placed."""
    velocity = STOP_VELOCITY * np.array([1.0, side])
    return impulse_at(stop_node(), (0.3, side * (STOP_LIMIT - o) - DT * velocity[1]), velocity)


def at_friction(o: float, side: int) -> dict:
    """At rest, under a torque on the first joint, the only one with friction, that needs ``FRICTION - o`` of it
    (``side`` +1; ``-FRICTION + o`` with -1) to hold it with the other joint free: at rest the torque itself."""
    node = impulse_node(COUPLED, (FRICTION, 0.0))
    q = jnp.asarray((0.3, 0.5))
    inverse = jnp.linalg.inv(mass_matrix(COUPLED, q))
    return impulse_at(node, q, DT * inverse @ jnp.asarray((side * (FRICTION - o), 0.0)))


#: The start's row: the coupled arm's shoulder bounded above, its elbow with friction at a small angle, where the two
#: joints are most strongly coupled, and the gap the shoulder's stop leaves it over the step, as a speed (rad/s).
START_FRICTION, START_LIMIT, START_ELBOW, START_GAP = 0.5, 0.3, 0.2, 0.5


def at_the_start(o: float) -> dict:
    """From rest, the shoulder driven past its stop, the elbow (the only joint with friction) under a torque whose
    impulse holding it still with the shoulder free is ``sqrt(1 - o)`` of its friction's, so that the start's one
    evaluation, its largest so far, is placed ``o`` from it. Each joint's update sets its own impulse from the
    others', so a joint's start reaches the result only through the joints updated before it in the first sweep: here
    the shoulder's stop. With that stop engaged the elbow needs no friction at all (its velocity is chosen so), so the
    sweeps leave a remainder at the coupled pair's rate, through which the start reaches the velocities (on an arm
    that sticks everywhere the start is the solution, and no output shows it). The forced branch starts the sweeps
    ``1 / (1 - o)`` beyond it."""
    node = impulse_node(COUPLED, (0.0, START_FRICTION), (np.array([-np.inf, -np.inf]), np.array([START_LIMIT, np.inf])))
    q = jnp.asarray((START_LIMIT - DT * START_GAP, START_ELBOW), dtype=jnp.float64)
    inverse = np.linalg.inv(np.asarray(mass_matrix(COUPLED, q)))
    elbow = math.copysign(inverse[1, 1] * DT * START_FRICTION * math.sqrt(1.0 - o), inverse[0, 1])
    # The shoulder's speed past its gap at which the stop's impulse brings the elbow to rest by itself.
    shoulder = START_GAP + inverse[0, 0] * elbow / inverse[0, 1]
    return impulse_at(node, q, (shoulder, elbow))


def chain_rows(declared) -> list[SwitchRow]:
    graph = chain_graph()
    lower, upper = (jnp.asarray(parameter_leaves(graph)["rpm.arm"][bound]) for bound in ("lower", "upper"))
    return [
        SwitchRow(declared["arm.limit.low"], lambda o: arm_at(graph, lower + o), 1e-3),
        SwitchRow(declared["arm.limit.high"], lambda o: arm_at(graph, upper - o), 1e-3),
        SwitchRow(declared["arm.stop.low"], lambda o: at_a_stop(o, -1), 1e-3),
        SwitchRow(declared["arm.stop.high"], lambda o: at_a_stop(o, 1), 1e-3),
        SwitchRow(declared["arm.friction.low"], lambda o: at_friction(o, -1), 1e-3),
        SwitchRow(declared["arm.friction.high"], lambda o: at_friction(o, 1), 1e-3),
        SwitchRow(declared["arm.stick"], at_the_start, 1e-3),
    ]


@pytest.mark.parametrize("side", [-1, 1], ids=["arm.stop.low", "arm.stop.high"])
def test_each_stops_row_moves_every_joint_by_the_stops_impulse(side):
    """The stop's forced-branch difference in the velocities at its row's point is its impulse's: the stopping
    joint's velocity changed from the one that leaves it ``o`` inside its stop to the one that brings it there,
    ``side o / dt``, and every joint by ``A^-1 e_2 / (A^-1)_22`` times that; not only an order."""
    o = 1e-3 / 4
    owner = "arm.stop.low" if side < 0 else "arm.stop.high"
    with switch.forcing({owner: True}):
        kept = np.asarray(at_a_stop(o, side)["joint_velocities"])
    with switch.forcing({owner: False}):
        held = np.asarray(at_a_stop(o, side)["joint_velocities"])
    q = jnp.asarray((0.3, side * (STOP_LIMIT - o) - DT * STOP_VELOCITY[1] * side))
    inverse = np.linalg.inv(np.asarray(mass_matrix(COUPLED, q)))
    expected = inverse[:, 1] / inverse[1, 1] * (side * o / DT)
    # The change is a difference of positions over the step: its rounding is the positions' over the step.
    tolerance = FLOOR_ULPS * float(np.finfo(np.float64).eps) * STOP_LIMIT / DT
    print(f"\n{owner}: the forced branches' velocities differ by {held - kept}, the impulse's {expected}")
    assert np.max(np.abs((held - kept) - expected)) <= tolerance


# -- A0's controller, its step as it runs inside the compiled step -------------------------------------------------

@cache
def tracking_law(feedback: bool, smoothing: float = 1.0):
    """A0's controller's law, made from its rig's plan (``chain_cases.tracked_rpm``) with one inverse-kinematics step,
    so that each switch of the pose error is evaluated once, at the point placed; unfiltered unless ``smoothing`` is
    given, so that its target is where it reads the body. Its step as inside the compiled step, traced afresh at
    every call, its multibody tree and its base pose."""
    plans = tracked_rpm("harness_tracking").plan()
    assert plans.controller is not None
    entry = dataclasses.replace(cast(Any, plans.controller.entry), orientation_feedback=feedback, iterations=1,
                                smoothing=smoothing)
    law = entry.law(control_context(plans))
    base = jnp.asarray((*AR4_BASE, 1.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    return inlined(law.step), law, entry.read(), base


def tracking_at(feedback: bool, body, *, twist=None, held=None, smoothing: float = 1.0) -> dict:
    """The controller's state and writes after one tick, the body at the pose ``body``, the arm at its home, the
    target's orientation ``held`` (its start where not given)."""
    step, law, _, _ = tracking_law(feedback, smoothing)
    state = {"target_position": law.state["target_position"],
             "target_orientation": law.state["target_orientation"] if held is None else jnp.asarray(held, jnp.float32)}
    reads = {"body_pose": jnp.asarray(body), "body_twist": jnp.zeros(6) if twist is None else jnp.asarray(twist),
             "joint_positions": jnp.asarray(AR4_HOME)}
    after, writes = step(state, reads)
    return {**after, **writes}


def turned_target(turn) -> jax.Array:
    """The target orientation (a state, in the world) whose turn from the end effector at home, as the inverse
    kinematics computes it in the arm's base frame, is the unit quaternion ``turn``."""
    _, law, tree, base = tracking_law(False)
    reached = frame_pose(tree, jnp.asarray(AR4_HOME, jnp.float32), "link_6")[3:]
    in_base = quaternion.multiply(jnp.asarray(turn, jnp.float32), reached)
    return quaternion.multiply(base[3:], in_base)


def about(axis, cosine_half: float) -> np.ndarray:
    """The unit quaternion with ``w = cosine_half`` turning about the unit ``axis``, in float64."""
    held = np.asarray(axis, dtype=np.float64)
    return np.asarray([cosine_half, *(np.sqrt(1.0 - cosine_half ** 2) * held / np.linalg.norm(held))])


def z_along(z) -> np.ndarray:
    """A body orientation whose z axis is ``z``: the shortest turn from the world's z onto it."""
    z = np.asarray(z, dtype=np.float64) / np.linalg.norm(z)
    q = np.asarray([1.0 + z[2], -z[1], z[0], 0.0])
    return q / np.linalg.norm(q)


def home_axes() -> np.ndarray:
    """The end effector's axes at home, in the world, as the controller holds them, columns x, y, z."""
    _, law, _, _ = tracking_law(True)
    return np.asarray(quaternion.rotation_matrix(law.state["target_orientation"]), dtype=np.float64)


def tracking_rows(declared) -> list[SwitchRow]:
    low, high = -0.05, 0.05
    hx, hy = home_axes()[:, 0], home_axes()[:, 1]

    def body_along(x):
        return (x, 0.0, -1e-3, 1.0, 0.0, 0.0, 0.0)

    def parallel(o):
        """The body's z this far, as a sine, from the home pose's x, plus the threshold."""
        sine = 1e-6 + o
        return tracking_at(True, (*body_along(0.0)[:3], *z_along(np.sqrt(1.0 - sine ** 2) * hx + sine * hy)))

    def hemisphere(o):
        """The target's orientation held so that the turn's quaternion q has q . held = o, the filter halfway: at
        once (unfiltered) the target is the turn whichever way round, and the edge shows in nothing."""
        body = (*body_along(0.0)[:3], *about((1.0, 0.0, 0.0), np.cos(0.15)))
        q = turn_quaternion(body)
        other = np.cross(np.cross(q[1:], np.array([0.0, 0.0, 1.0])), q[1:])
        across = np.asarray([0.0, *other])
        across -= np.dot(across, q) * q
        across /= np.linalg.norm(across)
        return tracking_at(True, body, held=o * q + np.sqrt(1.0 - o * o) * across, smoothing=0.5)

    return [
        SwitchRow(declared["arm_tracking.reach.low"], lambda o: tracking_at(False, body_along(low + o)), 1e-3),
        SwitchRow(declared["arm_tracking.reach.high"], lambda o: tracking_at(False, body_along(high - o)), 1e-3),
        SwitchRow(declared["arm_tracking.half_turn"],
                  lambda o: tracking_at(False, body_along(0.0), held=turned_target(about((0.3, 0.5, 0.8), -o))), 4e-3),
        SwitchRow(declared["arm_tracking.small_turn"],
                  lambda o: tracking_at(False, body_along(0.0), held=turned_target(
                      about((0.3, 0.5, 0.8), np.sqrt(1.0 - (1e-8 - o))))), 4e-6, side=-1),
        SwitchRow(declared["arm_tracking.parallel"], parallel, 4e-3),
        SwitchRow(declared["arm_tracking.hemisphere"], hemisphere, 4e-3),
    ]


def turn_quaternion(body) -> np.ndarray:
    """The unit quaternion of the turn the controller aims its end effector at for a body at ``body`` (pose, no
    twist): z the body's, x the home pose's x made perpendicular to it, in float64."""
    hx = home_axes()[:, 0]
    z = np.asarray(quaternion.rotation_matrix(jnp.asarray(body[3:])), dtype=np.float64)[:, 2]
    x = hx - np.dot(hx, z) * z
    x /= np.linalg.norm(x)
    turn = np.stack([x, np.cross(z, x), z], axis=1)
    outer = np.asarray(_outer(jnp.asarray(turn)), dtype=np.float64)
    column = outer[:, np.argmax(np.diag(outer))]
    return column / np.linalg.norm(column)


@cache
def unit_tube_arrays():
    """The shipped table of the unit tube (the asset, which the run has fetched), compiled in float64."""
    return tube_table_from(str(tube_asset.path())).arrays(jnp.float64)


def tube_rows(declared) -> list[SwitchRow]:
    """The tube wall kind's four switches, each placed on the tube's read of its table (``pair_block`` for the table's
    locating and the fold, ``tube_reads`` for the axis), where a single pair or sample can be placed: a read of two
    samples or more has self-pairs, whose angle and axial offset are exactly a knot, and the harness takes the smallest
    offset of a call. The table's row sits an offset past an interior knot on each of its four axes; the fold's rows
    at a pair a sine, or an axial offset, from the symmetric ends (``tube.angle`` and ``tube.axial`` are unmeasured: the
    table is mirrored there, so the two branches agree to rounding); the axis's a sample an offset from it, its
    diagonal block (the entries that are zero by symmetry are rounding, which the harness would read as a jump)."""
    table = unit_tube_arrays()
    knots = [np.asarray(axis.breaks) for axis in table.axes]
    radius_t, radius_s, angle, along = float(knots[0][10]), float(knots[1][12]), float(knots[2][10]), float(knots[3][20])

    def read(rho_t, rho_s, sine, offset):
        """The block of one pair, the cosine taken as the sine's positive root."""
        sine = jnp.asarray(sine)
        return pair_block(table, jnp.asarray(rho_t), jnp.asarray(rho_s), sine, jnp.sqrt(1.0 - sine ** 2),
                          jnp.asarray(offset))[0]

    def at_knots(o):
        return {"resistance": read(radius_t + o, radius_s + o, math.sin(angle + o), along + o)}

    def axis(o):
        found = tube_reads(jnp.asarray([[o, 0.0, 0.0]]), {"table": table, "frame": frame_across((0.0, 0.0, 1.0)),
                                                          "radius": 1.0, "centre": (0.0, 0.0, 0.0)})
        return {"resistance": jnp.diagonal(found.world[0, 0])}

    return [
        SwitchRow(declared["tube.table"], at_knots, 1e-4),
        SwitchRow(declared["tube.angle"], lambda o: {"resistance": read(0.4, 0.5, o, 0.31)}, 1e-3),
        SwitchRow(declared["tube.axial"], lambda o: {"resistance": read(0.4, 0.5, 0.37, o)}, 1e-3),
        SwitchRow(declared["tube.axis"], axis, 1e-3),
    ]


# -- the body held by its hull's points ----------------------------------------------------------------------------

#: The ball whose hull's points the hull contact's rows hold: a millimetre sphere, its quadrature at 0.4 mm (the
#: lowest of its points the ``BOTTOM``; ``NEXT`` the next lowest), and a cylinder along the tube's axis, whose caps'
#: centres lie on it.
HELD_BALL, BALL_SETTINGS, BALL_SPACING = Sphere(radius=mm(1.0)), {"radius": 1e-3}, 4e-4
ROD, ROD_SETTINGS = Cylinder(radius=mm(0.5), length=mm(2.0), axis=(1.0, 0.0, 0.0)), {
    "radius": 5e-4, "length": 2e-3, "axis": (1.0, 0.0, 0.0)}
HULL_MASS = 1e-5
#: FL-9's screw, and an ellipsoid shell, held at the wall for the guard points.
HELD_SCREW = (Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
                    starts=2), {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33, "starts": 2})
HELD_SHELL = (Ellipsoid(semi_axes=mm((2.0, 1.5, 1.2))), {"semi_axes": (2e-3, 1.5e-3, 1.2e-3)})


@cache
def ball_points() -> np.ndarray:
    return kind_points(HELD_BALL, BALL_SETTINGS, BALL_SPACING)


def lowest_two() -> tuple[int, int]:
    order = np.argsort(ball_points()[:, 2], kind="stable")
    return int(order[0]), int(order[1])


def held_ball(groups, pose):
    return hull_node(HELD_BALL, BALL_SETTINGS, BALL_SPACING, groups=groups, mass=HULL_MASS, pose=tuple(pose))


def ball_gap(index: int, pose) -> float:
    """The ball's point ``index``'s gap to the shrunk wall at ``pose``."""
    placed = np.asarray(pose[:3]) + ball_points()[index] @ rotation(pose[3:]).T
    return float(TUBE_RADIUS - HULL_MARGIN - np.hypot(placed[1], placed[2]))


def ball_below(index: int, gap: float, turn=(1.0, 0.0, 0.0, 0.0), *, others=()) -> np.ndarray:
    """The ball's pose at the orientation ``turn``, its centre below the tube's axis at the depth (found by bisection)
    at which the nearest of its point ``index`` and ``others`` is ``gap`` from the shrunk wall."""
    low, high = 0.0, TUBE_RADIUS
    for _ in range(200):
        middle = 0.5 * (low + high)
        nearest = min(ball_gap(i, (ALONG, 0.0, -middle, *turn)) for i in (index, *others))
        low, high = (middle, high) if nearest > gap else (low, middle)
    return np.asarray((ALONG, 0.0, -low, *turn))


def upright(index: int) -> tuple[float, float, float, float]:
    """The turn that takes the ball's point ``index`` straight below its centre, where the shrunk wall's normal
    through it passes through the centre."""
    point = ball_points()[index] / np.linalg.norm(ball_points()[index])
    axis = np.cross(point, (0.0, 0.0, -1.0))
    if np.linalg.norm(axis) < 1e-15:
        return (1.0, 0.0, 0.0, 0.0)
    angle = math.atan2(float(np.linalg.norm(axis)), float(point @ (0.0, 0.0, -1.0)))
    axis = axis / np.linalg.norm(axis) * math.sin(angle / 2)
    return (math.cos(angle / 2), float(axis[0]), float(axis[1]), float(axis[2]))


def ball_step(groups, pose, twist=(0.0,) * 6, held_passes: int | None = None):
    """The held ball's pose and twist after one step from ``pose`` and ``twist``, with no load (``held_passes``: the
    projection's passes, where a row measures one)."""
    import mime.nodes.body._rigid_inertial as rigid

    node = held_ball(groups, pose)
    state = {**node.initial_state(), "pose": jnp.asarray(pose), "twist": jnp.asarray(twist)}
    passes = rigid.HULL_PASSES
    rigid.HULL_PASSES = passes if held_passes is None else held_passes
    try:
        out = node.fresh_update(state, {"applied_wrench": jnp.zeros(6), "resistance": jnp.zeros((6, 6)),
                                        "background_load": jnp.zeros(6)}, DT)
    finally:
        rigid.HULL_PASSES = passes
    return {"pose": out["pose"], "twist": out["twist"]}


def tied(offset: float) -> np.ndarray:
    """The ball's pose with its two lowest points, ``first`` and ``second``, the nearest the shrunk wall and on it,
    ``second`` ``offset`` further from it: turned about the line across both and the vertical, by the angle (found by
    bisection, in the bracket of a coarse scan that holds it) that sets ``second``'s gap."""
    first, second = lowest_two()
    points = ball_points()
    across = np.cross(points[second] - points[first], (0.0, 0.0, 1.0))
    across = across / np.linalg.norm(across)

    def pose_at(angle: float) -> np.ndarray:
        """The ball turned by ``angle``, its own lowest point (the sphere's support straight down) on the shrunk wall,
        so that the step holds it where it is and every evaluation of the choice reads the row's offset."""
        turn = (math.cos(angle / 2), *(float(c) for c in math.sin(angle / 2) * across))
        return np.asarray((ALONG, 0.0, -(TUBE_RADIUS - HULL_MARGIN - BALL_SETTINGS["radius"]), *turn))

    def excess(angle: float) -> float:
        held = pose_at(angle)
        return ball_gap(second, held) - ball_gap(first, held)

    angles = np.linspace(-0.6, 0.6, 25)
    values = [excess(float(angle)) for angle in angles]
    k = next(k for k in range(len(angles) - 1) if (values[k] - offset) * (values[k + 1] - offset) < 0)
    low, high = (float(angles[k]), float(angles[k + 1])) if values[k] < offset else (float(angles[k + 1]),
                                                                                       float(angles[k]))
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if excess(middle) < offset else (low, middle)
    return pose_at(high)


def hull_rows(declared) -> list[SwitchRow]:
    """The hull contact's four switches, each placed on a held ball whose groups hold only the points the row needs, so
    that every evaluation of its owner over the sweeps is at the row's offset: ``contact.point`` its two lowest points
    in one group, ``offset`` apart in their gaps, the lowest on the shrunk wall and the ball moving into it at
    0.1 um/s (so that the choice reaches the twist, and the ball moves too little in the step for the projection's
    choice to read another offset); ``contact.gap`` the lowest point straight below the centre, ``offset`` from the
    shrunk wall, the ball moving into it at 1 mm/s; ``contact.impulse`` that point on the wall, the ball moving into it
    at the speed whose stopping takes a force of ``offset``; ``contact.push`` that point past the wall by the depth
    whose projection's multiplier is ``offset``, the ball at rest, its projection one pass (the second's multiplier is
    what the first leaves, which no row can place at the same offset)."""
    first, second = lowest_two()
    down = upright(first)
    alone = [[first]]

    def pressed(speed: float):
        return (0.0, 0.0, -speed, 0.0, 0.0, 0.0)

    return [
        SwitchRow(declared["contact.point"], lambda o: ball_step([[first, second]], tied(o), pressed(1e-7)), 1e-6),
        SwitchRow(declared["contact.gap"], lambda o: ball_step(alone, ball_below(first, o, down), pressed(1e-3)),
                  1e-8),
        SwitchRow(declared["contact.impulse"],
                  lambda o: ball_step(alone, ball_below(first, 0.0, down), pressed(o * DT / HULL_MASS)), 1e-6),
        SwitchRow(declared["contact.push"],
                  lambda o: ball_step(alone, ball_below(first, -o / HULL_MASS, down), held_passes=1), 1e-12),
    ]


def all_rows(graph) -> dict[str, SwitchRow]:
    declared = declarations()
    rows = [*body_rows(graph, declared), *cylinder_rows(declared), *solid_rows(declared), *drag_rows(graph, declared),
            *swim_rows(graph, declared), *chain_rows(declared), *tracking_rows(declared), *tube_rows(declared),
            *hull_rows(declared)]
    return {row.declaration.owner: row for row in rows}


OWNERS = sorted(["contact.band", "contact.outward", "contact.hold", "contact.closing", "contact.scale",
                 "contact.project", "contact.rule", "contact.rule_outward", "contact.point", "contact.gap",
                 "contact.impulse", "contact.push", "cylinder.axis", "cylinder.axis_norm", "cylinder.support",
                 "drag.axis", "drag.across", "drag.end", "drag.room", "drag.clamp", "drag.clamp_divisor",
                 "drag.feature_gap_floor", "drag.gap_floor", "drag.log_floor", "swim.window", "swim.mean",
                 "sphere.centre", "capsule.segment", "capsule.end.low", "capsule.end.high", "slab.mid",
                 "screw.seam", "screw.focus", "arm.limit.low", "arm.limit.high", "arm.stop.low", "arm.stop.high",
                 "arm.friction.low", "arm.friction.high", "arm.stick",
                 "arm_tracking.reach.low", "arm_tracking.reach.high", "arm_tracking.half_turn",
                 "arm_tracking.small_turn", "arm_tracking.parallel", "arm_tracking.hemisphere",
                 "tube.table", "tube.angle", "tube.axial", "tube.axis"])


def test_every_owner_mime_declares_has_a_row():
    assert sorted(declarations()) == OWNERS


@pytest.mark.parametrize("owner", OWNERS)
def test_each_switch_measures_within_its_declared_order(graph, owner):
    found = classify(all_rows(graph)[owner], KINK_LIMIT)
    print(f"\n{found.line()}; indicators {found.indicators}")
    assert found.problems == [], "\n".join(found.problems)
    declared = all_rows(graph)[owner].declaration
    jumps = {group for group, order in declared.orders.items() if order < 0 and found.measures[group].continuous}
    if jumps:
        print(f"{owner}: declared a jump in {sorted(jumps)}, measured continuous at this point")


# -- the kink limit ------------------------------------------------------------------------------------------------

def test_the_kink_limit_lies_between_drag_clamps_kink_and_drag_ends_jump(graph):
    rows = all_rows(graph)
    kink = classify(rows["drag.clamp"], KINK_LIMIT).measures["resistance"]
    jump = classify(rows["drag.end"], KINK_LIMIT).measures["resistance"]
    fitted = math.sqrt(kink.ratio * jump.ratio)
    print(f"\ndrag.clamp's ratio {kink.ratio:.4g}, drag.end's {jump.ratio:.4g}: their geometric mean {fitted:.4g}; "
          f"KINK_LIMIT {KINK_LIMIT}; rounding floor {kink.floor:.3g} and {jump.floor:.3g} (64 units in the last place "
          f"of the resistance's largest entry)")
    assert 2 * kink.ratio < KINK_LIMIT < jump.ratio / 2
    assert 0.5 < fitted / KINK_LIMIT < 2


PLANTED = switch.Declaration(owner="planted.switch", indicator="a planted offset", orders={"value": 0},
                             route="none: planted")


def analytic(value):
    return lambda o: {"value": value(jnp.asarray(1.0 + o), jnp.asarray(1.0))}


def mixed(share: float):
    """A branch difference of a jump beside a kink: ``share`` times the kink's growth over a quarter step (``1e-3 /
    4``), with the kink's slope 1."""
    return lambda x, y: switch.where(x > y, x - y + share * 1e-3 / 4, 0.0 * x, owner="planted.switch",
                                     resolution=0.0, indicator=x - y)


@pytest.mark.parametrize(("name", "value", "read"), [
    ("an analytic kink", lambda x, y: 3.0 * switch.maximum(x, y, owner="planted.switch", resolution=0.0,
                                                           indicator=x - y) + 2.0 * x, "continuous"),
    ("an analytic jump", lambda x, y: switch.where(x > y, x + 0.5, x, owner="planted.switch", resolution=0.0,
                                                   indicator=x - y), "jump"),
    ("branches that coincide away from the switch, which measure nothing",
     lambda x, y: switch.where(x > y, 2.0 * x, 2.0 * x, owner="planted.switch", resolution=0.0, indicator=x - y),
     "unmeasured"),
    ("a jump of 0.4 of the kink's growth, which reads as a jump", mixed(0.4), "jump"),
    ("a jump of 0.04 of the kink's growth, which reads continuous: the limit's resolution", mixed(0.04), "continuous"),
])
def test_control_the_kink_limit_classifies_independent_planted_rows(name, value, read):
    """Each row is declared a kink (order 0): a jump fails it, and so does a row that measures nothing."""
    found = classify(SwitchRow(PLANTED, analytic(value), 1e-3), KINK_LIMIT)
    assert (found.measures["value"].continuous, found.unmeasured) == {
        "continuous": (True, False), "jump": (False, False), "unmeasured": (True, True)}[read], found.line()
    expected = {"continuous": [], "jump": ["it is a jump"], "unmeasured": ["so its row measures nothing"]}[read]
    assert len(found.problems) == len(expected) and all(
        phrase in problem for phrase, problem in zip(expected, found.problems, strict=True)), found.problems


def test_control_a_jump_in_a_small_output_beside_a_large_one_is_measured():
    """Each output's floor is its own: a jump of 1e-15 in an output of about 1e-12 is far above that output's floor
    (about 1e-26), though far below a floor taken from its neighbour, 1."""
    def beside(o):
        x = jnp.asarray(o)
        small = switch.where(x > 0, 1e-12 + 1e-15, 1e-12, owner="planted.switch", resolution=0.0, indicator=x)
        return {"value": jnp.stack([jnp.asarray(1.0), small])}

    found = classify(SwitchRow(PLANTED, beside, 1e-3), KINK_LIMIT)
    assert found.measures["value"].continuous is False and not found.unmeasured, found.line()


def test_control_drag_end_declared_a_kink_fails_and_drag_clamps_continuity_passes(graph):
    rows = all_rows(graph)
    as_kink = switch.Declaration(owner="drag.end", indicator="planted", orders={"resistance": 0}, route="none: planted")
    found = classify(SwitchRow(as_kink, rows["drag.end"].evaluate, rows["drag.end"].delta), KINK_LIMIT)
    assert any("declared order 0 in resistance" in problem and "it is a jump" in problem for problem in found.problems)
    assert classify(rows["drag.clamp"], KINK_LIMIT).problems == []


class _ClampDivisorAsAJump:
    """``mime.core.switch`` as the drag's module sees it, with ``drag.clamp_divisor`` written as a jump (its divisor
    7 above its floor on the floor's side): the review's plant."""

    def __getattr__(self, name):
        return getattr(switch, name)

    @staticmethod
    def maximum(x, y, *, owner, resolution, indicator):
        if owner == "drag.clamp_divisor":
            return switch.where(x > y, x, 7.0 + y, owner=owner, resolution=resolution, indicator=indicator)
        return switch.maximum(x, y, owner=owner, resolution=resolution, indicator=indicator)


def test_control_the_clamps_divisor_written_as_a_jump_fails_its_declared_kink(graph, monkeypatch):
    row = all_rows(graph)["drag.clamp_divisor"]
    assert classify(row, KINK_LIMIT).problems == []
    monkeypatch.setattr(surrogate_drag, "switch", _ClampDivisorAsAJump())
    found = classify(row, KINK_LIMIT)
    assert any("declared order 0 in resistance" in problem and "it is a jump" in problem
               for problem in found.problems), found.line()


def test_control_a_row_that_measures_nothing_or_something_fails_where_its_declaration_says_otherwise(graph):
    rows = all_rows(graph)
    scale = rows["contact.scale"]
    assert classify(scale, KINK_LIMIT).unmeasured
    silent = SwitchRow(dataclasses.replace(scale.declaration, unmeasured=None), scale.evaluate, scale.delta)
    (problem,) = classify(silent, KINK_LIMIT).problems
    assert "so its row measures nothing" in problem
    clamp = rows["drag.clamp"]
    claimed = SwitchRow(dataclasses.replace(clamp.declaration, unmeasured="planted"), clamp.evaluate, clamp.delta)
    (problem,) = classify(claimed, KINK_LIMIT).problems
    assert "declares that no output shows its branches (planted), and its row measures a change in resistance" \
        in problem


def test_control_a_misplaced_point_or_a_flipped_indicator_fails(graph):
    rows = all_rows(graph)
    flipped = SwitchRow(rows["contact.project"].declaration, lambda o: rows["contact.project"].evaluate(-o), 1e-6)
    displaced = SwitchRow(rows["drag.clamp"].declaration, lambda o: rows["drag.clamp"].evaluate(1e3 * o), 1e-3)
    for row in (flipped, displaced):
        assert any("a sign or units error" in problem for problem in classify(row, KINK_LIMIT).problems)


def test_control_a_step_traced_before_the_setting_opened_fails(graph):
    node = graph.get_node("drag:umr")
    node.update(node_state(graph, "drag:umr"), {"pose": jnp.asarray((ALONG, 0.0, -5e-4, 1.0, 0.0, 0.0, 0.0))}, DT)

    def cached(o):
        pose = jnp.asarray((ALONG, 0.0, -5e-4, *tilted(math.acos(o - 1e-3))))
        return node.update(node_state(graph, "drag:umr"), {"pose": pose}, DT)

    found = classify(SwitchRow(declarations()["drag.end"], cached, 1e-3), KINK_LIMIT)
    assert any("the forced branch never reached it" in problem for problem in found.problems)


# -- the domain guards ---------------------------------------------------------------------------------------------

def guard_points(graph):
    """Points where each domain guard of a node's step has its condition false: the body at the world's origin, on
    the tube's axis and free (the norm's, the cylinder's two and the hold's); a body held by its hull's points, one of
    them on the tube's axis (the sweeps'); the drag on the axis; the swim speed
    before it has observed; the point dipole read away from its own centre, where its ``nonzero_norm`` is reached; and
    the drag's MRS samples on FL-9's screw and on an ellipsoid, where each kind's surface element (each sample's
    ``nonzero_norm``) is."""

    def body(state):
        unloaded = {"applied_wrench": jnp.zeros(6), "resistance": jnp.zeros((6, 6)), "background_load": jnp.zeros(6)}
        return graph.get_node("umr").fresh_update(state, unloaded, DT, params=leaves(graph, "umr"))

    body_state = {"pose": jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)), "twist": jnp.asarray((1e-4,) * 6)}
    rod_points = kind_points(ROD, ROD_SETTINGS, BALL_SPACING)
    on_the_axis = int(np.argmin(np.hypot(rod_points[:, 1], rod_points[:, 2])))
    rod = hull_node(ROD, ROD_SETTINGS, BALL_SPACING, groups=[[on_the_axis]], mass=HULL_MASS)

    def held_rod(state):
        """The rod held by one point of its hull, a cap's centre, on the tube's axis, where the wall has no normal
        and the point's row no length (the sweeps' guard)."""
        unloaded = {"applied_wrench": jnp.zeros(6), "resistance": jnp.zeros((6, 6)), "background_load": jnp.zeros(6)}
        out = rod.fresh_update({**rod.initial_state(), **state}, unloaded, DT)
        return {"pose": out["pose"], "twist": out["twist"]}

    screw_node = hull_node(*HELD_SCREW, 4e-4, mass=HULL_MASS)
    shell_node = hull_node(*HELD_SHELL, 4e-4, mass=HULL_MASS)
    at_the_wall = {"pose": jnp.asarray((ALONG, 0.0, -(TUBE_RADIUS - HULL_MARGIN - 2.0748e-3), 1.0, 0.0, 0.0, 0.0)),
                   "twist": jnp.asarray((0.0, 0.0, -1e-3, 3.0, 0.0, 0.0))}

    def held_at_the_wall(state):
        """FL-9 (the screw's support refined on its surface) and an ellipsoid shell (its closed form), each held by
        its hull at the wall, moving into it: every guard and nonzero norm of the supports and the solve's scaling."""
        unloaded = {"applied_wrench": jnp.zeros(6), "resistance": jnp.zeros((6, 6)), "background_load": jnp.zeros(6)}
        screw = screw_node.fresh_update({**screw_node.initial_state(), **state}, unloaded, DT)
        shell = shell_node.fresh_update({**shell_node.initial_state(), **state}, unloaded, DT)
        return {"screw": screw["twist"], "shell": shell["twist"]}

    def drag(pose):
        return graph.get_node("drag:umr").fresh_update(node_state(graph, "drag:umr"), {"pose": pose}, DT,
                                                       params=leaves(graph, "drag:umr"))

    drag_pose = jnp.asarray((ALONG, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

    def swim(twist):
        return swim_at_twist(graph, twist)

    swim_twist = jnp.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0))

    def solids(point):
        """Each solid's shape on its medial set (the ball's centre, the capsule's segment, the slab's mid-plane),
        and the screw near its axis."""
        found = {"ball": shape_at(SPHERE, BALL, point), "pill": shape_at(CAPSULE, PILL, point),
                 "plates": shape_at(SLAB, PLATES, point), "screw": shape_at(SCREW, SCREW_PARAMETERS, point + 1e-4)}
        return {f"{solid}.{group}": value for solid, outputs in found.items() for group, value in outputs.items()}

    on_the_sets = jnp.zeros(3)

    def focus(at):
        """The screw on its focal set, where its Newton step is not taken."""
        return shape_at(SCREW, SCREW_PARAMETERS, at)

    on_the_focus = jnp.asarray(_screw_focus_place(SCREW_PARAMETERS, 0.0, FOCUS_DRAW))

    def dipole(point):
        return dipole_at(chain_graph(), point)

    away = jnp.asarray((0.05, 0.0, 0.0))

    def elements(pose):
        """The drag's MRS rung on FL-9's screw, which samples the helix kind in its step: its outputs
        that have a derivative (``wall_accepted``, a flag, has none)."""
        built = mrs_graph()
        node = built.get_node("drag:screw")
        out = node.fresh_update(node_state(built, "drag:screw"), {"pose": pose}, DT,
                                params=parameter_leaves(built)["drag:screw"])
        return {key: value for key, value in out.items() if key != "wall_accepted"}

    screw_pose = jnp.asarray((0.0, 1e-2, 0.0, 1.0, 0.0, 0.0, 0.0))

    def egg_elements(pose):
        """The drag's MRS rung on an ellipsoid, which samples its quadrature in its step: its surface element (each
        sample's ``nonzero_norm``), away from zero at every Gauss-Legendre node."""
        built = mrs_solids_graph()
        node = built.get_node("drag:egg")
        out = node.fresh_update(node_state(built, "drag:egg"), {"pose": pose}, DT,
                                params=parameter_leaves(built)["drag:egg"])
        return {key: value for key, value in out.items() if key != "wall_accepted"}

    egg_pose = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

    def tube_read(points):
        """The tube wall kind's correction at samples on the tube's axis, where its rule on the axis is."""
        found = tube_correction(points, {"table": unit_tube_arrays(), "frame": frame_across((0.0, 0.0, 1.0)),
                                         "radius": 1.0, "centre": (0.0, 0.0, 0.0)}, 1.0)
        return {"matrix": found.matrix}

    on_the_tube_axis = jnp.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, -0.2]])
    log_table = Table((Axis("x", "1", (1.0, 2.0, 4.0)),), np.asarray([1.0, 2.0, 3.0]), scheme="loglog")

    def logarithmic(query):
        """A log-log table read at a query that is not positive, outside its readable range: the logarithm's guard."""
        return {"value": evaluate(log_table, query, owner="planted.table").value}

    not_positive = jnp.asarray(-1.0)
    return {
        "the body at the origin, on the axis, free": (lambda: body(body_state), lambda: jax.jacfwd(body)(body_state)),
        "the hull contact, its point on the tube's axis": (lambda: held_rod(body_state),
                                                           lambda: jax.jacfwd(held_rod)(body_state)),
        "the hull contact on FL-9 and an ellipsoid at the wall": (lambda: held_at_the_wall(at_the_wall),
                                                                  lambda: jax.jacfwd(held_at_the_wall)(at_the_wall)),
        "the drag on the axis": (lambda: drag(drag_pose), lambda: jax.jacfwd(drag)(drag_pose)),
        "the swim speed before observing": (lambda: swim(swim_twist), lambda: jax.jacfwd(swim)(swim_twist)),
        "the solids' shapes on their sets": (lambda: solids(on_the_sets), lambda: jax.jacfwd(solids)(on_the_sets)),
        "the screw on its focal set": (lambda: focus(on_the_focus), lambda: jax.jacfwd(focus)(on_the_focus)),
        "the dipole away from its centre": (lambda: dipole(away), lambda: jax.jacfwd(dipole)(away)),
        "the MRS samples on the screw": (lambda: elements(screw_pose),
                                               lambda: jax.jacfwd(elements)(screw_pose)),
        "the MRS samples on the ellipsoid": (lambda: egg_elements(egg_pose),
                                             lambda: jax.jacfwd(egg_elements)(egg_pose)),
        "the tube's read on the axis": (lambda: tube_read(on_the_tube_axis),
                                        lambda: jax.jacfwd(tube_read)(on_the_tube_axis)),
        "a log-log table read past its range": (lambda: logarithmic(not_positive),
                                                lambda: jax.jacfwd(logarithmic)(not_positive)),
        **tracking_points(),
    }


@cache
def mrs_graph():
    return mrs_free_space().build().graph


@cache
def mrs_solids_graph():
    return mrs_solids().build().graph


def tracking_points():
    """A0's controller with its end effector on its target's orientation (the rotation vector's series), turning
    it (its closed form), and with the body's z along the home pose's x (the target's x from the home pose's y, the
    feedback's norms away from zero); each derivative with respect to the body's pose."""
    hx = home_axes()[:, 0]
    still = jnp.asarray((0.0, 0.0, -1e-3, 1.0, 0.0, 0.0, 0.0))
    along_x = jnp.asarray((0.0, 0.0, -1e-3, *z_along(hx)))
    turning = turned_target(about((0.3, 0.5, 0.8), np.cos(0.2)))

    def point(feedback, body, held=None):
        def outputs(at):
            return tracking_at(feedback, at, held=held)

        return (lambda: outputs(body), lambda: jax.jacfwd(outputs)(body))

    return {"A0's controller on its target's orientation": point(False, still),
            "A0's controller turning its end effector": point(False, still, turning),
            "A0's controller, the body's z along the home pose's x": point(True, along_x)}


def swim_at_twist(graph, twist):
    node = graph.get_node("swim_speed:umr")
    state = {"speed": jnp.asarray(0.0), "mean_speed": jnp.asarray(0.0), "distance": jnp.asarray(0.0),
             "observed": jnp.asarray(0.0), "steps": jnp.asarray(0, dtype=jnp.int32)}
    return node.fresh_update(state, {"twist": twist}, DT, params=leaves(graph, "swim_speed:umr", after=1.0))


@pytest.mark.parametrize("where", ["the body at the origin, on the axis, free",
                                   "the hull contact, its point on the tube's axis",
                                   "the hull contact on FL-9 and an ellipsoid at the wall", "the drag on the axis",
                                   "the swim speed before observing", "the solids' shapes on their sets",
                                   "the screw on its focal set", "the dipole away from its centre",
                                   "the MRS samples on the screw",
                                   "A0's controller on its target's orientation",
                                   "A0's controller turning its end effector",
                                   "A0's controller, the body's z along the home pose's x"])
def test_replacing_every_fallback_changes_nothing_where_its_condition_is_false(graph, where):
    evaluate, derivative = guard_points(graph)[where]
    assert fallback_problems(evaluate, derivative, where).problems == []


def reached_at(evaluate) -> switch.Reached:
    """The domain guards and nonzero norms a point's step reaches, by call site."""
    with switch.replacing_fallbacks(1.0) as reached:
        evaluate()
    return reached


def coverage_problems(listed, reached: list[switch.Reached]) -> list[str]:
    """Each domain guard ``listed`` (the switch scan's ``guards``) that no point reaches with its condition false;
    each nonzero norm listed that no point reaches, or one reaches at zero; and each one reached that the scan does
    not list."""
    sites: dict[tuple[str, int, str], bool] = {}
    for held in reached:
        for kind, found in (("domain_guard", held.guard_sites), ("nonzero_norm", held.nonzero_norms)):
            for site, flag in found.items():
                file, _, line = site.rpartition(":")
                if Path(file).resolve().is_relative_to(PACKAGE.resolve()):
                    key = (Path(file).resolve().relative_to(PACKAGE.resolve()).as_posix(), int(line), kind)
                    sites[key] = sites.get(key, False) or flag
    problems = []
    matched = set()
    for guard in listed:
        hits = {key: flag for key, flag in sites.items()
                if (key[0], key[2]) == (guard.path, guard.kind) and guard.line <= key[1] <= guard.end_line}
        matched |= set(hits)
        flagged = any(hits.values())
        if guard.kind == "domain_guard" and not flagged:
            problems.append(f"{guard.where}: a domain guard no point reaches with its condition false, so nothing "
                            f"shows its fallback is discarded; add a point that reaches it")
        if guard.kind == "nonzero_norm" and not hits:
            problems.append(f"{guard.where}: a nonzero_norm no point reaches")
        if guard.kind == "nonzero_norm" and flagged:
            problems.append(f"{guard.where}: a nonzero_norm a point reaches at zero, which it says no valid state "
                            f"does: write it as switch.norm, with its owner, or refuse that state")
    problems += [f"{path}:{line}: a {kind} a point reaches that the switch scan does not list"
                 for path, line, kind in sorted(set(sites) - matched)]
    return problems


def test_every_guard_the_scan_lists_is_reached_false_and_every_nonzero_norm_away_from_zero(graph):
    """The scan lists every ``domain_guard`` and ``nonzero_norm`` call in scope; the guard points reach each guard with
    its condition false (so the test above judges it) and each nonzero norm, never at zero."""
    listed = tree_scan().guards
    reached = [reached_at(evaluate) for evaluate, _ in guard_points(graph).values()]
    print("\n" + "\n".join(f"{guard.where} {guard.kind}" for guard in listed))
    assert {guard.kind for guard in listed} == {"domain_guard", "nonzero_norm"}
    assert coverage_problems(listed, reached) == []


def test_control_a_guard_no_point_reaches_false_or_a_nonzero_norm_at_zero_fails(graph):
    listed = tree_scan().guards
    reached = [reached_at(evaluate) for evaluate, _ in guard_points(graph).values()]
    unreached = dataclasses.replace(next(guard for guard in listed if guard.kind == "domain_guard"), line=1,
                                    end_line=1, where="planted.py:1")
    (problem,) = coverage_problems([*listed, unreached], reached)
    assert problem.startswith("planted.py:1: a domain guard no point reaches with its condition false")
    unfit = reached_at(lambda: drag_at(graph, (ALONG, 0.0, -5e-4), vessel_axis=(0.0, 0.0, 0.0)))
    (problem,) = coverage_problems(listed, [*reached, unfit])
    assert "a nonzero_norm a point reaches at zero" in problem and "_surrogate_drag.py" in problem


@pytest.mark.parametrize(("name", "value"), [
    ("a fallback reaching the value", lambda x: 3.0 * switch.domain_guard(x > 0, x, 1.0)),
    ("a fallback reaching the derivative (_unit's old selection)",
     lambda x: x / switch.domain_guard(jnp.abs(x) > 0, jnp.abs(x), 1.0)),
    ("a zero fallback reaching the value, which scaling alone would leave unchanged",
     lambda x: 1.0 + switch.domain_guard(x > 0, x, 0.0)),
    ("no guard reached, which tests nothing", lambda x: 2.0 * x),
])
def test_control_a_planted_guard_whose_fallback_reaches_an_output_fails(name, value):
    point = jnp.asarray(0.0)
    found = fallback_problems(lambda: {"value": value(point)}, lambda: {"value": jax.grad(value)(point)}, name)
    assert found.problems, name


def test_a_zero_fallback_that_reaches_no_output_passes():
    """``nonzero_norm``'s zero branch, a domain guard with a zero fallback, at a vector that is not zero."""
    point = jnp.asarray([3.0, 4.0])
    value = lambda v: v / switch.nonzero_norm(v)  # noqa: E731 - one expression
    found = fallback_problems(lambda: {"value": value(point)}, lambda: {"value": jax.jacfwd(value)(point)}, "a norm")
    assert found.problems == [] and list(found.nonzero_norms.values()) == [False]


def test_the_harness_runs_in_float64():
    assert jnp.zeros(()).dtype == np.float64
