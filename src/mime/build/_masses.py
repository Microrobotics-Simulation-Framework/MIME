"""Mounted mass: where each mass on an apparatus's chain lands, and what carries it.

A mass is a part with a mass (its density's, from its material or declared, or its mass declared), a spinner's rotor
(a dynamic motor's), or a segment whose own description gives mass its joints move (an arm's links). Each is placed by
walking from the frame it is mounted on towards the world:

- on a frame of a segment whose dynamics carry mass (a dynamic arm), it is **carried**: a rigid body welded on that
  frame at build;
- on a spinner's rotor frame, it is part of the **rotor**;
- on a frame fixed to its segment's base (a motor's stator), it rides on the frame that segment mounts on, and the
  walk goes on from there, or it is **still**, fixed in the world;
- on any other frame, its segment's own joints move it: an **articulated** payload where a segment below carries mass,
  refused there, and otherwise **prescribed**, a kinematic host whose motion no mass changes, which takes every case.

A spinner on a carrying segment has its rotor checked: refused when its centre of mass lies off its axis, or its
inertia turns out of it, by more than a residue (below); its mass and mean inertia across the axis are welded on the
host's frame, its moment about the axis rides on its spin, and its half difference across the axis (a rotor unequal
across it) turns with the spin, which hands it to the host at each step. On every host the spin carries that half
difference's torque."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from mime.core.multibody import IDENTITY, Inertial, Payload, RigidTransform, RotorSplit, combined, split_rotor
from mime.core.multibody._tree import as_vector3
from mime.build._kicks import HostedSpinner
from mime.build._refusals import RefusalClass
from mime.core.multibody._urdf import ISSUES
from mime.registry import ChainSegment, Spinner

if TYPE_CHECKING:
    from mime.build._plan import EntryPlan

CARRIED, ROTOR, STILL, PRESCRIBED, ARTICULATED = "carried", "rotor", "still", "prescribed", "articulated"

#: The share of a rotor's weight (at standard gravity) up to which the rotating load a spinner does not carry, at the
#: rotor's top speed, is a residue and accepted on a dynamic segment: the force ``m r Omega^2`` of its centre of mass
#: ``r`` off the axis, and the torque ``|I e x e| Omega^2`` of its inertia turned out of the axis against its weight on
#: its radius of gyration about the axis, ``g sqrt(m J_parts)``. A labelled decision, not a measured bound: a CAD
#: placement's residue (1e-9 m off A0's axis, a microradian of turn) makes about 1e-6 of it at A0's 10 turns a second,
#: and a real offset (0.1 mm) 0.04, refused by name with its size. Neither load is carried: a rotor beyond it needs its
#: own joint in the host's tree (a grafted jointed sub-tree, logged for later), or the kinematic rung.
RESIDUE_SHARE = 1e-3
#: The weight's acceleration the residue is judged against, standard gravity (m/s^2), whatever the experiment's.
STANDARD_GRAVITY = 9.80665

type Refuse = Callable[[str, RefusalClass, str, str], None]


@dataclass(frozen=True)
class Placed:
    """Where a mass lands: how (one of the five above), the segment that carries, spins or moves it (its id), the
    frame of that segment, and its placement in that frame."""

    how: str
    segment: str
    frame: str
    placement: RigidTransform


def transform(pose: Sequence[float]) -> RigidTransform:
    return RigidTransform((pose[0], pose[1], pose[2]), (pose[3], pose[4], pose[5], pose[6]))


def on_dynamic(segments: Mapping[str, EntryPlan], plan: EntryPlan) -> str | None:
    """The segment below ``plan`` whose dynamics carry mass, if any: what its own motion would load."""
    mount = plan.mount
    while mount is not None:
        host = segments[mount.segment]
        if cast(ChainSegment, host.entry).carries_mass():
            return host.owner
        mount = host.mount
    return None


def placed(segments: Mapping[str, EntryPlan], owner: str, frame: str, placement: RigidTransform) -> Placed:
    """Where a mass at ``placement`` in the frame ``frame`` of the segment ``owner`` lands."""
    plan = segments[owner]
    entry = cast(ChainSegment, plan.entry)
    spinner = entry.spinner(plan.settings)
    if entry.carries_mass():
        return Placed(CARRIED, owner, frame, placement)
    if spinner is not None and frame == spinner.frame:
        return Placed(ROTOR, owner, frame, placement)
    if frame in entry.riding():
        if plan.mount is None:
            return Placed(STILL, owner, frame, placement)
        own = cast(tuple[tuple[float, ...], ...], plan.segment.initial if plan.segment is not None else ())
        frames = plan.segment.frames if plan.segment is not None else ()
        offset = transform(own[frames.index(frame)]) if own else IDENTITY
        base = transform(entry.placement(plan.settings))
        return placed(segments, plan.mount.segment, plan.mount.frame, base.then(offset).then(placement))
    return Placed(ARTICULATED if on_dynamic(segments, plan) else PRESCRIBED, owner, frame, placement)


def _articulated(refuse: Refuse, what: str, mover: str, host: str) -> None:
    refuse("chain.articulated-payload", RefusalClass.USER,
           f"{what} is moved by the joints of {mover!r}, a segment mounted on {host!r}, whose dynamics carry mass: an "
           f"articulated payload, whose joints {host!r}'s tree does not hold",
           f"use the kinematic rung of {host!r}, whose motion no mounted mass changes and which takes every mounted "
           f"mass (Arm.Kinematic(...) or Motor.Kinematic(...)); or mount the mass rigidly on one of {host!r}'s frames, "
           f"Part(..., mount=\"{host}.<frame>\"); jointed sub-trees grafted on a link are logged for later: ask at "
           f"{ISSUES}")


@dataclass
class MassPlan:
    """What the walk found: the rigid bodies to graft on each carrying segment, each spinner's moment about its axis,
    the build report's lines, and each spinner whose rotor rides on a carrying segment, whose kick on it the build
    judges (``mime.build._kicks``)."""

    mounted: dict[str, list[Payload]]
    spin_inertia: dict[str, float]
    lines: list[str]
    spin_asymmetry: dict[str, tuple[float, ...]]
    hosted: list[HostedSpinner]


def plan_masses(segments: Mapping[str, EntryPlan],
                parts: Sequence[tuple[str, str, tuple[float, float, float], Inertial]], refuse: Refuse) -> MassPlan:
    """``parts`` are each part with a mass that is mounted on a chain: its id, the frame it is mounted on
    (``<segment id>.<frame>``), its centroid there, and its inertial in its own frame (its centroid at its origin)."""
    found = MassPlan({}, {}, [], {}, [])
    rotors: dict[str, list[tuple[str, Inertial]]] = {}
    for name, mount, at, held in parts:
        owner, _, frame = mount.rpartition(".")
        where = placed(segments, owner, frame, RigidTransform(at))
        mass = held.mass
        if where.how == CARRIED:
            found.mounted.setdefault(where.segment, []).append(Payload(f"mounted:{name}", where.frame,
                                                                       where.placement, held))
            found.lines.append(f"{name}: {mass:.6g} kg on {mount}, welded on {where.segment}.{where.frame} at "
                               f"{_vec(where.placement.translation)} m")
        elif where.how == ROTOR:
            rotors.setdefault(where.segment, []).append((name, held.placed(where.placement)))
        elif where.how == ARTICULATED:
            _articulated(refuse, f"part {name!r} ({mass:.6g} kg, on {mount})", where.segment,
                         cast(str, on_dynamic(segments, segments[where.segment])))
        else:
            found.lines.append(f"{name}: {mass:.6g} kg on {mount}, " + (
                "fixed in the world" if where.how == STILL else
                f"moved by {where.segment!r}, a prescribed motion, which no mass changes: not carried"))
    for owner, plan in segments.items():
        entry = cast(ChainSegment, plan.entry)
        if plan.segment is not None and plan.segment.massive:
            host = on_dynamic(segments, plan)
            if host is not None:
                _articulated(refuse, f"the links of {owner!r}, which its description gives mass", owner, host)
        spinner = entry.spinner(plan.settings)
        if spinner is not None:
            _plan_spinner(segments, plan, spinner, rotors.get(owner, []), found, refuse)
    return found


def _vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.6g}" for c in values) + ")"


def _plan_spinner(segments: Mapping[str, EntryPlan], plan: EntryPlan, spinner: Spinner,
                  parts: Sequence[tuple[str, Inertial]], found: MassPlan, refuse: Refuse) -> None:
    owner = plan.owner
    entry = cast(ChainSegment, plan.entry)
    rotor = combined([inertial for _, inertial in parts]) if parts else Inertial(0.0)
    split = split_rotor(rotor, spinner.axis)
    axial = spinner.inertia + split.axial
    found.spin_inertia[owner] = axial
    named = ", ".join(name for name, _ in parts) or "no parts"
    found.lines.append(f"{owner}'s rotor: about its axis {axial:.6g} kg m^2 ({spinner.inertia:.6g} declared, "
                       f"{split.axial:.6g} from {named}), carried by its absolute spin")
    if parts and not split.symmetric:
        found.spin_asymmetry[owner] = tuple(float(c) for row in split.deviator for c in row)
        found.lines.append(f"{owner}'s rotor across its axis: moments {split.moments[0]:.6g} and "
                           f"{split.moments[1]:.6g} kg m^2, their mean carried as a body and their half difference "
                           f"{split.asymmetry:.6g} kg m^2 turned with the spin, whose torque the spin carries")
    if plan.mount is not None and plan.mount.rates is None:
        refuse("chain.mount-rates", RefusalClass.USER,
               f"{owner} carries its rotor's absolute spin, which needs the angular velocity of the frame it is "
               f"mounted on, and {plan.mount.segment!r} publishes none (apparatus.frame_angular_velocities), or does "
               f"not know its own mount's",
               f"publish apparatus.frame_angular_velocities from {plan.mount.segment!r}'s segment (in its "
               f"capabilities()), or use the kinematic rung of {owner!r}, Motor.Kinematic(...)")
        return
    if plan.mount is None:
        return
    host = placed(segments, plan.mount.segment, plan.mount.frame, transform(entry.placement(plan.settings)))
    if host.how == ROTOR:
        below = on_dynamic(segments, segments[host.segment])
        if below is not None:
            _articulated(refuse, f"{owner}, a rotor spun on the rotor of {host.segment!r}", host.segment, below)
        return
    if host.how == ARTICULATED:
        _articulated(refuse, f"{owner}'s rotor", host.segment, cast(str, on_dynamic(segments, segments[host.segment])))
        return
    if host.how != CARRIED:
        if parts:
            found.lines.append(f"{owner}'s rotor: {split.mass:.6g} kg, on a "
                               f"{'frame fixed in the world' if host.how == STILL else 'prescribed motion'}: its "
                               f"mass is not carried, and its tilt and imbalance are accepted there and not carried in "
                               f"its spin")
        return
    turned = host.placement.matrix() @ np.asarray(spinner.axis, dtype=np.float64)
    found.hosted.append(HostedSpinner(owner, host.segment, host.frame, as_vector3(turned), axial, spinner.top_speed))
    if not parts:
        return
    rung = (f"the kinematic rung of {host.segment!r}, whose motion no mounted mass changes, takes it "
            f"(Arm.Kinematic(...) or Motor.Kinematic(...))")
    if not (split.balanced and split.aligned) and not _residues(owner, named, split, spinner, host, rung, found,
                                                                refuse):
        return
    found.mounted.setdefault(host.segment, []).append(
        Payload(f"mounted:{owner}.rotor", host.frame, host.placement, split.carried(spinner.axis)))
    found.lines.append(f"{owner}'s rotor: {split.mass:.6g} kg and {split.across:.6g} kg m^2 across its axis (the mean "
                       f"of its two moments), welded on {host.segment}.{host.frame}")


def _residues(owner: str, named: str, split: RotorSplit, spinner: Spinner, host: Placed, rung: str, found: MassPlan,
              refuse: Refuse) -> bool:
    """Whether a rotor's imbalance and tilt on a carrying segment are residues (``RESIDUE_SHARE``), each reported;
    refused by name, with its size, where one is not, or where no top speed judges it."""
    if spinner.top_speed is None:
        what = f"its centre of mass {split.off_axis:.3g} m off its axis" if not split.balanced else \
            f"its inertia turned off its axis (its products of inertia with the axis reach {split.tilt:.3g} kg m^2)"
        refuse("chain.spinner-speed", RefusalClass.USER,
               f"{owner}'s rotor ({named}) has {what}, beyond rounding, and judging whether that is a residue needs "
               f"its top_speed, the most its spin reaches",
               f"declare it (top_speed=rad_per_s(...)), place the rotor's parts on its axis, or use the kinematic "
               f"rung: {rung}")
        return False
    top = spinner.top_speed
    weight = split.mass * STANDARD_GRAVITY
    force = split.mass * split.off_axis * top * top
    torque = split.tilt * top * top
    lever = STANDARD_GRAVITY * math.sqrt(split.mass * max(split.axial, 0.0))
    held = True
    if force > RESIDUE_SHARE * weight:
        refuse("chain.spinner-unbalanced", RefusalClass.USER,
               f"{owner}'s rotor ({named}) has its centre of mass {split.off_axis:.3g} m off its axis: an unbalanced "
               f"rotor, whose rotating force at its top speed {top:.6g} rad/s, {force:.3g} N, is {force / weight:.3g} "
               f"of its weight, above the residue of {RESIDUE_SHARE:g}: its weight and spin load "
               f"{host.segment!r} by terms a spinner does not carry",
               f"place its parts with their centre of mass on the axis (each Part(..., at=...)), or use the "
               f"kinematic rung: {rung}")
        held = False
    if torque > RESIDUE_SHARE * lever:
        refuse("chain.spinner-tilted", RefusalClass.USER,
               f"{owner}'s rotor ({named}) has its inertia turned off its axis (its products of inertia with the axis "
               f"reach {split.tilt:.3g} kg m^2): a tilted rotor, whose rotating torque at its top speed {top:.6g} "
               f"rad/s, {torque:.3g} N m, is {torque / lever:.3g} of its weight on its radius of gyration, above the "
               f"residue of {RESIDUE_SHARE:g}: its spin and {host.segment!r}'s turning couple by terms a spinner does "
               f"not carry",
               f"place its parts symmetric about the axis (each Part(..., at=...)), or use the kinematic rung: "
               f"{rung}")
        held = False
    if held:
        found.lines.append(f"{owner}'s rotor: its residues at its top speed {top:.6g} rad/s, not carried: a rotating "
                           f"force {force:.3g} N ({split.off_axis:.3g} m off its axis, {force / weight:.3g} of its "
                           f"weight) and torque {torque:.3g} N m ({split.tilt:.3g} kg m^2 of tilt, "
                           f"{torque / lever if lever > 0.0 else 0.0:.3g} of its weight on its radius of gyration), "
                           f"each accepted up to {RESIDUE_SHARE:g}")
    return held


__all__ = ["ARTICULATED", "CARRIED", "PRESCRIBED", "RESIDUE_SHARE", "ROTOR", "STILL", "MassPlan", "Placed",
           "on_dynamic", "placed", "plan_masses"]
