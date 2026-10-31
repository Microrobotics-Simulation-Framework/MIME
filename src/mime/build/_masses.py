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

A spinner on a carrying segment has its rotor checked: refused when tilted or unbalanced, and when unequal across its
axis beyond the threshold below; its mass and mean inertia across the axis are welded on the host's frame, and its
moment about the axis rides on its spin."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from mime.core.multibody import IDENTITY, Inertial, Payload, RigidTransform, combined, split_rotor
from mime.build._refusals import RefusalClass
from mime.core.multibody._urdf import ISSUES
from mime.registry import ChainSegment, Spinner

if TYPE_CHECKING:
    from mime.build._plan import EntryPlan

CARRIED, ROTOR, STILL, PRESCRIBED, ARTICULATED = "carried", "rotor", "still", "prescribed", "articulated"

#: The share of the gyroscopic torque a spinner carries, at its worst case, beyond which the worst case of the term
#: it does not carry (its rotor's inequality across its axis) refuses it on a dynamic segment. The term not carried
#: averages to zero over every half turn of the rotor on its host, so the turn-averaged reaction is exact; at a half,
#: the instantaneous reaction is still mostly the carried term at every instant. A rotor beyond it needs its own joint
#: in the host's tree (a grafted jointed sub-tree, logged for later), or the kinematic rung.
ASYMMETRY_THRESHOLD = 0.5

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
           f"mass; or mount the mass rigidly on {host!r}'s frames; jointed sub-trees grafted on a link are logged for "
           f"later: ask at {ISSUES}")


@dataclass
class MassPlan:
    """What the walk found: the rigid bodies to graft on each carrying segment, each spinner's moment about its axis,
    and the build report's lines."""

    mounted: dict[str, list[Payload]]
    spin_inertia: dict[str, float]
    lines: list[str]


def plan_masses(segments: Mapping[str, EntryPlan],
                parts: Sequence[tuple[str, str, tuple[float, float, float], Inertial]], refuse: Refuse) -> MassPlan:
    """``parts`` are each part with a mass that is mounted on a chain: its id, the frame it is mounted on
    (``<segment id>.<frame>``), its centroid there, and its inertial in its own frame (its centroid at its origin)."""
    found = MassPlan({}, {}, [])
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
    if plan.mount is not None and plan.mount.rates is None:
        refuse("chain.mount-rates", RefusalClass.USER,
               f"{owner} carries its rotor's absolute spin, which needs the angular velocity of the frame it is "
               f"mounted on, and {plan.mount.segment!r} publishes none (apparatus.frame_angular_velocities), or does "
               f"not know its own mount's",
               f"publish apparatus.frame_angular_velocities from {plan.mount.segment!r}'s segment, or use the "
               f"kinematic rung of {owner!r}")
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
                               f"mass is not carried, and its tilt, imbalance and inequality across its axis are "
                               f"accepted there and not carried in its spin")
        return
    if not parts:
        return
    carrier = segments[host.segment]
    rung = f"the kinematic rung of {host.segment!r}, whose motion no mounted mass changes, takes it"
    if not split.balanced:
        refuse("chain.spinner-unbalanced", RefusalClass.USER,
               f"{owner}'s rotor ({named}) has its centre of mass {split.off_axis:.3g} m off its axis: an unbalanced "
               f"rotor, whose weight and spin load {host.segment!r} by terms a spinner does not carry",
               f"place its parts with their centre of mass on the axis, or use the kinematic rung: {rung}")
        return
    if not split.aligned:
        refuse("chain.spinner-tilted", RefusalClass.USER,
               f"{owner}'s rotor ({named}) has its inertia turned off its axis (its products of inertia with the axis "
               f"reach {split.tilt:.3g} kg m^2): a tilted rotor, whose spin and {host.segment!r}'s turning couple by "
               f"terms a spinner does not carry",
               f"place its parts symmetric about the axis, or use the kinematic rung: {rung}")
        return
    if not split.symmetric:
        turning = cast(ChainSegment, carrier.entry).turning_bound(host.frame)
        if turning is None or spinner.top_speed is None:
            missing = "its top_speed" if spinner.top_speed is None else \
                f"a speed limit on every joint that turns {host.segment}.{host.frame}"
            refuse("chain.spinner-speed", RefusalClass.USER,
                   f"{owner}'s rotor ({named}) is unequal across its axis (moments {split.moments[0]:.6g} and "
                   f"{split.moments[1]:.6g} kg m^2), and the worst case of the term not carried needs {missing}",
                   f"declare it (top_speed=rad_per_s(...), or the joints' velocity limits in the description), make "
                   f"the rotor symmetric about its axis, or use the kinematic rung: {rung}")
            return
        top = spinner.top_speed
        neglected = 2.0 * split.asymmetry * turning * (top + turning)
        carried = axial * top * turning
        share = neglected / carried if carried > 0.0 else float(np.inf)
        found.lines.append(
            f"{owner}'s rotor across its axis: moments {split.moments[0]:.6g} and {split.moments[1]:.6g} kg m^2; "
            f"the term not carried at worst 2 x {split.asymmetry:.6g} kg m^2 x {turning:.6g} rad/s x ({top:.6g} + "
            f"{turning:.6g}) rad/s = {neglected:.6g} N m, from {host.segment}.{host.frame}'s joint speed limits and "
            f"the top speed; the gyroscopic torque carried at worst {axial:.6g} x {top:.6g} x {turning:.6g} = "
            f"{carried:.6g} N m: a share of {share:.3g}, accepted up to {ASYMMETRY_THRESHOLD}")
        if share > ASYMMETRY_THRESHOLD:
            refuse("chain.spinner-asymmetric", RefusalClass.USER,
                   f"{owner}'s rotor ({named}) is unequal across its axis (moments {split.moments[0]:.6g} and "
                   f"{split.moments[1]:.6g} kg m^2): at worst the term not carried is {neglected:.3g} N m, "
                   f"{share:.3g} of the gyroscopic torque carried, above {ASYMMETRY_THRESHOLD}",
                   f"make the rotor nearer symmetric about its axis, lower its top_speed only if it never spins "
                   f"faster, or use the kinematic rung: {rung}")
            return
    found.mounted.setdefault(host.segment, []).append(
        Payload(f"mounted:{owner}.rotor", host.frame, host.placement, split.carried(spinner.axis)))
    found.lines.append(f"{owner}'s rotor: {split.mass:.6g} kg and {split.across:.6g} kg m^2 across its axis (the mean "
                       f"of its two moments), welded on {host.segment}.{host.frame}")


__all__ = ["ARTICULATED", "ASYMMETRY_THRESHOLD", "CARRIED", "PRESCRIBED", "ROTOR", "STILL", "MassPlan", "Placed",
           "on_dynamic", "placed", "plan_masses"]
