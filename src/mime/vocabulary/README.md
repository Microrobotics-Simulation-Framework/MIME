# `mime.vocabulary`

**The names of every interface in MIME, and what each one means.**

A dynamics exposes ports, and an interaction reads some of them and delivers into others. Every one of those ports
has an entry here. The build checks every read and every target against this list, so a mismatch is caught before
any graph exists.

## What goes here

- **Interface entries.** Each has a kind, units, frame, shape, sign, dtype and a typical magnitude. Entries are
  versioned per namespace.
- **The six target kinds** an interaction can deliver into:
  - load, kinematic and operator, which are summed (additive);
  - equality constraint, which is exclusive;
  - inequality constraint (confinement);
  - registration, where the receiving dynamics includes the thing in its own update, as a flow solver includes an
    immersed body.
- **Reserved names.** A name that is planned but not built fixes its kind, units, frame and sign now. Only its shape
  stays open. A later addition then cannot clash with what exists, and using it before it is built is refused with
  that reason.
- **Contribution classes**, and the **closed list of capability flags**. Support tables match on these flags, never
  on model names, so a lab's own model can slot in.

## Boundaries

- Pure data, with no MADDENING or physics code.
- Everything above it reads it: `build`, the entries, `saved`.

## Status

Landed in dev1's phase 2:
- `Interface` entries with kind, units, frame, shape, sign, dtype and typical magnitude; `define` and `lookup`;
- namespaces with their vocabulary versions (`register_namespace`, `versions`); MIME's own names are unprefixed;
- the six target kinds (`TargetKind`) and the two read kinds (`ReadKind`);
- the closed list of capability flags (`Capability`);
- reserved entries, whose use the build refuses as "reserved, not built".

Since phase 4b, MIME's own names for a body and the magnetic field (`_mime.py`): `body.pose` `(7,)` and `body.twist`
`(6,)` (state); `body.applied_wrench` and `body.background_load` (loads, `(6,)`), `body.resistance` (an operator,
`(6, 6)`) and `body.confinement` (an inequality constraint); `field.magnetic.flux_density` `(3,)` and
`field.magnetic.gradient` `(3, 3)`. Every one is in the world frame, a body's wrench and twist about its centre of
mass; a composite's units list its parts (`N; N m`). And the flag `Capability.UNIFORM`, for a field that is the same
everywhere. Contribution classes come with the families that need them.

Since dev1's first documentation phase, `register_namespace` refuses a name that is the first word of one of MIME's
refusal codes (`load`, `drag`): the namespace would take those codes for its own, and a refusal under them would no
longer be checked against MIME's refusal table. And, since its follow-ups, a name that is the first word of one of
MIME's registered entries (`gravity`, `magnetic`), which `namespace_of` would then answer for that entry.

Since phase 7a2, `apparatus.frame_poses` (state: a row per frame a chain segment names, in its order, each a pose)
and `apparatus.link_wrench` (a load on a chain's link, reserved for the dynamic arm).

Since phase 7a3, `apparatus.link_wrench` is built: a load target, a row per frame of the segment, each a force and a
torque about the frame's origin, which a dynamic segment accepts; and `apparatus.joint_velocities` (state, a dynamic
segment's joint rates).

Since phase 7a4, `apparatus.frame_angular_velocities` (state: a row per frame a chain segment names, its angular
velocity in the world, `rad/s`), which every MIME segment publishes on every rung and a mounted spinner reads.

Phases and releases are explained in the [package manifest](../README.md#status).
