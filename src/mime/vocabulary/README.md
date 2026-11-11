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
- **Port kinds and energy classes** (`_ports.py`): what the energy ledger pairs each delivery by, and the class a law
  is. A port kind names a receiver's flow and, for each interface it accepts, what a delivery into it is
  (`DeliveryKind`: a load read at the step's start, a force the step solves for, a flow handed in, a delivered
  inertia); each delivery's units pair with the flow's to watts, part by part, checked when the kind is made. Kinds
  are registered (`register_port_kind`), open to labs. A law is dissipative, potential or workless
  (`EnergyClass`); there is no source class. A dissipative law's operators are definite by construction or at the
  state (`Definiteness`).
- **Contribution classes**, and the **closed list of capability flags**. Support tables match on these flags, never
  on model names, so a lab's own model can slot in.

## Boundaries

- Pure data, with no MADDENING or physics code.
- Everything above it reads it: `build`, the entries, `saved`.

## Status

The first form:
- `Interface` entries with kind, units, frame, shape, sign, dtype and typical magnitude; `define` and `lookup`;
- namespaces with their vocabulary versions (`register_namespace`, `versions`); MIME's own names are unprefixed;
- the six target kinds (`TargetKind`) and the two read kinds (`ReadKind`);
- the closed list of capability flags (`Capability`);
- reserved entries, whose use the build refuses as "reserved, not built".

MIME's own names for a body and the magnetic field (`_mime.py`): `body.pose` `(7,)` and `body.twist`
`(6,)` (state); `body.applied_wrench` and `body.background_load` (loads, `(6,)`), `body.resistance` (an operator,
`(6, 6)`) and `body.confinement` (an inequality constraint); `field.magnetic.flux_density` `(3,)` and
`field.magnetic.gradient` `(3, 3)`. Every one is in the world frame, a body's wrench and twist about its centre of
mass; a composite's units list its parts (`N; N m`). And the flag `Capability.UNIFORM`, for a field that is the same
everywhere. Contribution classes come with the families that need them.

`register_namespace` refuses a name that is the first word of one of MIME's
refusal codes (`load`, `drag`): the namespace would take those codes for its own, and a refusal under them would no
longer be checked against MIME's refusal table. And, since its follow-ups, a name that is the first word of one of
MIME's registered entries (`gravity`, `magnetic`), which `namespace_of` would then answer for that entry.

`apparatus.frame_poses` (state: a row per frame a chain segment names, in its order, each a pose)
and `apparatus.link_wrench` (a load on a chain's link, reserved for the dynamic arm).

`apparatus.link_wrench` is built: a load target, a row per frame of the segment, each a force and a
torque about the frame's origin, which a dynamic segment accepts; and `apparatus.joint_velocities` (state, a dynamic
segment's joint rates).

`apparatus.frame_angular_velocities` (state: a row per frame a chain segment names, its angular
velocity in the world, `rad/s`), which every MIME segment publishes on every rung and a mounted spinner reads; and two
operators a dynamic segment takes from a spinner mounted on it, `apparatus.link_momentum` (a row per frame, the
rotors' momentum about their axes, `N m s`, whose gyroscopic torque the segment applies) and `apparatus.link_inertia`
(a row per frame, the rotors' inertia beyond the frame's own at the step's start and end, in kg m²).

The port kinds and energy classes: `PortKind`, `register_port_kind`, `port_kind`, `port_kinds`; MIME's six kinds,
`rigid_body` (a body's twist, against its applied wrench and background load read at the step's start, and its
resistance and confinement solved for in its step), `joint` (a dynamic arm's joint rates, against the loads on its
frames, which reach them through its Jacobian, its spinners' momenta, whose gyroscopic torque it returns at the rates
it applies it at, and the inertia its frames carry, paired by its change), `spinner` (a dynamic motor's rotor, against
the loads on its frames, of which the rotor's axial torque is its own and the rest passes to its host),
`delivered_inertia`, `flow_in` and `fluid` (a medium's fluid's momentum, `flow.momentum`, against the body force per
unit mass each law driving it delivers, and the drive's waveform, which reaches it through the flow's own evaluation at
its stage instants) and `opening` (a flow's openings' outward volume fluxes, `flow.opening_flux`, a row per opening,
against their pressures: a pressure row's offset a load, `flow.opening_pressure`, its block solved,
`flow.opening_impedance`; a flux row is held in the flow's step, as the laws at its openings are, registered with it
through `flow.opening`, so no delivery pairs through it yet: its laws are registrations, booked as the flow's
hosted rows); `DeliveryKind`, `EnergyClass`,
`Definiteness`, and `POWER` and `ENERGY`, what a delivery and its flow pair to.

**What a kind checks, and who reads it.** A kind fixes the flow and the delivery kind of each interface it accepts,
and pairs their units to watts; shape and sign live on the interface entries and are not compared, and an interface
listed in `mapped` skips the unit pairing (its map is the receiver's own: a dynamic arm's Jacobian, a motor's
axis, a flow's evaluation of a waveform). With the energy ledger attached the build reads a metered
dynamics' kinds (`Dynamics.port_kinds`, each delivery paired by the one kind naming its interface) for how each delivery into it is paired, and the receiver's node refuses a delivery
paired otherwise than it pairs that input; it reads each law's class and definiteness for its records and its run-time
check. A declaration settles nothing by itself: each class is checked by an exact identity on the law alone (the energy
case of its family's suite, `mime.testing`), and the registry walk requires every entry of MIME's to declare or state
why it is unmetered.

Releases are explained in the [package manifest](../README.md#status).
