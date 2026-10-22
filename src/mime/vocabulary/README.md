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

MIME's own interfaces arrive with the first physics, in phase 4. The only entries so far are the toys' (namespace
`toy`, in `mime.testing.toys`). Contribution classes come with the families that need them.

Phases and releases are explained in the [package manifest](../README.md#status).
