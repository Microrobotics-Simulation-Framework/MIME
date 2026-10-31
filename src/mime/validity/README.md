# `mime.validity`

**What a result can be trusted for.**

## What goes here

- **Validity lines**: one per assumption, each with its covered range.
- **Watches**: run-time checks that report and never halt.
- **Evidence** in the build report.
- **MIME's profile lines**, for `exploratory`, `research` and `strict`, merged with OpenMaterica's.
- **The stamp** a build carries, and `exp.status()`.

## The rule

A model outside its validated range is reported, never silently switched for another. `strict` holds a higher
evidence standard. It confers nothing by itself.

## Status

Since phase 7l-a (designs #190 and #191), **the profiles and the tables they act by**, built on a1's seams:

- `Profile`: `exploratory` (the default), `research`; `strict` refused by name (`profile.reserved`) until a1.
- **Classes as data** (`finding_classes.txt`, `FindingClasses`): each finding code's class, and each class with what
  covers it (design #188's covered-by column). dev1 classes the codes MIME meets today: OpenMaterica's
  `condition_unstated` and `stated_without_origin` are assumptions; a pending reason, a numerical default taken, a
  physical value with no source and a numerical value with no evidence are the build's own. Every other code keeps its
  behaviour from before profiles.
- **The profile rules** (`profile_rules.txt`, `ProfileRules`): (profile, class) to generate, basis, list or refuse. The
  build's code reads a finding's class and its rule (`Judge`), never a code, so a1 adds classes and rows as data; a
  test adds one and needs no other edit.
- **Profiles act on waivers alone** (`waivers_alone`): a rule covers a finding with a waiver, generated with a pending
  basis or the experiment's own, and nothing else; the build refuses any other product (`profile.not-a-waiver`).
- **A reason's basis** (`Basis`: a text, `chosen(...)`, `pending()`), one type for a waiver's basis and a value's
  source, so a1's named bases serve both; the saved form keys each kind by its own name.
- **What is pending** (`PendingItem`), through one formatter (`pending_text`) with the slot where a reason goes, which
  a1's drafts fill; and **the stamp** (`Stamp`) every output carries.

Validity lines, watches and evidence records are a1's.

Phases and releases are explained in the [package manifest](../README.md#status).
