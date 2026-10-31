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
  default its entry declares nothing of (`undeclared`), a physical value with no source (a physical convention taken
  included) and a numerical value with no evidence are the build's own. Every other code keeps its behaviour from
  before profiles.
- **The profile rules** (`profile_rules.txt`, `ProfileRules`): (profile, class) to generate, basis, list or refuse. The
  build's code reads a finding's class and its rule (`Judge`), never a code, and **one function applies every action
  at every site** (`decide`): a read's unknown reason, a pending reason, a default taken, a value's missing source or
  evidence. A class with no rule in a profile is what its site did before profiles. a1 adds classes and rows as data;
  a test applies every action to every class at every site, and another adds a class and a row with no other edit.
- **Profiles act on waivers alone** (`waivers_alone`): a rule covers a finding with a waiver, generated with a pending
  basis or the experiment's own, and nothing else; `rule_problems` adds the rows no site can apply (a `list` of what
  only a declaration covers, a class with no code), and a build under tables with any is refused (`profile.tables`);
  the build refuses any other product a rule makes at run time (`profile.not-a-waiver`).
- **A reason's basis** (`Basis`: a text, `chosen(...)`, `pending()`), one type for a waiver's basis and a value's
  source, so a1's named bases serve both; the saved form keys each kind by its own name.
- **What is pending** (`PendingItem`), one item per thing pending (a read's reasons in one acceptance), in a canonical
  order (by target, then code), through one formatter (`pending_text`) with the slot where a reason goes, which a1's
  drafts fill, the imports its lines need and the route a sealed document takes; and **the stamp** (`Stamp`) every
  output carries, with what was measured of a numerical default.

Validity lines, watches and evidence records are a1's.

Phases and releases are explained in the [package manifest](../README.md#status).
