# `mime.docs`

**The machine-readable sources MIME's documentation is generated from, and the checks that keep them true.**

A page is generated from what the code declares, never typed beside it, so these sources come first: an entry's
fields, a node's `NodeMeta` and symbol table, and the refusal table. The pages are generated from them later.

## What goes here

- **The registry export**, `mime registry --json`: every registered entry, MIME's and each installed `mime.entries`
  package's, with every `ClassVar` and dataclass field it declares, each with its type, its default and its value.
  It runs in a fresh process, so an entry a test registered on its own never appears. A namespace that needs the
  `verify` extra is listed as not loaded; any other failure to import one is an error. A completeness test reads each
  entry class itself and fails on a value held at class level that the export does not carry.
- **The entry fields the pages need**, declared on every entry (`mime.registry`):
  - `builds`, each rung's node classes (`""` for an entry without rungs), which the node walk checks against what a
    round-trip case builds;
  - `gradient`, a `GradientRecord` of where the derivative is not exact, why, and the route to `exact`, for every
    status but `exact`; a `cut` or `unverified` entry states its reason and its route;
  - `references`, the bibliography keys of the works an entry rests on;
  - `family`, on an interaction, the family whose suite it passes, which the family walk reads.

  A value the code does not establish is `NOT_ESTABLISHED`, never a guess, and is listed with its reason.
- **`NodeMeta` and the symbol table**: MADDENING's `NodeMeta` on a node class, whose `governing_equations` are the
  equations' one home, and `symbols`, a row per symbol (`mime.core.Symbol`: its LaTeX, kind, name, units and
  `implemented_by`). The checks bound names, units and existence:
  - each row's symbols occur in the equations as tokens, never as substrings;
  - each row's name is declared where its kind says, with the units the code declares there (checked on a built
    node, in the node walk);
  - `implemented_by` names a function, method or property where it is defined: a module, or the own `__dict__` of
    the node's class or one of its bases. A class is not accepted;
  - each equation symbol outside a braced subscript has a row (`x_{i}` needs none for `i`; `x_i` is listed);
  - every pinned row is still in its node's table (the floor);
  - every algorithm id is unique, a 0.2 id taken only by the node that held it (`docs/data/node_ids_0.2.yaml`);
  - `hazard_hints` stays empty, and an order is declared once.

  Whether the code implements the equation is the oracle's to show; these checks do not.
- **The refusal scan and table.** Every code MIME constructs, with the classes it is refused in, is a row of
  `mime/build/refusal_table.txt`, generated from a scan of the source: codes at construction calls, in `(code,
  message, action)` tuples, and, by a cross-check of every string shaped like a code, those passed by a variable. A
  `Refusal` raises where it is constructed with a MIME code the table does not hold; a code in a registered namespace
  (`labx.too-far`) is accepted, uncatalogued, so a lab's entry is refused, never crashed.
- **The ratchets, kept as lists**: `docs/ratchets/<name>.txt`, one item per line as `item  # reason`. A hit not
  listed fails, and so does a listed item that now passes, naming its line to delete; a line with no reason, a
  duplicate, a malformed item and a bare count fail. No count is stored: each list's length is derived and printed.
  The floor of pinned symbol rows runs the other way: a listed row that is missing fails.
- **`mime docs check`**: all of the above over a checkout (`--root`), then `scripts/check_citations.py` and
  `scripts/check_anomalies.py` unchanged. It builds no graph: the checks that need a built node run in the node walk.
  It needs the `verify` extra, and without it exits naming it.

## Boundaries

- `docs` may import `testing`; `testing` never imports `docs`.
- Nothing `mime registry --json` imports needs the `verify` extra, so it runs on a base install, and `mime.cli`
  imports `docs` only inside its commands.
- The wheel ships these modules; the hand-written sources in the repository's `docs/` do not ship.

## Status

Since dev1's first documentation phase: the export, the entry fields filled for every registered entry, `NodeMeta`
and the symbol-table form with their checks (on test-only nodes until the first node's table lands), the refusal
table with its constructor check, the lists, and `mime docs check`. The docs walk, the number rule, the pages and
their generators come next.

Phases and releases are explained in the [package manifest](../README.md#status).
