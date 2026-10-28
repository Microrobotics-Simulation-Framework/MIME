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
  - `references`, the bibliography keys of the works an entry rests on, each checked to give what the entry takes
    from it, `()` for none (a constant drive, a prescribed field, a definition). Every key of a MIME entry's is an
    entry of `docs/bibliography.bib`, and the check fails on one that is not, naming the entry and the key; a lab's
    keys are counted, not checked, since its bibliography is its own. A toy's is `()`: its law is made up for the
    tests;
  - `family`, on an interaction, the family whose suite it passes, which the family walk reads.

  A value the code does not establish is `NOT_ESTABLISHED`, never a guess, and is listed with its reason. A lab's
  is counted and named in the summary, not failed: its list is the lab's own.
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
  - every algorithm id is unique, and a 0.2 id is taken only by the class the frozen list names as keeping it
    (`docs/data/node_ids_0.2.yaml`). A node that changed in any way takes a new id, so the list keeps an id only for
    a 0.3 class that is the same node, with the evidence, and retires the rest, saying what changed; a class's name
    earns it nothing. Every 0.2 node changed or has no counterpart, so every 0.2 id is retired;
  - `hazard_hints` stays empty, and an order is declared once.

  Whether the code implements the equation is the oracle's to show; these checks do not.
- **The refusal scan and table.** Every code MIME constructs, with the classes it is refused in, is a row of
  `mime/build/refusal_table.txt`, generated from a scan of the source: codes at construction calls, in `(code,
  message, action)` tuples, and, by a cross-check of every string shaped like a code, those passed by a variable. A
  `Refusal` raises where it is constructed with a MIME code the table does not hold; a code in a registered namespace
  (`labx.too-far`) is accepted, uncatalogued, so a lab's entry is refused, never crashed. A code built where it is used
  (an f-string) fails the scan, which cannot read it: each code a construction can be is written as a literal, if
  need be in a mapping of literal codes. A code on two rows of the table fails, and a namespace named like the first
  word of a MIME code (`load`), or of a MIME entry's name (`gravity`), is refused, since it would take those codes out
  of the check, or that entry's name out of MIME's namespace.
- **The ratchets, kept as lists**: `docs/ratchets/<name>.txt`, one item per line as `item  # reason`. A hit not
  listed fails, and so does a listed item that now passes, naming its line to delete; a line with no reason, a
  duplicate, a malformed item and a bare count fail. No count is stored: each list's length is derived and printed.
  The floor of pinned symbol rows runs the other way: a listed row that is missing fails.
- **The switch scan** (`_switches.py`). Every registered node's step is traced in the configurations a caller
  builds (MIME's: script A's arms and the round-trip cases), and the functions of the package JAX records for its
  equations are its scope: code no step runs is never scanned. In scope, a raw non-smooth operation (called, or passed
  uncalled to `functools.partial`, a reduction or a map), a square root or power of a half not of a domain guard's
  output, a comparison cast to a number (`astype`, `asarray`, a dtype's constructor, `convert_element_type`, a sum)
  or in arithmetic, `stop_gradient`, `custom_jvp` and `custom_vjp` (as decorators, calls, through `partial`, or among
  a module's own statements) fail, and so does a switch function inside a transform, read from the traces; every
  owner passed must be declared and every declared owner passed, a lab's under its package's name, and an owner
  passed by position to a helper the scan can see fails. A hit not listed in `docs/ratchets/raw_switches.txt` fails,
  a listed site no longer hit fails, and a listed `stop_gradient` names the status it caps its entry at, or the test
  that clears it. A registered node class no configuration traces fails. It also lists every `domain_guard` and
  `nonzero_norm` call in scope, which the domain-guard test must reach. `mime docs check` holds every entry's status
  to the one its switches and listed sites give, each owner to one declaration and the list to its form; declared
  owners are names to the refusal scan.

  **What it does not see** (its blind spots): an operation reached through a name it does not resolve: a dispatch
  table (`OPS["max"](x, y)`), an attribute alias on an object (`self.mx = jnp.maximum`), `getattr(jnp, "maximum")`,
  a name rebound more than once, a method called on `self`; an owner passed by position to a method; a comparison cast
  where its value passes through an expression or a name bound more than once; and functions that are smooth but
  singular somewhere (a cube root, a logarithm or a division at zero, `arcsin` at one), which are no family. A raw
  operation in an untraced branch of a function in scope is scanned (the whole function is), and one in a lazy
  indicator is not.

  **Its scope rests on JAX's source information**, which is internal to JAX (`equation_frames` reads it, and the
  batching frames that show a `vmap`). It was measured on JAX 0.12.0.1, the lock's, and a test pins the minor
  release (`test_jax_is_the_version_the_source_information_was_measured_on`); `pyproject.toml` also allows v0.10 and
  v0.11, on which it is unvalidated, not known wrong. A traced step whose equations carry no frame at all fails the
  scan with that cause first, and nothing that rests on it.
- **The docs walk** (`_walk.py`), over a docs root it takes as a parameter. Every entry will have a generated page,
  so it checks the hand-written half: (a) a source exists where one is required, an entry's file
  `entries/<kind>/<name>.md`, a mechanism page `mechanisms/<target kind>.md` for each target kind, a `NodeMeta` and a
  symbol table on each node of MIME's an entry builds, and a card `assets/<name>.md` for each packaged asset; (b) every
  page source's id names something that exists, a link to a page source names one that does (an inline link or image,
  with a title or in angle brackets, a reference definition, or an HTML `href`), and an entry's file names the version
  it describes (front matter, `describes: <version>`, quoted where YAML would read a number), which must be the
  registered one; a front matter that is not YAML fails, naming the file. MIME's namespace
  is walked in full; `toy` is exempt by namespace; a lab's entries and nodes without a source are counted, not failed,
  unless the lab opts in (the walk's `held`). A walk that finds no entry of MIME's fails rather than pass on nothing.
- **The ledger**, `docs/unpaged.txt`: each required source that merged before the walk without it, `item  # reason`,
  generated from the registry when the walk landed. An item missing its source and not listed fails, and so does a
  listed item that has its source now, so the ledger only shrinks; a line added needs its reason logged.
- **The number rule** (`_numbers.py`): no measured, computed or fitted number in a hand-written page source or a
  package README; a number such a file needs is a claim record's, named by its id (`see MIME-ANO-114`). A line carrying
  a number token fails unless listed, keyed by its file, its number tokens in order and its occurrence index among the
  file's lines with the same tokens (computed, never stored): a wording edit keeps an entry, and a changed, added,
  removed or swapped number, or a copy of a listed line, is a new key. `docs/ratchets/number_allowlist.txt` lists a
  line with its own reason, and `docs/ratchets/readme_numbers.txt` is the READMEs' frozen start, which only shrinks:
  with `--base <revision>` (CI's is `origin/v0.3`), a line the base's list does not hold fails. What one token is, and
  what is exempt (a code fence, an identifier of MIME's by its prefix, a version, a release the code reads after its
  product's name, a phase name, an ordered list's count, an idiom of `docs/number_idioms.txt`), is printed with every
  run, and so are the releases; an unclosed code fence fails, since it would hide the rest of its file. A number in an equation, a node's `governing_equations`, is a
  symbol bound to a parameter or comes from a claim record; no check holds that yet, so review does. A number spelled
  in words is invisible to the rule: a warning names each one before a unit, "per cent", "times" or "fold", and fails
  nothing.
- **The pages** (`mime docs build`; `_build.py`, with a generator for each page type). The pages are never committed:
  `mime docs build` writes `index.md` and `pages/` to `docs/_build/` (untracked), or to the directory `--out` names, and
  CI builds them, scans them and uploads them as a run artifact. A page is a function of the registry export, the
  vocabulary, the registered families and the hand-written sources, so two builds of one checkout are the same text:
  - an **entry page** for every registered entry whatever its namespace, generated from the export (every class
    variable and field, with its type and default), so a lab's entry has one with no work; its hand-written part, when
    its source exists, is printed in it, and when it does not the page says so and the build counts it. An "at a
    glance" block at the top links the node page's equation and the family page, and copies neither;
  - a **node page** for every node class an entry builds: its `NodeMeta` (the equations' only home, printed once), its
    symbol table, and the entries that build it; a node with no `NodeMeta` gets a page that says it is undocumented;
  - a **family page** for every registered family: the contract its harness holds, its members, and, once two built
    rungs can replace each other, a comparison in which no cell is blank (`not measured` or `declared`, counted);
  - a **mechanism page** for every target kind: how its deliveries combine, its interfaces, the dynamics that accept it
    and the entries that deliver into it;
  - the **vocabulary** page, every interface with its kind, units, frame, shape, sign, typical magnitude and
    description, a reserved name marked; the **index**, a front door by goal; and the hand-written **concepts** note.

  What the build refuses: a hand-written source whose entry, version or id no longer resolves (it exits non-zero naming
  the file and writes no page, rather than skip it and leave a page missing), a display equation on two pages, and a
  page that holds an absolute path of the machine that built it. A path the build would write is made relative to the
  checkout. An empty equations, assumptions or references field prints `none declared` and is counted, with the other
  hatches (`undocumented`, `declared, unchecked`, `not measured`, `free text, not parsed`, `no hand-written part`), in
  the build's summary; and `mime docs check` lists the fields of MIME's own (`docs/ratchets/none_declared.txt`): a new
  empty field fails until it has its own line and reason, and a listed one that is filled fails until its line is
  deleted. The toys' and a lab's fields are counted, not listed.
- **The citation check** (`_citations.py`). A citation is read by Pandoc's grammar: `@key`, `[@key]`, `[@a; @b]`,
  `[see @key, p. n]`, `[-@key]` and `@{any key}`; code, links, URLs and an `@` after a word are not citations. The
  check counts the citations it checked against `docs/bibliography.bib` and the ones it did not find, and fails when
  there are pages and none holds a citation. It reads the built pages and every other hand-written `.md` under `docs/`
  (a file or directory named `_*` is a template, and `_build` the build's output). A lab's pages are counted and not
  judged: its bibliography is its own. A string from code that holds an `@` after a space and a word (`an @dataclass
  law`) reads as a citation of that word and fails the build; write it in backticks.
- **`mime docs check`**: all of the above over a checkout (`--root`, and `--base` for the start list), the pages built
  in this process for the citation check and for the number rule's reading of the text they print from code (a
  validity line, a gradient record's text, a node's assumptions, an interface's description: the pages of MIME's
  namespace are judged, the others counted), the pages built in fresh processes under eight hash seeds, which must
  give byte-equal pages (that is all it bounds: a clock is kept out of the generators by a test of their imports, and
  address-dependent order, directory order and locale are not bounded), then
  `scripts/check_anomalies.py` unchanged. It builds no graph: the checks that need a built node run in the node walk.
  Its summary names every entry whose derivative is not exact, with its record, counts and names a lab's values not
  established, the keys it checks against the bibliography and a lab's it only counts, and prints the walk's counts,
  each list's length and the word-numeral warnings. It needs the `verify` extra, and without it exits naming it.

## Boundaries

- `docs` may import `testing`; `testing` never imports `docs`.
- Nothing `mime registry --json` imports needs the `verify` extra, so it runs on a base install, and `mime.cli`
  imports `docs` only inside its commands.
- The wheel ships these modules; the hand-written sources in the repository's `docs/` do not ship.
- The pages need the `verify` extra (the families and the node walk live in `mime.testing`); `mime docs build` exits
  naming it without. Nothing the export imports reaches the pages' modules.

## Status

Since dev1's first documentation phase: the export, the entry fields declared for every registered entry, `NodeMeta`
and the symbol-table form with their checks (on test-only nodes until the first node's table lands), the refusal
table with its constructor check, the lists, `mime docs check`, the docs walk with its ledger, and the number rule.
Since the generators' phase: the page generators and `mime docs build`, the vocabulary, the index and the concepts
note, the citation parser, and the number rule over the text the pages print.

Since phase 7e, a geometry kind's entry page has **capability rows**, generated from which of `Geometry`'s optional
methods it implements (the export's `capabilities`), with its native form and its placement settings.

Not yet generated: the plan-only items of an entry page (confinements, registrations and groups, which come from
running `plan` on a test fixture), a worked fragment from the fixture CI builds, the state, parameter, input and output
tables of a node page (they come from a built node), the performance profile and the claims citing a node (their
records do not exist), the lag classes and group settings on a mechanism page, the asset card, the experiment card, the
refusal catalogue, and the index's part for scripts (the index lists entries by kind, by target and the families).

Phases and releases are explained in the [package manifest](../README.md#status).
