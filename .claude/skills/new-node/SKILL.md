# New Node Skill — MIME

This skill guides the agent through adding a new `MimeNode` subclass and the entry that builds it. A node arrives with
its documentation's sources (its `NodeMeta` and symbol table, the entry's fields and hand-written file), which
`mime docs check` checks; the pages are generated from them.

## Trigger

When the user asks to add a new physics node, or invokes `/new-node`.

---

## Step 0 — Place the node and take its id

- The node goes in `src/mime/nodes/<body, actuation, magnetic, hydro or acoustic>/`, beside the entry that builds it
  (`src/mime/interactions/`, `dynamics/`, `fields/`, `observations/`). Nodes import nothing above `mime.nodes`.
- **Its `algorithm_id` is the next free `MIME-NODE-nnn`, from `MIME-NODE-104` onward.** An id names one node for
  good, so a node that changed in any way from a 0.2 node takes a new id: every 0.2 id is retired
  (`docs/data/node_ids_0.2.yaml`), and the list keeps an id only for a 0.3 class it names as `kept_by`, with its
  evidence. A class's name earns it nothing. When an id is missing, retired or taken, `mime docs check` names the
  next free one, past the frozen list's ids and every id the walked nodes use.

---

## Step 1 — Implement the node

```python
"""MyNode -- one-line description."""

from collections.abc import Mapping
from typing import Any, override

from jax import Array
from maddening.core.compliance.metadata import NodeMeta

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, node_fields, node_input, typed_view


class MyNode(MimeNode):
    """The contract: what it computes and from what. The equations' one home is `meta.governing_equations`."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-104",
        governing_equations=r"\mathbf{F} = -c \, \mathbf{v}",
        # discretization, assumptions, limitations, validated_regimes, references as the node has them;
        # hazard_hints stays empty; leave discretization_order None where the class defines the hook.
    )
    symbols = (
        Symbol(r"\mathbf{F}", SymbolKind.STATE, "force", "N", "mime.nodes.hydro._my_node.MyNode.advance"),
        Symbol("c", SymbolKind.PARAMETER, "drag", "kg s^-1", "mime.nodes.hydro._my_node.MyNode.advance"),
        Symbol(r"\mathbf{v}", SymbolKind.INPUT, "velocity", "m/s", "mime.nodes.hydro._my_node.MyNode.advance"),
    )

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        ...  # pure JAX
```

- Type its state and inputs as `src/mime/nodes/README.md` sets out: a closed TypedDict each, inside MADDENING's
  signatures, read with `typed_view` and returned with `node_fields`.
- **Every non-smooth operation goes through `mime.core.switch`**, never raw: `switch.maximum`, `minimum`,
  `absolute`, `clip`, `norm`, `where` and `predicate` for a switch, each with its `owner` (`<entry>.<switch>`), its
  `resolution` and its `indicator` (each a value already computed there, or a lambda, which only a collector
  evaluates); `switch.domain_guard` for a selection that only keeps an undefined value out of a discarded branch;
  `switch.static_select` for a mask fixed at build. A square root takes a domain guard's output. No switch function
  inside a `lax.cond`, `scan`, loop, `vmap`, `checkpoint` or nested `jit`. A core helper takes its owner from its
  caller. `src/mime/core/README.md` says how; the switch scan (`tests/saved/test_switch_scan.py`) fails on a raw one
  a step runs that `docs/ratchets/raw_switches.txt` does not list.
- **The symbol table** (`mime.core.Symbol(latex, kind, name, units, implemented_by)`): a row per symbol of the
  equations. Each row's symbol occurs in `governing_equations` as a token; its `name` is declared where its kind says
  (`param_specs()`, `initial_state()`, `boundary_input_spec()`, `boundary_flux_spec()`, the building entry's
  dimensioned settings, or a module constant's dotted path) with the units the code declares there; `implemented_by`
  names a function, method or property where it is defined: a module-level function, or the node's class or one of
  its bases. Never a class.
- **The lists** (`docs/ratchets/`, `item  # reason`): pin every row in `symbol_rows.txt` when the table lands
  (`MyNode <latex>  # pinned at landing`); an index outside a braced subscript (`x_i`, not `x_{i}`) goes in
  `symbol_indices.txt`; a row whose units the code does not declare goes in `symbol_units.txt`. `mime docs check`
  names the line to add or delete.
- **A refusal code** the node or its entry raises is a row of `src/mime/build/refusal_table.txt`, written as a literal
  (a lookup in a mapping of literal codes where it is chosen), never built with an f-string; a code no test names goes
  in `docs/ratchets/refusal_codes.txt` until one does.

---

## Step 2 — The entry that builds it

- **`builds`**: each built rung's node classes, `""` for an entry without rungs (`builds = {"": (MyNode,)}`). The
  node walk checks it against what the entry's round-trip case builds; a rung no case builds goes in
  `docs/ratchets/builds_unchecked.txt`.
- **`gradient`**: a `GradientRecord` whose `switches` declare every owner the entry's node evaluates, each a
  `switch.Declaration` (the indicator with its units, orders per output group, a time switch's settings, the route to
  `exact`), with a row in `tests/saved/test_switch_harness.py` that measures its orders. The status is derived from
  them: `piecewise_jump` with a jump (order -1), `piecewise_kink` with kinks only, `exact` with none. A status no
  switch gives (`fixed_key`, `unverified`, `cut`) declares `where`, `reason` and `route` instead; a `cut` or
  `unverified` entry states its reason and route, and needs the maintainer's logged decision.
- **`references`**: keys of `docs/bibliography.bib`, each checked to give what the entry takes from it (`mime docs
  check` fails on a missing key); `()` only where the entry rests on no work, with the reason in the pull request. A
  bibliography entry added has every field checked against its DOI record or publisher, and the URL recorded.
- **`family`**, on an interaction: the family whose suite it passes.
- A value the code does not establish is `NOT_ESTABLISHED`, never a guess, with a line in
  `docs/ratchets/not_established.txt`.
- **The entry's hand-written file**, `docs/entries/<kind>/<name>.md` (`<kind>` with `_` for a space, the dotted name
  as the file name): what it models and when not to use it, its version history, at most a screen, starting with the
  version it describes:
  ```markdown
  ---
  describes: 0.1.0
  ---
  ```
  The docs walk fails without it: an entry registered after the walk landed never goes on the ledger
  (`docs/unpaged.txt`). A new target kind needs `docs/mechanisms/<kind>.md`, a packaged asset its card
  `docs/assets/<name>.md` (or, for a table landing at 7f to 7i, a ledger line with its reason logged, paid at 7i).
- **No measured number** in that file or a package README: a number is a claim record's, named by its id (`see
  MIME-ANO-nnn`). The number rule fails on a number token outside a code fence; `mime docs check` prints what one token
  is. A line that must carry one is listed in `docs/ratchets/number_allowlist.txt` with its reason, by a logged
  decision.
- A lab's namespace may not be the first word of a MIME entry's name, nor of a MIME refusal code.

---

## Step 3 — Write tests

The tests go where the tree has them (`tests/README.md` says what each folder holds):

- **`tests/unit/`**: the node's contracts. Its typed state and inputs are in `test_mime_node_fields.py`, which also
  requires every input to declare its units; each public method, and that `jax.jit(node.update)` traces.
- **`tests/oracles/`**: its physics, against a closed form or a software oracle, in float64, with a control that
  fails (a planted error the oracle must catch on every run).
- **`tests/differential/`**: a node ported from the 0.2 line, against the port's outputs (`references/`, generated
  once in the port's own environment), with its declared differences and planted mutations as the controls.

**MADDENING's node battery**: every node passes `verify_node`, and `verify_node_order` on each order it declares.
Give the entry that builds it a `NodeCase` in `tests/saved/node_cases.py` (the node walk fails CI without one), with
the `verify_node` options that make a drawn example a valid state and an `OrderLadder` per declared order; a lab
calls `mime.testing.assert_nodes_verified` from its own tests.

**No benchmark registry.** MIME keeps none of its own (`mime.testing`), so a MIME node takes no `MIME-VER` id. A lab
registers its own benchmarks with MADDENING's decorator, under its own id prefix; the decorator's signature is
`verification_benchmark(benchmark_id, description, node_type, benchmark_type, acceptance_criteria, references=())`,
from `maddening.core.compliance.validation`.

**Integration test**: the entry in an experiment, built and run.

---

## Step 4 — Known limitations

If the node has known failure modes, add entries to `docs/validation/known_anomalies.yaml` using the **new-anomaly**
skill; a page or README names the anomaly's id rather than its numbers.

---

## Quick checklist

```markdown
- [ ] `MimeNode` subclass with `initial_state()` and `advance()`, typed state and inputs
- [ ] `update()` is JAX-traceable
- [ ] Every non-smooth operation through `mime.core.switch`, its owners declared, each with a harness row
- [ ] `meta = NodeMeta(...)`: the next free `MIME-NODE-nnn` (from MIME-NODE-104), the equations, `hazard_hints` empty
- [ ] `symbols`: a row per equation symbol, every row pinned in `docs/ratchets/symbol_rows.txt`
- [ ] The entry's `builds`, `gradient` (unless exact), `references`, `family` (interactions)
- [ ] The entry's file `docs/entries/<kind>/<name>.md`, naming the version it describes, with no measured number
- [ ] New refusal codes in `src/mime/build/refusal_table.txt`
- [ ] A `NodeCase` in `tests/saved/node_cases.py`; tests in `tests/unit/` and `tests/oracles/` (an oracle with its
      control), `tests/differential/` for a ported node, and an integration test
- [ ] Known limitations in known_anomalies.yaml
- [ ] Bibliography entries for cited references, each field checked
- [ ] `uv run --no-sync mime docs check` passes, and every CI check
```
