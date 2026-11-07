---
describes: "0.1.0"
---
# The energy ledger

**What it does.** `exp.observe(EnergyLedger())` keeps an account of energy across the experiment's couplings. Attached,
it has the build meter each body: each law then delivers into an input of its own on the body, and the body records,
inside its own step, each law's work on it, the energy it stores and its scheme's own numerical terms, and checks its
step's energy identity against rounding at every step. It adds no node and reads nothing of its own; without it, no
program changes, bit for bit.

The build report's `energy` lines say, for each law, what kind of exchange it is (dissipative, potential, or unmetered
with the law's reason) and what is outside the components, worked out from the graph: a thing that receives no load
from them, such as a field turning on its own or a drive. A run's result gives the account in plain words,
`result.energy.text()`: what each body stores, what each dissipative law lost, what was supplied (named by what supplies
it, taken from the dissipation side: what was lost plus the change in what is stored) and each potential's change; with
`details=True`, each law's work on each body, the scheme's own terms, the work the supplied laws paired in the steps,
and the checks: the step identity's worst step against its rounding, the work that arrived unattributed (exactly zero
where every delivery is a law's) and the books' balance against its rounding. `result.energy.efficiency(useful)` is a
useful work as a share of what was supplied.

**When not to use it.**
- As a measurement of a model's accuracy: it checks that the books close and that each law is the class it says it is;
  a drag scaled wrong, a coupling's sign flipped or a wrong parameter passes every check (each family's oracles are
  what see those).
- To find a wrong delivery between laws: the build compares its graph with the deliveries it made and refuses a
  swap, a duplicate or a missing delivery by name (`post-build.ledger-inputs`). The books cannot see them, and a law's
  residual shows only a swap that involves gravity and a moving load.
- On a body whose dynamics is not metered yet (a chain segment's): the build report lists it, with the reason, and the
  account leaves it out.
- On a run read back from a folder: the account needs each law's parameters, which only the built experiment holds.

**Its derivative** passes through the account's sums. Called inside a function being differentiated,
`mime.results.energy_account` forms `supplied_total`, `efficiency(useful)` and each potential's change from the final
state and the parameters by traced arithmetic, so that a propulsive efficiency's derivative in a drive's parameter, taken
through the compiled step with the ledger attached, agrees with central differences. A `result.energy` read from a
finished run holds numbers (and its `text()` formats them); take the derivative of the function that forms the account.

**What `supplied` is.** The account takes it from the dissipation side, an estimate under the ledger's exact identity,
not a bound: it differs from the work the laws paired in the steps (the paired supply, in `text(details=True)`) by
the scheme's own terms (small on script A, large on a case at a coarse step), and which of the two lies nearer the
continuum value is not established at that step. Shared among several things outside, it is a ratio of their paired
works, unbounded where they nearly cancel.

**Versions.** `energy_ledger` 0.1.0 is its first.
