---
describes: "0.1.0"
---
# Two openings into one shared distal resistance

**What it models.** Two openings each through its own resistance into one shared distal resistance to a distal
pressure of zero: two outlets draining into one downstream bed, `exp.attach(SharedDistal(first=Pa_s_per_m3(...),
second=Pa_s_per_m3(...), distal=Pa_s_per_m3(...)), medium=blood, surface=(tree.end_0, tree.end_1))`, `first` at the
first opening named and `second` at the other. Each opening's pressure is its own resistance times its flux plus the
distal resistance times the two fluxes together, so the law couples its openings (a block over them, with
off-diagonals), and the flow solves the block whole each step. `surface=` names its two openings as a tuple or as a
region's openings (`surface=tube.lumen`); given another count it is refused (`binding.opening-count`). Dissipative:
its block is symmetric and positive definite for positive settings, and with the energy ledger its own books (its
loss, every joule it receives) close at every step under its name.

**When not to use it.**
- Where each outlet drains to its own bed: a law at each opening (a `Windkessel`) holds that.
- Where the shared bed's compliance matters: this law holds resistances only.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `shared_distal` 0.1.0 is its first.
