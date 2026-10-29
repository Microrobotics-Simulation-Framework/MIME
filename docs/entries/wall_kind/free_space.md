---
describes: 0.1.0
---

# `free_space`

No wall: a medium that fills no region, unbounded, so a solver adds nothing to its free-space kernel. It is the
trivial analytic wall kind, and the one a body far from every wall is read in.

**When not to use it.** For a medium that fills a region (a tube's lumen): a wall left out changes the flow, so the
experiment is refused with it, naming the wall kinds that match the region.

**Versions.** The first, landed with the wall-kind interface in phase 7h; phase 7g's free-space boundary-element
method took the same setting.
