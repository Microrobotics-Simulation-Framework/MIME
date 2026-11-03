# Equality constraints

An equality constraint is a target a dynamics holds to a value a law prescribes for one coordinate, rather than a
load it adds up (the fixture the build's tests use prescribes a body's velocity along one axis). The rule that sets it apart is that **a target of this kind takes at most one
delivery.** Two laws that both said what the same coordinate must be would disagree, and a sum of two such statements is
not either of them.

**What happens between the nodes.**
1. A law declares a delivery into a target of this kind, by role, as it does into a load. The build resolves the role to
   a thing, finds the thing's dynamics, and requires the dynamics to accept the target; one that does not is refused
   (`target.not-accepted`), naming the dynamics that do.
2. The build then looks, over every law, for a second delivery into the same port of the same dynamics. Two or more are
   refused at plan time (`target.exclusive`), naming each law that delivers, before any node exists.
3. The law's node is connected to the dynamics' port by an edge that is **not additive**: the value delivered replaces
   what the port held, where a load is added to it. That replacement is what makes a second delivery dangerous (it would
   silently displace the first), and why the plan refuses it rather than leaving the last edge to win.
4. After the graph is built, the same rule is checked again on the edges the graph actually holds
   (`post-build.exclusive`), together with the rules every delivery obeys: that no edge into a target transforms the
   amount (`post-build.transform`), and that no target is also an external input, which would replace everything
   delivered into it (`post-build.external-input`).

**What this page does not say.** What the delivered value means, its units, its frame and its sign, is the
interface's own, in the vocabulary table above, and how a dynamics holds its state to it is that dynamics' node page.
The rule here is only the exclusivity.
