# Kinematic targets

A kinematic target is one of the three kinds of target whose deliveries are **summed**: **every delivery into a target
of this kind is added to the others**, as it is for a load or an operator. The vocabulary says no more of the kind than
that rule; what a kinematic quantity means for a body, its units, frame and sign, is the row of the interface in the
table above, and the table is empty while no interface of this kind is defined.

**What happens between the nodes.**
1. A law declares a delivery into a target of this kind, by role. The build resolves the role to a thing, finds its
   dynamics, and requires the dynamics to accept the target (`target.not-accepted` otherwise, naming the dynamics that do).
2. The law's node is connected to the dynamics' port by an **additive** edge, so that several laws deliver into one
   port and the port receives their sum. Every edge into a summed target must add: a non-additive edge replaces what
   was summed before it, and the build refuses one (`post-build.additive`).
3. No edge into a target transforms the amount (`post-build.transform`): the dynamics receives what the law computed.
   And no target is also an external input (`post-build.external-input`), which would replace everything delivered.

**What no test shows.** No interface of this kind is defined and no dynamics accepts one, so nothing builds a kinematic
target: the three steps above are the build's code for a summed target, read from the source, and no fixture drives them
for this kind.

**What this page does not say.** Whether the sum of two deliveries of a given interface is a physical quantity is the
interface's to say, not the build's: it adds whatever it is given. Read the row of the interface before delivering two
laws into one port.
