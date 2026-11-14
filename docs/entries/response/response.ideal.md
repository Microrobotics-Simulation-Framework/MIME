---
describes: "0.1.0"
---
# A command delivered as driven

**What it models.** The last rung of a command port, `CommandPort(response=...)`, that changes nothing: what the port's
amplifier drives, after its rate, latency, bounds and slew, is what the receiver gets, the same step. It is the response
of a port that declares none, so `CommandPort(latency=...)` delays the command and shapes it no further.

It builds no node of its own: the port's stage (`ports:<apparatus>`) holds every rung of the apparatus's ports, and a
port whose every rung is ideal builds no stage at all.

**When not to use it.**
- For an amplifier or a coil whose output trails what it is driven to: `responses.FirstOrderLag`, a coil's L/R or a
  servo's time constant.

**Its derivative** is `exact`: the value delivered is the value driven.

**Versions.** `response.ideal` 0.1.0 is its first.
