---
describes: "0.1.0"
---
# A first-order lag

**What it models.** The last rung of a command port, `CommandPort(response=FirstOrderLag(tau=...))`: what the
receiver gets trails what the port's amplifier drives as a first-order lag of time constant `tau`, solved exactly over
each step for a value held through it, `y + (y_prev - y) exp(-dt / tau)`. A coil's current behind its amplifier (its
L/R, `tau = L / R`, a signal model with no back-EMF and no mutual inductance) and a servo's joint behind its command are
each one. It starts at the port's rest.

**Where it is computed.** For the same declaration, where its receiver says. A coil array's currents are lagged in the
port's stage (`ports:<apparatus>`), after the port's rate, latency, bounds and slew. A kinematic arm lags its joints in
its own node, its servo, so a port on its `joint_positions` or `joint_displacements` hands the arm the time constant
and builds no stage for it: the program is the one the arm's servo built before the saved form held ports, which an
earlier form's saved folder shows, hash for hash, and the lag has one implementation per receiver. The arm lags every joint with one time
constant, so the lag is declared on each position command it has, with one `tau`, or refused (`port.declaration`). A
port on an arm that also declares a latency or a slew stages those first, its bounds the joints' stated limits, so that
the arm's own clip and lag act on a value already within them, in the ladder's order.

**When not to use it.**
- For a coil whose back-EMF, mutual inductance or eddy currents matter: the coil array's circuit rung, which arrives in
  a later release. For two lags in series: a seam of the port.

**Its derivative** is `exact`: the lag is linear in what it is driven to and in its own value.

**Versions.** `response.first_order_lag` 0.1.0 is its first.
