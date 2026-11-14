# Ports: the links between a controller and what it drives

An apparatus's ports say how a command reaches what it drives and how a measurement reaches the controller that reads
it: at what rate, how late, within what bounds, how fast it may move, and through what response. A port is data on the
apparatus (`Apparatus(ports={...})`), its numbers each declared with the reason they come from. A port left out, or one
at every default, is ideal and builds nothing: an experiment with no port runs the program it ran before, its derived
hash and its bits the same.

## Command ports

`mime.CommandPort`, keyed by the command it carries (`"currents"`, `"arm.joint_positions"`), takes its writer's value
through these rungs, in this order, the order of an amplifier chain:

1. **sample**: the writer is sampled every `rate` and held between samples;
2. **delay**: what was sampled arrives `latency` later, the port's `rest` before it;
3. **bound**: beyond its `bounds` it is clipped (`on_bound="clip"`), or the port trips (`on_bound="trip"`) and delivers
   its rest for the rest of the run, flagged and warned. A modelled trip is not any hardware's protection;
4. **slew**: it moves at most `slew` a second from what the amplifier drove last;
5. **response**: it reaches the receiver through its response (`mime.responses.FirstOrderLag(tau=...)`, a coil's L/R
   lag, a servo).

A rate and a latency are whole numbers of base steps. A kinematic arm's port takes its joints' stated limits as its
bounds where it declares none, and the arm's node computes a first-order lag on its position commands itself: declare
the same lag on each of them.

## Measurement ports

`mime.Measurement`, keyed by a name of its own, measures a thing's state (`of="sphere.pose"`) or an observation's output
(`of="camera:.expected"`): sampled every `rate`, `latency` late, with seeded Gaussian `noise` and a `quantum`. A
controller reads it as `<apparatus>.<port>`, and the age of what it reads, in base steps, as `<apparatus>.<port>.age`;
the build report classes both reads `measurement`. A camera's image already carries its sensor's model, so noise on it
is refused; on the camera's expected image it is allowed, and the report says the sensor model is bypassed.

## What a build shows

The build report's `ports` lines name each port and what it does; its controller block gives the loop's chain, the base
steps from a state to the first step whose command was made from it. Each apparatus with staged ports gains at most two
nodes, `ports:<apparatus>` and `measure:<apparatus>`, and their fixed cost a step (on a GitHub runner, A0's chunk 1.15
times as long, a coil-array sphere's 1.68 times). A run records a port's traffic when asked: `record=` names the stage's
outputs (`ports:coils.currents`, `ports:coils.written_currents`, `measure:coils.tracker_age`).

A saved experiment holds its ports (saved form 15). The details, and every refusal by its code, are in
[the build's notes](../../src/mime/build/README.md).
