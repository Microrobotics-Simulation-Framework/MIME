---
describes: "0.1.0"
---
# A command held constant

**What it models.** The value of a command, the same at every step. An apparatus's `commands` map each command it takes
to the drive that writes it, and a quantity written there is short for this drive: `commands={"frequency": Hz(f)}` is
`commands={"frequency": drives.Constant(Hz(f))}`. The command may be a field provider's (`"frequency"`, `"amplitude"`) or
a chain segment's, named `<segment>.<command>` (`"motor.velocity"`).

Its node, `ConstantNode`, holds the value as a parameter of the graph, one exclusive edge into the command it writes, so a
study can sweep the value without rebuilding the experiment. The setting `value` is a design setting, and it is marked
as a command: its units are those of the command it is written into, checked when the experiment is planned, not by the
node (whose page says its rows' units are declared, not checked, for that reason).

**When not to use it.**
- For a command that must change during a run: this one does not. A controller replaces the drive of each command it
  writes, and is how a command follows a law.
- For a ramp, a pulse or a schedule: this drive is a constant and has none.

**Its derivative** is `exact`, as the generated section states: the command is the parameter, so the derivative of
anything downstream with respect to the value is the derivative of the run with respect to that command.

**Versions.** `drive.constant` 0.1.0 is its first.
