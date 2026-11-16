# Units

**A value in MIME is a number with its unit, checked when the experiment is planned.** A script writes a quantity with
its unit, and the build checks the quantity's dimension against what the setting needs and turns it into a plain SI float.
Nothing after the build's first phase sees a quantity: nodes, `params` and JAX arrays hold SI numbers only. A bare number
is never taken for a quantity of some unit; it is refused with the quantity to write.

<!-- mime-fragment: examples of what is written, not a run -->
```python
from mime.units import mm, ms, mm_per_s

radius = mm(3.175, reason="the tube's datasheet")   # a quantity, held in SI, with its reason beside it
step = ms(20)
speed = result[ball].speed                          # a result's quantity may hold an array
speed.to(mm_per_s)                                  # read in a unit of its dimension
```

## Quantities

A unit makes a quantity, and `quantity.to(unit)` reads one back in a unit of its dimension. A quantity may carry its reason
beside its value; equality and arithmetic ignore it. Results come back as quantities too, and may hold arrays (a series of
a run's speeds); an array quantity is a result and never a declaration.

- **Arithmetic** keeps the dimension: sums and orders need one dimension, products and ratios combine them.
- **Powers, absolute values and sums.** An integer power raises the value and the dimension together (a whole number
  written as a float is the same). A fractional power is refused, because a dimension's exponents are whole numbers: read the
  magnitude in a unit with `.to(unit)`, take its root, and write the result in its own unit. `abs` of a quantity is its
  absolute value in the same dimension, elementwise for a series; a vector quantity has no one absolute value and is
  refused. `sum` of quantities works because Python's `sum` starts at the `int` zero, and a quantity accepts that on the
  left of a `+` and returns itself. **Exactly the Python `int` zero is accepted there, and nothing else**: a float zero,
  `False`, a NumPy zero (a scalar or a zero-dimensional array) and an array of zeros are refused as any bare number is, and so
  is a quantity plus the int zero. An array of numbers plus a quantity is that refusal too, not an array of one quantity per
  number. The start is the one bare number accepted so that `sum` works, and it opens no hole in `+`.
- **`==` with a bare number is `False`, not an error.** A quantity compared for equality with a bare number is `False` (and
  `!=` is `True`), where an order (`<`) with a bare number is refused. Python's equality must not raise without breaking
  containers and dataclass comparison, which ask `==` of what they hold, so MIME leaves it. Write the comparison in a unit,
  with `.to(unit)`, or against a quantity of the same dimension.
- **A frequency and an angular velocity share a dimension and differ by two pi.** A quantity in `Hz` counts turns a second
  and one in `rad_per_s` radians a second; a quantity knows which it counts, and a site that takes one refuses the other with
  the conversion to write. `rpm` counts turns like `Hz`, and `deg_per_s` counts radians like `rad_per_s`. A power of a
  quantity that counts an angle is refused beyond the first and the reciprocal, since a quantity counts one angle at most.

<!-- mime-fragment: examples of what is written, not a run -->
```python
m(2.0) ** 2            # 4.0 m^2: an integer power
abs(m(-2.0))           # 2.0 m
sum([m(1.0), m(2.0)])  # 3.0 m: sum starts at the int 0
0 + m(1.0)             # 1.0 m: the int zero, and only it
m(1.0) + 0             # refused, as is 0.0 + m(1.0), False + m(1.0) and np.zeros(3) + m(1.0)
m(1.0) == 1.0          # False, silently: compare in a unit instead
```

## The units

Every unit is a name in `mime.units`; its symbol is what it prints with, and `units_by_name()` gives them by name. The units
added with the label grammar, with their values in SI:

<!-- mime-fragment: examples of what is written, not a run -->
```text
name       symbol  value in SI        counts
kPa        kPa     1e3 Pa
MPa        MPa     1e6 Pa
mmHg       mmHg    133.322387415 Pa   the conventional millimetre of mercury: a column of 13595.1 kg/m^3 at 9.80665 m/s^2, 1 mm high
mL         mL      1e-6 m^3           a cubic centimetre
cP         cP      1e-3 Pa s          the centipoise, exactly one millipascal second
rpm        rpm     1/60 Hz            turns
deg_per_s  deg/s   pi/180 rad/s       radians
```

## Unit labels

A **label** is the text MIME writes a dimension's unit as in a saved form, in the vocabulary's interfaces, in a node's
state and in its symbol table. It holds a dimension and no scale factor, so it is written in SI. `parse_unit` reads every
label MIME writes, and `unit_symbol` writes a dimension as it always did.

The grammar:

- **names**: the SI base symbols and the named units, `m kg s A K mol cd` and `N Pa J W T Hz rad`;
- the number **one** for a dimensionless value;
- a **product** by spaces (any whitespace; a factor never runs into the next, so `m1` and `m(s)` are refused, and `ms` is
  no name here: it is the millisecond, a prefixed unit);
- one **quotient** by `/` and a single factor after it. More than one term in the denominator is written in parentheses, so
  a label is never read two ways;
- **integer powers** by `^`, written right after what they raise and never zero;
- **parentheses**.

<!-- mime-fragment: examples of what is written, not a run -->
```text
kg m^2        a product            N m s        a product of named units
m/s^2         a quotient           N s/m        a quotient of a product
m/(s Pa)      a parenthesised denominator
(m s)^2       a group raised to a power          m^-1 kg s^-2   a negative power
1             dimensionless      1/s       a rate that says neither turns nor radians
J/mol K       refused: more than one term after the /, write J/(mol K)
m; 1          a composite: a pose is a length and a dimensionless part
N s/m; N s; N m s   a composite: a wrench's operator, its blocks in order
1/(rad s)     refused: it counts its radian minus one time, and is no rad/s
rad^2/s       refused: it counts its radian twice
s^-1/rad      refused: minus one time
1/(Hz s^2)    refused: minus one time
m1            refused: a factor never runs into the next, write m 1 or m
```

A label with a semicolon is a composite, its parts separated (a pose, a wrench's operator: the examples below).
`parse_label` reads the parts apart. A prefixed unit is no name here, because a label has no scale; it is a quantity. A label
MIME cannot read is refused with the code `units.label`, naming the character where it stopped and, for a name that is close
to one, the name it is close to.

**A turn and a radian stay apart.** `Hz` reads as `FREQUENCY` and `rad/s` as `ANGULAR_VELOCITY`: equal as dimensions, and
different in what they count, as the quantities and the sites that take them keep them. `rad` alone is dimensionless, as the
unit `rad` holds it. A label that counts both turns and radians is refused, and so is a rate per second that counts its
angle other than once (the examples below): such a label is never read as `rad/s` or `Hz`.

### Labels from dimension markers

The markers an entry declares its settings with (`Length`, `Force`, `DIMENSIONLESS`, and the rest) are labels too:
`Units(Length)` is the label for a length, and `Units(Length, DIMENSIONLESS)` a pose's, stored as that string, so a saved
form holds the same text either way and an editor completes the markers. `mime.units.unit_label(...)` writes the same
string for a place that holds a string, such as an interface's `units=`. An angular velocity marker is written as `rad/s`
and a frequency as `Hz`.

<!-- mime-fragment: markers written as the labels they stand for; not a run -->
```python
Units(Length)                 # the label "m"
Units(Length, DIMENSIONLESS)  # the label "m; 1", a pose's
unit_label(Length, DIMENSIONLESS)
```

### Where a label is checked

A label is read **where it is defined**: an interface of the vocabulary when it is made, at import; a symbol-table row
(`Symbol(...)`, at import, unless it says its units are prose, `prose_units=True`); a node's `node_input(...)` and
`param_spec(...)` where the node declares them; and a state's `Units(...)` when the annotation holding it is evaluated, which
for a module that postpones its annotations is when the node's typed state is first read (a build, a save, or the tests that
read every node's). A typo, or a value that is not text (a number given to `Units`), is the refusal `units.label` and not a silent label
that nothing parses. A registration's interface carries no values, so its units may be empty. A test collects
every label MIME writes from the registry, the node classes, the vocabulary and the source, and parses them all.

## Errors

A unit error is a MIME error with a code and what to do, and still a `ValueError`: reading a length in a time, `speed.to(s)`,
is refused with the code `call.units` and the line that says to read it in a unit of length, and `except ValueError`,
`except UnitError` and `except MimeError` all catch it. Its message, `str(error)`, is the plain sentence the sites that quote
it have always read. The refusal's lines are the error's notes, made when it is raised, which Python prints under the
message; they are the exception's own notes, so `add_note` keeps working, and so does what Hypothesis adds to a failing
example.

**Where it is raised decides the refusal it is.** Raised from a script's own expression, a unit error reaches the script
with its own code. Raised in code MIME is checking or reading, it is that code's failure and takes the code a plain
`ValueError` from there takes: in an entry's own methods (`setting_problems`, `build`, `plan`, a stage's) it is the entry's
defect, `entry.defect`; in a controller's `law()` the same; in a results manifest it is `result.format`. A unit bug inside a
lab's entry never reads as the user's mistake.

## On the command line

`mime run --override path=value` and a study's spec file read the same value text: numbers separated by commas and a unit,
written as the unit's **name** in `mime.units`, its **symbol** or SI base units. A bare number is a dimensionless value, and
a setting that needs a unit refuses it. The unit's spaces are kept, so a unit written with them is a product (the first
three lines below, each judged against the setting's dimension), and never the prefixed unit that its letters would spell
without them. A unit written without a space is read as written, and one with spaces that is a unit's symbol as printed
is that unit.

<!-- mime-fragment: examples of what is written, not a run -->
```text
--override field.commands.frequency=20Hz        a unit's name
--override water.viscosity=2mPa_s               a name with an underscore
--override umr.initial.position=0,0,-1mm        components
--override water.viscosity="2 mPa s"            a unit's symbol
--override water.viscosity="1e-3 kg m^-1 s^-1"  SI base units
--override x="3 m s"      metre seconds, never 3 ms
--override x="3 m m"      square metres, never 3 mm
--override x="2 m T"      tesla metres, never 2 mT
--override x=3ms          no space: the millisecond, as written
--override x=2mPas        no space: the symbol of the unit mPa s with its space left out
--override x="3 N m"      a unit's symbol as printed
```
