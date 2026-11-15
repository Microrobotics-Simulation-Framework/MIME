---
describes: "0.1.0"
---
# The field a body is exposed to

**What it measures.** Three quantities of the magnetic field at a body, each step, read where a law bound to the body
reads the field (its magnet's centre, as the magnetic laws 0.2.0 read it), with the maximum of each over the run and a
flag per quantity against a limit the experiment declares: `Exposure(umr, field="coils.magnetic",
flux_density=mT(40, reason="..."), field_gradient=T_per_m(1, reason="..."), rate_of_change=T_per_s(2,
reason="..."))`. Its outputs, by name:

- `flux_density`, `|B|` (T);
- `field_gradient`, the Frobenius norm of the gradient `G_ij = dB_i/dx_j` (T/m), `sqrt(sum_ij G_ij^2)`; zero in a
  uniform field, which has none;
- `rate_of_change`, `|dB/dt|` (T/s), estimated as `|B_n - B_(n-1)| / dt`: a one-step difference of the flux density at
  the body's point. Because the point moves with the body, it holds the body's motion through the field as well as the
  field's own change (the derivative along the body's path, not at a point fixed in space). It is an estimate, with an
  error of order `dt`: the field's second derivative along the path times the step. In a field turning uniformly at `f`
  it reads the chord, `2 B sin(pi f dt) / dt`, against `2 pi f B`;
- at the end of the run, `max_flux_density`, `max_field_gradient` and `max_rate_of_change`, the largest of each.

**Step 0's rate of change is not measured.** No reading exists before step 0, so its `rate_of_change` is not measured
and is held at 0: it raises no flag and sets no maximum. A field on from the start is never seen switching on; one
switched on between step 0 and step 1 is seen at step 1, as `|B_1| / dt`, and flagged there.

**The limits and the flags.** Each limit is optional and is a quantity carrying its reason (`mT(40, reason="where the
number comes from")`): a limit with no reason, a pending one, or one not above zero is refused
(`exposure.declaration`). A declared limit gives its quantity a flag, `above_flux_density`, `above_field_gradient` or
`above_rate_of_change`, raised at each step where the quantity is above the limit; a run reports each flag's count, its
first and its last step, as every flag. A limit is a number the experiment declares, and a flag says only that a modelled
quantity passed it: it bounds nothing about what the modelled field does to anything, and makes no claim beyond the
model and the number.

**Where it reads the field.** A field that varies in space is read from the slot a law bound to the body already reads
at the same point (the same pose and the same point fixed in it), so it adds no slot to the provider: the provider's node
and the physics' derived hash are what they are without the observation, and the build report names the law it reads
beside. A uniform field's flux density is read where a law reads it (its gradient is zero). Where no law reads the field
at the body, the exposure is refused (`observation.field-read`): a point no law reads needs the observation's own
instance of the provider, which is later work (the route named in the refusal: attach the law the field acts through at
that body, or observe the state the field acts on).

**The rate's rounding in float32.** The rate is a difference of two consecutive readings over `dt`, so the rounding of
each reading enters as its size over `dt`: noise of order `eps |B| / dt` (`eps` float32's), and more where the provider's
sources cancel (a coil array's field is a sum of sources that cancel to a small share `c` of their size, and rounds at
`eps / c` of itself). Shrinking `dt` makes it larger, while the estimate's own error shrinks with it. On a sphere
carried through a coil array's field at a millisecond step (`tests/oracles/test_exposure.py`), the float32 rate misses
`|G v|` (the gradient times the point's velocity, which the rate estimates) by several per cent at a step, where the
float64 rate of the same run misses it by a few parts in ten thousand: the excess is rounding, and the test holds it
within the noise's scale and above the float64 miss. Three consequences:
- **A flag limit near the true rate flickers in float32**: `above_rate_of_change` is raised at some steps and not at
  their neighbours, for a limit within the noise of the rate (several per cent of it at that step size). Declare the
  limit with the margin the noise needs, or run the experiment in float64.
- **`max_rate_of_change` reads high**: a maximum of noisy readings is biased upwards, by the noise's size.
- **Computing the difference in float64 inside the node would not help**: the two readings arrive already rounded to
  float32, their difference is exact there, and the same difference taken in float64 from the same stored readings is
  the node's rate (measured, equal to a part in ten million). The rounding is in the provider's field, which float64
  (the run's dtype) removes. The flux density and its gradient have no such noise: it is the difference that amplifies
  it.

**When not to use it.**
- At a point no law reads: refused, as above.
- For the field's change at a point fixed in space while the body moves: the rate of change is along the body's path.
- For a rate of change faster than the step resolves: a one-step difference cannot see a change within a step, and its
  error grows with the field's second derivative along the path.
- Acoustic quantities: none is modelled; they arrive with the first acoustic provider.
- For a claim: it is a number the run produced, against a number the experiment declared.

**Its derivative** is piecewise: each magnitude has a kink at zero, where its derivative is taken as zero
(`exposure.norm`), and each maximum takes the derivative of the step that sets it (`exposure.maximum`), each declared
in its gradient record. The flags are reports no edge reads.
