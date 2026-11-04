# Scope and limits

What MIME declares of itself, and what it does not claim: the list of what the code states, so that you can judge whether
an experiment of yours is inside it.

## What MIME claims, and what it does not

- **A declared limit is a statement, not a test.** An entry's `validity` lines say where its model holds. The build prints them
  as *declared, not yet checked* ([the build report](build_report.md#validity)), and the checks the build does compute are
  listed apart (`checked`). A limit nothing tests is not a limit that was measured.
- **Agreement with a reference is not accuracy.** Where an experiment is held to the earlier line's outputs, as A and A0 are
  ([tutorial A](tutorial_a.md), [tutorial A0](tutorial_a0.md)), the test bounds how far MIME 0.3 is from the earlier line at the operating
  points it runs (A's one, and A0's two frequencies). The earlier code is a weak oracle, so it says nothing about the flow
  or about hardware.
- **A bound and an estimate are different things.** A bound is an upper limit on an error over a stated domain; a
  measured maximum over some points is an estimate of it, not a bound. A comparison with measurements from hardware bounds
  the model's error from above and includes the measuring apparatus's own. Where MIME states a number of this kind, it names
  the claim record that holds it (`see MIME-ANO-114` is one), and does not repeat the number in prose.
- **A first look is not a converged value.** In the `exploratory` profile a numerical setting you did not choose takes a stated default, which the profile
  records as pending; `research` refuses it ([profiles](../profiles.md)), and `docs/choosing_spacing.md` prices and judges the
  default for the MRS drag rung.

## The profiles

A profile says how much the build asks of an experiment. It never relaxes a validity limit or an input error. The
definitions below are generated from the code's own, and the table is the rules it applies:

<!-- mime-generate: profiles -->

`strict` holds a higher evidence standard and confers nothing by itself. It is refused by name until it is built, because a
strict build now could apply only research's rules and would claim a standard it does not apply.

## The limits MIME's entries declare

Each entry declares where its model holds. These are the lines of MIME's own entries, as their entry pages print them; the
numbers in them are judged where they come from, on the entry's page.

<!-- mime-generate: limits -->

## What is not covered

- A lab's entries declare their own limits; MIME prints them, and counts what a lab leaves unstated, and judges none of it
  ([the lab-author guide](lab_author.md)).
- The drag surrogate's weights are the earlier line's and inherit the old wall tables' defect (MIME-ANO-117, in
  `docs/validation/known_anomalies.yaml`). Experiments that use them are held to a reference on the same weights, not to the
  physics.
- Platforms other than Linux are verified within the lock's tolerances and by smoke test, not bit for bit
  (`docs/platforms.md`).
