This document is not a regulatory record or certificate; bounds and estimates are as labelled.

This page counts what MIME's records claim and what its specifications state, and says for each whether a test that CI
runs holds it. It is a count and a table, not a verdict on the models. The claims are those of the known-anomalies
registry (`docs/validation/known_anomalies.yaml`); the statements are the normative sentences of the specifications
(`docs/specification/`).

A **bound** is an upper bound on an error over a stated domain. An **estimate** is a value with an uncertainty that
bounds nothing: a maximum over finitely many points is an estimate, however it was computed. An **acceptance test**
agrees with a physical reference to ground a sign, a frame or a unit, and claims no accuracy, so it is never read as
one. A **limit** is the stated edge of where a model applies. A **rule** is a normative sentence, which has no number
and no oracle, so it has no basis: it prints with the tests that hold it. The groups below keep the roles apart.

Evidence is a test that ran and passed in the default lane. A test marked `skip`, `xfail`, `slow` or `gpu` is not run
by CI's default lane, so it is not evidence, whatever it would have done. The version of this page that also runs the
`slow` and `gpu` tests, and says which device each ran on, is 0.3.0's.
