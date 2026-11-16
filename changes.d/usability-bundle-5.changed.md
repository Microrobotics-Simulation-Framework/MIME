- **Usability bundle 5: messages that name the problem once and blame the right party.** A body whose part was refused (a
  solid's radius below zero) is one problem: what follows from it (the body's derived mass, a drag's need of its hull)
  is no longer refused apart, and the part's refusal says so. A run of zero steps, a negative one or one under a step
  is named as such (`run.duration`), and its "do:" offers one run, not the same one twice. A reason that is only a
  placeholder word (`"pending"`, `"TODO"`, `"TBD"`, `"FIXME"`), or starts as a label with one (`"TODO: fill in"`,
  `"TBD - ask"`), is refused with "did you mean `pending()`"; prose that merely begins with the word still passes. An entry's
  or a namespace's version must be a version (numbers joined by dots, PEP 440; `"banana"` stops registering, and so does a text with whitespace around it, which would have been a second key of the
  same version; a second spelling of a registered version, `0.1` beside `0.1.0`, is refused) and versions list in that order
  (`0.9.0` before `0.10.0`). `run()`'s docstring and the README say that a run continues the
  last one. A check of the built graph that faults a node (`maddening.shape`, `maddening.dtype`,
  `maddening.units`, `maddening.error` where an edge reads a field the node's state lacks, `post-build.units`,
  `post-build.input-units`, `post-build.unsaved-arguments`, `run.run-constant`) is blamed on the code that has the flaw,
  the node's class, whatever entry built it: a class outside MIME is no longer worded "MIME defect: report it with this
  build report" but names the node's class and its package, says what its author changes, and the footer says it is not
  MIME's defect; a node of one of MIME's own classes is still MIME's, under a lab's entry too. MADDENING's other errors
  (a coupling group mixed, a non-float loop, a cycle, a missing node) are worded as they were.
