# `lab_fixture/`

**A lab's package in miniature**, kept outside `src/mime` on purpose. It registers an interaction, `labx.nudge`, under
its own namespace, from its own distribution (`mime-lab-fixture`), without editing MIME. CI builds and installs it in
every lane, and `tests/test_lab_fixture.py` builds a toy experiment with it beside MIME's toy entries: the control that
registering from outside MIME works. If that test ever needs an edit to MIME to pass, the extension contract is wrong.

Its `pyproject.toml` declares the `mime.entries` entry point for `labx`, which is how a saved experiment that uses
`labx.nudge` finds it when loaded: a load imports that module and nothing the document names. The registry walk
(`tests/saved/`) round-trips an experiment using it through a fresh process, and its lock names this package as the
entry's distribution.

Install it next to MIME with `uv pip install --no-deps ./lab_fixture`.

## Status

dev1, phase 2: one interaction; phase 3: its entry point, and a round-trip case; phase 7a2: a chain segment with no
URDF, `labx.gantry`, a linear stage described by its own settings, publishing the same frame poses as MIME's
segments, on whose carriage a magnet mounts unchanged; phase 7k: an observation, `labx.height`, which declares its
outputs as MIME's own observations do (a body's height each step, its integral at the end), and gets the same result
accessors and page with no edit to MIME. A lab's own field provider, target kind, mapping kind and
geometry kind join it as each becomes registrable.
