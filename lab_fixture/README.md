# `lab_fixture/`

**A lab's package in miniature**, kept outside `src/mime` on purpose. It registers one interaction, `labx.nudge`, under
its own namespace, from its own distribution (`mime-lab-fixture`), without editing MIME. CI builds and installs it in
every lane, and `tests/test_lab_fixture.py` builds a toy experiment with it beside MIME's toy entries: the control that
registering from outside MIME works. If that test ever needs an edit to MIME to pass, the extension contract is wrong.

Its `pyproject.toml` declares the `mime.entries` entry point for `labx`, which is how a saved experiment that uses
`labx.nudge` finds it when loaded: a load imports that module and nothing the document names. The registry walk
(`tests/saved/`) round-trips an experiment using it through a fresh process, and its lock names this package as the
entry's distribution.

Install it next to MIME with `uv pip install --no-deps ./lab_fixture`.

## Status

dev1, phase 2: one interaction; phase 3: its entry point, and a round-trip case. A lab's own dynamics, field provider, target kind, mapping kind and geometry kind join it
as each becomes registrable.
