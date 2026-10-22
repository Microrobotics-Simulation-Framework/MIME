# `lab_fixture/`

**A lab's package in miniature**, kept outside `src/mime` on purpose. It registers one interaction, `labx.nudge`, under
its own namespace, from its own distribution (`mime-lab-fixture`), without editing MIME. CI builds and installs it in
every lane, and `tests/test_lab_fixture.py` builds a toy experiment with it beside MIME's toy entries: the control that
registering from outside MIME works. If that test ever needs an edit to MIME to pass, the extension contract is wrong.

Install it next to MIME with `uv pip install --no-deps ./lab_fixture`.

## Status

dev1, phase 2: one interaction. A lab's own dynamics, field provider, target kind, mapping kind and geometry kind join it
as each becomes registrable.
