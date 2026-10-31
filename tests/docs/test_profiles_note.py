"""The profiles note's example runs (``docs/profiles.md``): its Python block, exactly as written, builds in exploratory
and leaves pending the four items the note names; and each refusal code the note gives a section is a code of the
refusal table, so the note cannot name a code MIME does not refuse with."""

import re
from pathlib import Path

from mime.build import REFUSAL_TABLE

NOTE = Path(__file__).parents[2] / "docs" / "profiles.md"


def test_the_note_s_example_builds_and_leaves_pending_what_it_says():
    (example, *_) = re.findall(r"```python\n(.*?)```", NOTE.read_text(encoding="utf-8"), flags=re.DOTALL)
    namespace: dict[str, object] = {}
    exec(compile(example, str(NOTE), "exec"), namespace)  # noqa: S102 - the note's own example
    pending = namespace["built"].report.profile.pending
    assert [item.target for item in pending] == ["ball.parts.shell.density", "material 'water' fluid.dynamic_viscosity",
                                                 "drag:ball.spacing", "drag:ball.blob"]


def test_each_code_the_note_gives_a_section_is_a_refusal_code():
    codes = re.findall(r"^### `([a-z.-]+)`$", NOTE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert len(codes) == 10
    assert [code for code in codes if code not in REFUSAL_TABLE] == []
