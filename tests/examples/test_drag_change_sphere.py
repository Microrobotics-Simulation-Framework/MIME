"""``examples/drag_change_sphere.py``: the four runs the development record's line on drag 0.4.0 states, as the example
runs them.

**What this bounds.** That the numbers the line writes are the ones the repository gives at the settings it names
(``spacing=um(20)``, ``blob=0.5``, a 75 ms run read from 50 ms): each pairing's settling speed to the line's four
significant figures, and the change from drag 0.3.0 on the sphere 0.1.0 to drag 0.4.0 to its tenth of a per cent. It
says nothing of which is nearer the truth: the sphere settles outside the rung's declared range."""

import importlib.util
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: Each pairing's settling speed (m/s), as the line writes it.
STATED = {"drag 0.4.0, sphere 0.2.0": 0.03565, "drag 0.4.0, sphere 0.1.0": 0.03565,
          "drag 0.3.0, sphere 0.1.0": 0.03647, "drag 0.3.0, sphere 0.2.0": 0.03620}
LINE = "from 0.03647 to 0.03565 m/s (-2.3 %"


@cache
def example():
    spec = importlib.util.spec_from_file_location("example_drag_change_sphere",
                                                  ROOT / "examples" / "drag_change_sphere.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def speed(name: str) -> float:
    return example().speed(*example().RUNS[name])


@pytest.mark.parametrize("name", list(STATED))
def test_each_pairing_settles_at_the_speed_the_line_writes(name):
    assert round(speed(name), 5) == STATED[name]


def test_the_change_the_line_writes_is_the_runs_and_the_line_writes_it():
    change = speed("drag 0.4.0, sphere 0.2.0") / speed("drag 0.3.0, sphere 0.1.0") - 1.0
    assert round(100.0 * change, 1) == -2.3
    record = (ROOT / "docs" / "development" / "changes_0.3.0.dev1.md").read_text(encoding="utf-8")
    assert LINE in record and "examples/drag_change_sphere.py" in record
