"""``docs/choosing_spacing.md`` prints measured tables; this holds them to the measurements, so that no number on the
page is typed by hand and none can drift from the data it came from.

**What this bounds.** That every table on the page is, character for character, what ``scripts/example_tables.py
render`` makes of ``docs/data/spacing_tables.json``, that the data says what machine it was measured on and the page
says it too, and that the examples' starting sizes are rows of the tables. It does not re-measure anything: the
measurements are the script's output, run by hand on the machine the page names, and are a record, not a test of the
code. **Controls.** A page whose table has one number changed, or one row dropped, is refused; so is data that names no
machine."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "choosing_spacing.md"
DATA = ROOT / "docs" / "data" / "spacing_tables.json"


@pytest.fixture(scope="module")
def script():
    path = ROOT / "scripts" / "example_tables.py"
    spec = importlib.util.spec_from_file_location("example_tables_for_the_page", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measurements():
    return json.loads(DATA.read_text(encoding="utf-8"))


def problems(script, measurements, page: str) -> list[str]:
    found = [f"the table {name!r} is not on the page as the measurements make it"
             for name, text in script.render(measurements).items() if text not in page]
    machine = measurements["runs"]["machine"]
    if not machine.get("cpu") or not machine.get("threads"):
        found.append("the measurements name no machine")
    elif machine["cpu"] not in page:
        found.append(f"the page does not name the machine the measurements were made on, {machine['cpu']!r}")
    return found


def test_every_table_on_the_page_is_what_the_measurements_make(script, measurements):
    assert problems(script, measurements, PAGE.read_text(encoding="utf-8")) == []


def test_the_control_a_changed_number_or_a_dropped_row_or_an_unnamed_machine_is_refused(script, measurements):
    page = PAGE.read_text(encoding="utf-8")
    tables = script.render(measurements)
    first = tables["sphere_spacing"].splitlines()
    changed = page.replace(first[2], first[2].replace("|", "| 9", 1), 1)
    assert any("sphere_spacing" in line for line in problems(script, measurements, changed))
    dropped = page.replace("\n" + tables["screw_spacing"].splitlines()[3], "", 1)
    assert any("screw_spacing" in line for line in problems(script, measurements, dropped))
    unnamed = json.loads(json.dumps(measurements))
    unnamed["runs"]["machine"]["cpu"] = ""
    assert problems(script, unnamed, page) == ["the measurements name no machine"]


def test_the_examples_starting_sizes_are_rows_of_the_tables(measurements):
    sphere = {round(row["spacing_per_radius"], 3) for row in measurements["runs"]["sphere"]}
    screw = {round(row["spacing_mm"], 3) for row in measurements["runs"]["screw"]}
    sphere_dt = {round(row["dt_ms"], 3) for row in measurements["steps"]["sphere"]}
    screw_dt = {round(row["dt_ms"], 3) for row in measurements["steps"]["screw"]}
    spec_sphere = importlib.util.spec_from_file_location("example_sphere_sizes", ROOT / "examples" / "sphere_settling.py")
    spec_screw = importlib.util.spec_from_file_location("example_screw_sizes", ROOT / "examples" / "screw_in_tube.py")
    assert spec_sphere is not None and spec_sphere.loader is not None and spec_screw is not None and spec_screw.loader is not None
    sphere_module, screw_module = (importlib.util.module_from_spec(s) for s in (spec_sphere, spec_screw))
    spec_sphere.loader.exec_module(sphere_module)
    spec_screw.loader.exec_module(screw_module)
    assert 0.25 in sphere and round(sphere_module.STEP_MS, 3) in sphere_dt
    assert round(screw_module.START_SPACING_MM, 3) in screw and round(screw_module.START_DT_MS, 3) in screw_dt
