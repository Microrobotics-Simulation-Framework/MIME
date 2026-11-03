"""``docs/choosing_spacing.md`` prints measured tables; this holds them to the measurements, so that no number on the
page is typed by hand and none can drift from the data it came from.

**What this bounds.** That every table on the page is, character for character, what ``scripts/example_tables.py
render`` makes of ``docs/data/spacing_tables.json``, and every number the prose states of them is the phrase ``facts``
makes of it (whitespace aside), that the screw's sample counts the prose explains are the kind's own, that the data says what machine it was measured on and the page
says it too, and that the examples' starting sizes are rows of the tables. It does not re-measure anything: the
measurements are the script's output, run by hand on the machine the page names, and are a record, not a test of the
code. **Controls.** A page whose table has one number changed, or one row dropped, is refused; so is data that names no
machine."""

import importlib.util
import json
import re
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
    flowed = " ".join(page.split())
    found += [f"the phrase {name!r} is not on the page as the measurements make it: {text!r}"
              for name, text in script.facts(measurements).items() if " ".join(text.split()) not in flowed]
    machine = measurements["runs"]["machine"]
    if not machine.get("cpu") or not machine.get("threads"):
        found.append("the measurements name no machine")
    elif machine["cpu"] not in page:
        found.append(f"the page does not name the machine the measurements were made on, {machine['cpu']!r}")
    runner = measurements.get("ci", {}).get("machine", {})
    if runner and runner["cpu"] not in page:
        found.append(f"the page does not name CI's runner, {runner['cpu']!r}")
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


def test_a_changed_number_in_the_prose_is_refused(script, measurements):
    page = PAGE.read_text(encoding="utf-8")
    facts = script.facts(measurements)
    stated = " ".join(facts["sphere_to_a_per_cent"].split())
    flowed = " ".join(page.split())
    assert stated in flowed
    first = re.search(r"\d+(\.\d+)?%", stated)
    assert first is not None
    changed = flowed.replace(stated, stated.replace(first.group(0), "99.9%", 1), 1)
    assert any("'sphere_to_a_per_cent'" in line for line in problems(script, measurements, changed))


def test_the_screws_samples_the_prose_explains_are_the_kinds_own():
    """The default's row differs from the 0.83 mm row because the grid has ``ceil(length / spacing)`` stations along the
    axis: the page says 16 round the turn, 9 and 10 stations, and two caps of 31, so 206 and 222 samples."""
    from mime.scene import Screw
    from mime.units import dimensionless, mm

    kind = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
                 starts=2)
    held = {"radius": 1.56e-3, "length": 7.47e-3, "wavenumber": 2.33, "modulation": 0.33}
    data = {round(row["spacing_m"] * 1e3, 3): row["samples"] for row in
            json.loads(DATA.read_text(encoding="utf-8"))["coarse"]["rows"]}
    for spacing, around, stations, caps in ((0.83e-3, 16, 9, 31), (0.829e-3, 16, 10, 31)):
        assert kind.sample_counts(held, spacing, "lateral") == (around, stations)
        assert kind.sample_counts(held, spacing, "caps") == (caps,)
        assert around * stations + 2 * caps == data[round(spacing * 1e3, 3)]
