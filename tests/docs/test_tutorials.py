"""The user guide's tutorials run. ``examples/tutorial_a.py`` and ``examples/tutorial_a0.py``, each in a process of its
own as a user would run it, print exactly the text of the output block the guide shows under them: the guide's output is
this test's captured standard output, and a changed character in either fails it.

**What this bounds.** That the tutorials build and run as written, in the default lane's time budget, and that the guide
shows what they print. They are short runs on purpose, so nothing here claims a physical number: the speed they print is
a short run's, and the guide says so. What the experiments are held to is the differential tests' (``test_script_a.py``,
``test_a0.py``), which this does not repeat.

**Control.** A tutorial named by an output block of the guide that this file does not list fails (a new tutorial gets its
test), and a tutorial listed that no output block names fails (a test over nothing)."""

import re
from pathlib import Path

import pytest

from mime.docs._executable import hand_written, parse, run_script

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = ("examples/tutorial_a.py", "examples/tutorial_a0.py")


def output_blocks() -> dict[str, list[tuple[str, int, str]]]:
    """Each script an output block of the guide names, with the file, line and text of each block that names it."""
    found: dict[str, list[tuple[str, int, str]]] = {}
    for relative in hand_written(ROOT):
        for block in parse((ROOT / relative).read_text(encoding="utf-8")).blocks:
            if block.directive is not None and block.directive.kind == "output":
                found.setdefault(block.directive.argument, []).append((relative.as_posix(), block.line, block.text))
    return found


def test_the_tutorials_named_by_the_guide_are_the_tutorials_this_file_runs():
    assert sorted(output_blocks()) == sorted(TUTORIALS)


@pytest.mark.parametrize("script", TUTORIALS)
def test_a_tutorial_prints_the_text_the_guide_shows_under_it(script):
    printed = run_script(ROOT / script)
    shown = output_blocks()[script]
    assert shown, f"no output block of the guide names {script}"
    for where, line, text in shown:
        assert text.rstrip("\n") == printed.rstrip("\n"), f"{where}:{line} is not what {script} prints"


def a0_distance_excess_and_depth(shown: str, source: str) -> tuple[float, float]:
    """The printed distance's excess over ``ArmTracking``'s standoff, and the screw's starting depth below the tube's
    axis, both in millimetres: the guide explains the first as the second plus a fraction of a millimetre."""
    from mime.control import ArmTracking

    printed = float(re.search(r"the magnet is ([\d.]+) mm from the body", shown).group(1))  # type: ignore[union-attr]
    depth = -float(re.search(r"position=mm\(\(0\.0, 0\.0, (-[\d.]+)\)\)", source).group(1))  # type: ignore[union-attr]
    return printed - 1e3 * ArmTracking.__dataclass_fields__["standoff"].default.value, depth


def test_the_a0_tutorials_distance_is_the_standoff_plus_the_screws_depth_and_a_fraction_of_a_millimetre():
    """The tutorial's explanation of 151.0 mm. The printed text is the guide's output block, which the test above holds
    to what the script prints, so this does not run the script again. The tolerance is the explanation's own words: a
    fraction of a millimetre, taken as under half of one, and the print's rounding."""
    ((_, _, shown),) = output_blocks()["examples/tutorial_a0.py"]
    source = (ROOT / "examples" / "tutorial_a0.py").read_text(encoding="utf-8")
    excess, depth = a0_distance_excess_and_depth(shown, source)
    assert depth == pytest.approx(1.0) and abs(excess - depth) < 0.5, (excess, depth)


@pytest.mark.parametrize("printed", ["150.0", "152.0"])
def test_control_a_distance_the_depth_does_not_explain_fails(printed):
    """A distance that is not the standoff plus the depth is not explained by it; the arms are decided by the printed
    text alone, so they fail on every run."""
    source = (ROOT / "examples" / "tutorial_a0.py").read_text(encoding="utf-8")
    excess, depth = a0_distance_excess_and_depth(f"the magnet is {printed} mm from the body", source)
    assert not abs(excess - depth) < 0.5
