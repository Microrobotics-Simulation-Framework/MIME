"""The figures the guides say in words, held to the code that holds them. The number rule reads digits, and a figure in
words is invisible to it (its warning sees some, not all), so each spoken figure of the guides that a script or a table
holds is compared here: the step and the frequency of tutorial A, the length of the short runs, and the size of the
tube's table.

**What this bounds.** That the words match the code now. It does not find a new figure spoken in words: a count of
things (``three inputs``) has no cue the number rule's warning could read, and stays a reviewer's.

**Control.** Each comparison takes the page's text and the source's text as arguments, and a planted wrong figure in
either must fail it on every run."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide"
WORDS = {"half a millisecond": 0.5, "ten hertz": 10.0, "one second": 1.0}


def says(page: str, phrase: str) -> bool:
    return phrase in " ".join(page.split())


def default_of(source: str, name: str, unit: str) -> float:
    """The default ``name=unit(value)`` of ``script_a``'s signature."""
    found = re.search(rf"^def script_a\([^\n]*\b{name}={unit}\(([\d.]+)\)", source, re.MULTILINE)
    assert found is not None, f"script_a has no default {name}={unit}(...)"
    return float(found.group(1))


def short_run(source: str) -> float:
    found = re.search(r"^SHORT = s\(([\d.]+)\)", source, re.MULTILINE)
    assert found is not None
    return float(found.group(1))


def figure_problems(a: str, a0: str, script_a: str, tutorial_a: str, tutorial_a0: str) -> list[str]:
    found: list[str] = []
    if says(a, "half a millisecond") != (default_of(script_a, "dt", "ms") == WORDS["half a millisecond"]):
        found.append("tutorial A says 'half a millisecond' and script_a's default step is not that (or the reverse)")
    if says(a, "ten hertz") != (default_of(script_a, "frequency", "Hz") == WORDS["ten hertz"]):
        found.append("tutorial A says 'ten hertz' and script_a's default frequency is not that (or the reverse)")
    if says(a0, "one second") != (short_run(tutorial_a0) == WORDS["one second"]):
        found.append("tutorial A0 says 'one second' and its short run is not that (or the reverse)")
    if short_run(tutorial_a) != WORDS["one second"]:
        found.append("tutorial A's short run is not one second, which its prose and output say")
    return found


def files() -> tuple[str, str, str, str, str]:
    return ((GUIDE / "tutorial_a.md").read_text(encoding="utf-8"), (GUIDE / "tutorial_a0.md").read_text(encoding="utf-8"),
            (ROOT / "examples" / "script_a.py").read_text(encoding="utf-8"),
            (ROOT / "examples" / "tutorial_a.py").read_text(encoding="utf-8"),
            (ROOT / "examples" / "tutorial_a0.py").read_text(encoding="utf-8"))


def test_the_figures_the_tutorials_say_in_words_are_the_scripts():
    a, a0, script_a, tutorial_a, tutorial_a0 = files()
    assert says(a, "half a millisecond") and says(a, "ten hertz") and says(a0, "one second"), "the words moved"
    assert figure_problems(a, a0, script_a, tutorial_a, tutorial_a0) == []


@pytest.mark.parametrize("plant", ["step", "frequency", "run"])
def test_a_wrong_figure_in_the_page_or_the_script_fails_every_run(plant):
    a, a0, script_a, tutorial_a, tutorial_a0 = files()
    if plant == "step":
        script_a = script_a.replace("dt=ms(0.5)", "dt=ms(0.4)", 1)
    elif plant == "frequency":
        a = a.replace("ten hertz", "twelve hertz")
    else:
        tutorial_a0 = tutorial_a0.replace("SHORT = s(1.0)", "SHORT = s(2.0)")
    assert figure_problems(a, a0, script_a, tutorial_a, tutorial_a0) != []


def test_the_size_the_install_page_gives_the_tube_table_is_the_assets_size_to_the_nearest_hundred_megabytes():
    from mime import assets

    text = (GUIDE / "install.md").read_text(encoding="utf-8")
    assert "unit_tube_R0.997" in text and says(text, "about two hundred megabytes")
    assert round(assets.known()["unit_tube_R0.997"].size / 1e8) == 2
