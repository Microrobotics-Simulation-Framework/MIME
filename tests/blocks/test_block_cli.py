"""``mime show`` and ``mime check`` of a block folder, offline: what the block is and needs, and what publishing it
still needs, on the profile workflow. Nothing is installed, downloaded, built or run."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from blocks import block_copy, edited, host, own_cache

from mime.blocks._check import REFERENCED_WORDS
from mime.blocks._show import NOTHING_RUN, shown
from mime.cli import main
from mime.push import check


def test_mime_show_says_what_the_block_is_and_needs_and_that_nothing_was_run(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    assert main(["show", str(folder)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "block labx.nudged_probe 0.1.0, CC-BY-4.0 (open: copied into the document that uses it)"
    assert "  A nudged probe (a lab's block in miniature). Creators: Lab, Demo (Demo Lab)" in printed
    assert "sockets (you bind these): medium (medium)" in printed
    assert "Adds 1 body (probe), 4 laws, 1 declared value." in printed
    assert printed[-1] == NOTHING_RUN
    assert any(line.startswith("  mime-lab-fixture  >=0.1,<0.2  ") and line.endswith("in its range") for line in printed)
    assert main(["show", str(folder), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert (data["id"], data["tier"], data["sockets"]) == ("labx.nudged_probe", "copied", {"medium": "medium"})


def test_mime_show_of_a_block_whose_requirement_is_not_met_names_the_install_line_and_refuses_nothing(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'mime-lab-fixture = ">=0.1,<0.2"', 'mime-lab-fixture = ">=7"')
    lines = shown(folder).lines()
    assert "! its requirements are not all met here: its cases cannot run, and it cannot be used" in lines
    assert "  pip install 'mime-lab-fixture>=7'" in lines


def test_mime_show_draws_the_same_words_on_a_terminal(tmp_path: Path):
    from mime.cli._display import show_block

    stream = io.StringIO()
    show_block(shown(block_copy(tmp_path)), stream)
    drawn = stream.getvalue()
    assert "labx.nudged_probe" in drawn and "mime-lab-fixture" in drawn and NOTHING_RUN in drawn


def test_mime_show_refuses_the_graph_of_a_block(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as stopped:
        main(["show", str(block_copy(tmp_path)), "--graph"])
    assert stopped.value.code == 2 and "--tree, --graph and --out are a saved experiment's" in capsys.readouterr().err


def test_mime_check_lists_the_missing_case_as_a_to_do_and_research_makes_it_a_finding(tmp_path: Path,
                                                                                     capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    assert main(["check", str(folder)]) == 0
    printed = capsys.readouterr().out
    assert "a block's manifest, checked under the exploratory profile (the default: a block declares none): 0 " \
           "findings, 1 to do." in printed and "has no case: `mime pack` refuses a block with none" in printed
    assert main(["check", str(folder), "--profile", "research"]) == 1
    assert "1 finding" in capsys.readouterr().out
    from mime.blocks._check import case_refusals
    from mime.blocks._definition import definition_of

    definition = definition_of(folder)
    assert [(cls, refusal.code) for cls, refusal in case_refusals(definition, definition.content_hash())] == [
        ("needed", "block.no-case")]


def test_mime_check_of_a_block_with_its_case_has_nothing_to_do(tmp_path: Path):
    folder = block_copy(tmp_path)
    experiment, water = host("case")
    experiment.use(folder, medium=water)
    experiment.licence("CC-BY-4.0")
    with own_cache(tmp_path):
        experiment.save(folder / "cases" / "in_water", steps=2)
    checked = check(folder)
    assert checked.findings == [] and checked.of("to do") == [] and checked.notes == ()


def test_mime_check_finds_a_requirement_not_met_and_an_unlisted_entry_as_defects(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'mime-engine = ">=0.3.0.dev1,<0.4"', 'mime-engine = ">=99"')
    assert any("so its entries were not resolved here" in item.what for item in check(folder).findings)
    other = block_copy(tmp_path, "unlisted")
    edited(other, "mime.toml", 'mime-lab-fixture = ">=0.1,<0.2"\n', "")
    assert any("from the distribution mime-lab-fixture, which its [requires] does not name" in item.what
               for item in check(other).findings)


def test_mime_check_tells_the_author_of_a_referenced_block_what_its_tier_cannot_hide(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'licence = "CC-BY-4.0"', 'licence = "LicenseRef-labx-terms"\n\n'
                                                          '[licences.LicenseRef-labx-terms]\ntitle = "Demo terms"\n'
                                                          'url = "https://example.org/terms"')
    checked = check(folder)
    assert checked.notes == (REFERENCED_WORDS,) and REFERENCED_WORDS in checked.text()
    assert "The values its fragment sets appear in every user's saved lock" in REFERENCED_WORDS


def test_mime_check_finds_a_key_of_the_definition_wrong(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'id = "labx.nudged_probe"', 'id = "Nudged Probe"')
    found = check(folder).findings
    assert any("[block] id: 'Nudged Probe' is not <namespace>.<name>" in item.what for item in found), found
